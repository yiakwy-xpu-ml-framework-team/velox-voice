"""CUDA JIT kernels via TVM-FFI (apache-tvm-ffi).

Stream rule: every call into a TVM-FFI module must run inside
`tvm_ffi.use_torch_stream(...)` so TVMFFIEnvGetStream resolves to torch's
current stream — otherwise kernels escape CUDA-graph capture and replay
garbage. See graphs/cuda_runner.py for the capture-time usage.

Production export path per the kernel-library conventions:
`TVM_FFI_DLL_EXPORT_TYPED_FUNC` + `tvm_ffi.cpp.load_inline`, compiled for the
exact device arch (-gencode compute_XX,code=sm_XX), disk-cached by
sha256(name | source | flags | arch).

For one-shot inline sources use this class; for structurally versioned kernels add
a .cu file under `veloxvoice/kernels/csrc/` and a wrapper in `kernels/ops/`.
"""

from __future__ import annotations

import os

from ..helper_cuda import cuda_arch_str, cuda_build_flags
from .base import FlashFloatJitKernel


class CudaJitKernel(FlashFloatJitKernel):
    """One named inline CUDA kernel pack built + cached through TVM-FFI."""

    def __init__(
        self,
        name: str,
        cuda_src: str,
        functions,
        extra_cuda_cflags: tuple[str, ...] = ("-O3", "--use_fast_math"),
    ):
        self.name = name
        self._cuda_src = cuda_src
        self._functions = tuple(functions)
        self._extra_flags = tuple(extra_cuda_cflags)
        self._mod = None

    def source(self) -> str:
        return self._cuda_src

    def build(self, **_):
        if self._mod is None:
            from tvm_ffi.cpp import load_inline

            from ..utils import BUILD_CACHE, source_key

            flags = list(self._extra_flags) + cuda_build_flags()[2:]  # keep dup-free
            arch = cuda_arch_str()
            key = source_key(self.name, [self._cuda_src], flags, arch)
            d = BUILD_CACHE / f"{self.name}-{key}"
            d.mkdir(parents=True, exist_ok=True)
            self._mod = load_inline(
                name=f"velox_{self.name}",
                cuda_sources=[self._cuda_src],
                functions=[],  # self-exported via TVM_FFI_DLL_EXPORT_TYPED_FUNC
                extra_cuda_cflags=flags,
                build_directory=str(d),
                verbose=os.environ.get("VELOXVOICE_JIT_VERBOSE", "0") == "1",
            )
        return self._mod

    def __getattr__(self, item):
        return getattr(self.build(), item)
