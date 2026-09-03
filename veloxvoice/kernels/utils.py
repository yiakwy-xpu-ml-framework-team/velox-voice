"""Shared kernel-pack utilities: roots, hashing, cached TVM-FFI builders."""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path

from veloxvoice.kernels.helper_cuda import cuda_build_flags

KERNELS_ROOT = Path(__file__).resolve().parent
CSRC = KERNELS_ROOT / "csrc"
BUILD_CACHE = Path.home() / ".cache" / "veloxvoice" / "tvmffi"

import os as _os

if _os.environ.get("VELOXVOICE_KERNEL_CACHE"):
    BUILD_CACHE = Path(_os.environ["VELOXVOICE_KERNEL_CACHE"]) / "tvmffi"


def read_source(rel: str) -> str:
    return (KERNELS_ROOT / rel).read_text(encoding="utf-8")


def source_key(name: str, sources: list[str], flags: list[str], arch: str) -> str:
    h = hashlib.sha256()
    h.update(f"veloxvoice:{name}\n".encode())
    for s in sources:
        h.update(s.encode())
    h.update(" | ".join(flags).encode())
    h.update(arch.encode())
    return h.hexdigest()[:16]


@functools.cache
def build_cuda_module(
    name: str,
    csrc_files: tuple[str, ...],
    functions: tuple[str, ...],
    extra_cuda_cflags: tuple[str, ...] = (),
    arch_override: str | None = None,
):
    """Compile csrc/*.cu files into one TVM-FFI module (disk-cached, process-cached).

    Returns `tvm_ffi.Module`; entry points are `mod.<function>` callables that take
    torch tensors zero-copy (DLPack).
    """
    import tvm_ffi  # noqa: F401
    from tvm_ffi.cpp import load_inline

    from veloxvoice.kernels import helper_cuda

    sources = tuple(read_source(f"csrc/{f}") for f in csrc_files)
    if arch_override:
        flags = list(extra_cuda_cflags)  # replace arch flags entirely
        arch = arch_override
    else:
        flags = cuda_build_flags() + list(extra_cuda_cflags)
        arch = helper_cuda.cuda_arch_str()
    key = source_key(name, list(sources), flags, arch)
    build_dir = BUILD_CACHE / f"{name}-{key}"
    build_dir.mkdir(parents=True, exist_ok=True)

    prev = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    if arch_override:
        os.environ["TVM_FFI_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=f"velox_{name}",
            cuda_sources=list(sources),
            functions=[],  # entry points are self-exported via TVM_FFI_DLL_EXPORT_TYPED_FUNC
            extra_cuda_cflags=flags,
            extra_ldflags=[f"-L/usr/lib/aarch64-linux-gnu", "-lcuda"],
            build_directory=str(build_dir),
        )
    finally:
        os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
        if prev is not None:
            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = prev
