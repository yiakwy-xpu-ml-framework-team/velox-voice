"""VeloxVoice kernel pack.

Layout mirrors the flash-float-jit-kernel conventions:
  helper_cuda.py   platform/arch detection & -gencode flags (sm_90 / sm_121)
  utils.py         csrc path resolution, source hashing, cached TVM-FFI builders
  csrc/*.cu        versioned CUDA kernels (TVM-FFI exported)
  triton3_7/       gluon reference kernels for quick verification
  metal/           mx.fast.metal_kernel wrappers (PR-style Metal kernels)
  jit/             FlashFloatJitKernel base + inline CUDA (TVM-FFI) + Metal wrappers
  ops/             stable python entry points used by models/frontends
"""

from .helper_cuda import compute_capability, cuda_arch_str, cuda_available
from .jit import CACHE_ROOT, CudaJitKernel, FlashFloatJitKernel, MetalJitKernel

__all__ = [
    "CACHE_ROOT",
    "CudaJitKernel",
    "FlashFloatJitKernel",
    "MetalJitKernel",
    "cuda_arch_str",
    "compute_capability",
    "cuda_available",
]
