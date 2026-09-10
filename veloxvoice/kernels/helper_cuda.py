"""CUDA platform detection and compile flags (flash-float-jit-kernel conventions).

Targets: Hopper sm_90 / DGX-Spark GB10 sm_121. Flags are -gencode entries plus
opt flags; arch string is part of every kernel cache key.
"""

from __future__ import annotations

# GB10 (DGX Spark, sm_121) kernel packs are written against the 'a'-suffix
# feature set; every other target uses the plain detected arch.
_GB10_ARCH = "12.1"


# TODO (yiakwy) : remove, prefer _is_cuda variable
def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# TODO (yiakwy) : remove, prefer ARCH variable
def compute_capability(device: int = 0) -> tuple[int, int]:
    import torch

    return torch.cuda.get_device_capability(device)


def cuda_arch_str(device: int = 0) -> str:
    """e.g. '12.1' for GB10, '9.0' for Hopper."""
    return ".".join(str(x) for x in compute_capability(device))


def velox_arch_str(device: int = 0) -> str:
    """Arch override for kernel packs: '12.1a' on GB10, native arch elsewhere."""
    arch = cuda_arch_str(device)
    return f"{arch}a" if arch == _GB10_ARCH else arch


def cuda_driver_ldflags() -> list[str]:
    """Link flags locating libcuda across aarch64 (DGX Spark) / x86_64 hosts."""
    import ctypes.util
    import glob
    import platform

    for libdir in (
        f"/usr/lib/{platform.machine()}-linux-gnu",
        "/usr/lib64",
        "/usr/local/cuda/lib64/stubs",
    ):
        if glob.glob(f"{libdir}/libcuda.so*"):
            return [f"-L{libdir}", "-lcuda"]
    soname = ctypes.util.find_library("cuda")  # e.g. 'libcuda.so.1'
    return [f"-l:{soname}"] if soname else ["-lcuda"]


def cuda_gencode_flag(device: int = 0) -> str:
    arch = cuda_arch_str(device).replace(".", "")
    return f"-gencode=arch=compute_{arch},code=sm_{arch}"


def cuda_build_flags(device: int = 0) -> list[str]:
    return ["-O3", "--use_fast_math", "-std=c++17", cuda_gencode_flag(device)]


def is_hopper_or_newer(device: int = 0) -> bool:
    return compute_capability(device)[0] >= 9
