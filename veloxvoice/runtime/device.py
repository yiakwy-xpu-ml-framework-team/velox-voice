"""Platform detection and backend selection.

Supported targets:
  - "cuda-sm90"   : H800 / Hopper
  - "cuda-sm121"  : DGX-Spark (GB10)
  - "metal"       : Apple silicon (Mac Studio M5 Ultra / M3 Ultra / M3 Mini)
  - "cpu"         : fallback for tests
"""

from __future__ import annotations

import os
import platform as _plat
from dataclasses import dataclass

import numpy as _np  # noqa: F401  (kept: detection must stay lightweight otherwise)

_backend_cache: "PlatformInfo | None" = None


@dataclass(frozen=True)
class PlatformInfo:
    name: str  # cuda-sm90 | cuda-sm121 | cuda-generic | metal | cpu
    module_backend: str  # "torch" | "mlx"
    device: str  # torch device string or mlx device type
    compute_capability: tuple[int, int] | None = None
    mlx_gpu: bool = False
    has_cuda_graphs: bool = False
    has_metal_graphs: bool = False  # compiled-segment piecewise execution

    @property
    def is_accelerated(self) -> bool:
        return self.module_backend != "mlx" or self.mlx_gpu


def _detect() -> PlatformInfo:
    forced = os.environ.get("VELOXVOICE_BACKEND")
    if forced == "mlx":
        return _detect_metal()
    if forced == "torch":
        return _detect_torch()

    if _plat.system() == "Darwin" and _plat.machine() == "arm64":
        return _detect_metal()
    try:
        info = _detect_torch()
        if info.name != "cpu":
            return info
    except Exception:
        pass
    return (
        _detect_metal()
        if _plat.system() == "Darwin"
        else PlatformInfo(name="cpu", module_backend="torch", device="cpu")
    )


def _detect_torch() -> PlatformInfo:
    import torch

    if not torch.cuda.is_available():
        return PlatformInfo(name="cpu", module_backend="torch", device="cpu")
    cap = torch.cuda.get_device_capability(0)
    name = {9: "cuda-sm90", 12: "cuda-sm121"}.get(cap[0], "cuda-generic")
    return PlatformInfo(
        name=name,
        module_backend="torch",
        device="cuda:0",
        compute_capability=cap,
        has_cuda_graphs=True,
    )


def _detect_metal() -> PlatformInfo:
    try:
        import mlx.core as mx  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("mlx backend requested but mlx is not installed") from e
    return PlatformInfo(
        name="metal",
        module_backend="mlx",
        device="gpu",
        mlx_gpu=True,
        has_metal_graphs=True,
    )


def detect_platform(refresh: bool = False) -> PlatformInfo:
    global _backend_cache
    if _backend_cache is None or refresh:
        _backend_cache = _detect()
    return _backend_cache


def get_backend() -> str:
    """Return "torch" or "mlx" — the active python-module backend."""
    return detect_platform().module_backend


def get_device_array_lib():
    """Return the array library of the active backend (torch or mlx.core)."""
    return (
        __import__("torch") if get_backend() == "torch" else __import__("mlx.core").core
    )
