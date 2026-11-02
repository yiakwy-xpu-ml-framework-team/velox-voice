"""Apple Neural Engine bridge for the MLX runtime lane."""

from .core import (
    ANEExecutor,
    benchmark,
    convert_module,
    convert_mlx_linear,
    is_available,
)

__all__ = [
    "ANEExecutor",
    "benchmark",
    "convert_module",
    "convert_mlx_linear",
    "is_available",
]
