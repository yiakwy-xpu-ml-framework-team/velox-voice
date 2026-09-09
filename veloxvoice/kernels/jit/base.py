"""flash-float-style JIT kernel base.

Lineage / contract (the "flash-float-jit-kernel" pattern):
  * Kernels are *named, precision-templated, shape-templated* units.
    Cache key = sha256(name | source | template config | toolchain arch).
    Compiled once, cached on disk, reused verbatim thereafter.
  * Each kernel exposes `__call__` in the backend's native array type plus a
    pure-python / framework fallback for CPU-only environments and tests.
  * Kernels never own scratch state: everything shape-static is a template
    parameter, everything shape-dynamic is passed as scalars.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from pathlib import Path

CACHE_ROOT = Path(
    os.environ.get("VELOXVOICE_KERNEL_CACHE")
    or Path.home() / ".cache" / "veloxvoice" / "kernels"
)


class FlashFloatJitKernel(ABC):
    name: str = "abstract"
    version: int = 1

    def cache_key(self, extra: str = "") -> str:
        h = hashlib.sha256()
        h.update(f"{self.name}:v{self.version}\n".encode())
        h.update(self.source().encode())
        h.update(extra.encode())
        return h.hexdigest()[:16]

    def cache_dir(self, extra: str = "") -> Path:
        d = CACHE_ROOT / f"{self.name}-{self.cache_key(extra)}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @abstractmethod
    def source(self) -> str:
        """Backend source text of the kernel."""

    @abstractmethod
    def build(self, **template_kwargs):
        """Compile (or fetch from cache) and return the callable kernel."""
