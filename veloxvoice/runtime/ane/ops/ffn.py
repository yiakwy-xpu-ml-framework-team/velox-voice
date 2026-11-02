"""fused FFN row-split across the ANE and Metal.

The feature is off unless ``ENABLE_ANE_OFFLOAD`` is truthy.  

When enabled, the top rows of a two-layer GELU MLP run in a Core ML 
fp16 mlprogram on the ANE, while the remaining rows run in MLX on the GPU.  
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import pathlib
import platform
import re
import warnings
from multiprocessing import shared_memory
from typing import Callable

import mlx.core as mx
import numpy as np

_TRUTHY = {"1", "true", "yes", "on"}


def enabled() -> bool:
    return os.environ.get("ENABLE_ANE_OFFLOAD", "").strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def offload_rows() -> int:
    return _env_int("ANE_OFFLOAD_ROWS", 1500)


def _cache_dir() -> pathlib.Path:
    raw = os.environ.get("ANE_OFFLOAD_CACHE")
    path = pathlib.Path(raw).expanduser() if raw else (
        pathlib.Path.home() / ".cache" / "veloxvoice" / "ane_ffn")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)[:80] or "default"


def _build_mlp(rows: int, hidden: int, inter: int, path: pathlib.Path,
               w1: np.ndarray, b1: np.ndarray,
               w2: np.ndarray, b2: np.ndarray) -> None:
    import torch
    from ..core import convert_module

    net = torch.nn.Sequential(
        torch.nn.Linear(hidden, inter),
        torch.nn.GELU(),
        torch.nn.Linear(inter, hidden),
    )
    with torch.no_grad():
        net[0].weight.copy_(torch.from_numpy(w1))
        net[0].bias.copy_(torch.from_numpy(b1))
        net[2].weight.copy_(torch.from_numpy(w2))
        net[2].bias.copy_(torch.from_numpy(b2))
    example = torch.randn(rows, hidden)
    convert_module(net, (example,), path, input_names=["x"], output_name="y",
                   compute_units="ane", precision="float16")


def _worker(shm_in: str, shm_out: str, cmd_recv, res_send,
            rows: int, hidden: int) -> None:
    import coremltools as ct

    si = shared_memory.SharedMemory(name=shm_in)
    so = shared_memory.SharedMemory(name=shm_out)
    
    x = np.ndarray((rows, hidden), dtype=np.float32, buffer=si.buf)
    y = np.ndarray((rows, hidden), dtype=np.float32, buffer=so.buf)
    cache: dict[str, object] = {}
    res_send.send("ready")
    while True:
        cmd = cmd_recv.recv()
        if cmd == "stop":
            break
        try:
            model = cache.get(cmd)
            if model is None:
                model = ct.models.MLModel(cmd,
                                          compute_units=ct.ComputeUnit.CPU_AND_NE)
                cache[cmd] = model
            y[...] = model.predict({"x": x})["y"]
            res_send.send(None)
        except Exception as exc:  # pragma: no cover - defensive
            res_send.send(repr(exc))
    si.close()
    so.close()


class ANEFusedFFN:
    """Persistent ANE worker for fused ``fc1 -> GELU -> fc2`` row splits."""

    def __init__(self, hidden: int, rows: int | None = None,
                 cache_dir: str | os.PathLike | None = None,
                 tag: str = "default"):
        if not enabled():
            raise RuntimeError("ENABLE_ANE_OFFLOAD is not set")
        if platform.machine() != "arm64":
            raise RuntimeError("ANE offload requires Apple silicon")
        if mp.current_process().daemon:
            raise RuntimeError("ANE offload cannot run inside a daemon process")
        parent = getattr(mp, "parent_process", lambda: None)()
        if parent is not None:
            raise RuntimeError("ANE offload must start from the main process")
        self.hidden = int(hidden)
        self.rows = int(rows if rows is not None else offload_rows())
        if self.rows <= 0:
            raise ValueError("ANE_OFFLOAD_ROWS must be positive")
        self.tag = _safe_tag(tag)
        self.cache = pathlib.Path(cache_dir) if cache_dir else _cache_dir()
        self.cache.mkdir(parents=True, exist_ok=True)
        self._paths: dict[str, str] = {}
        self._closed = False

        self._si = shared_memory.SharedMemory(
            create=True, size=self.rows * self.hidden * 4)
        self._so = shared_memory.SharedMemory(
            create=True, size=self.rows * self.hidden * 4)
        self._x = np.ndarray((self.rows, self.hidden), dtype=np.float32,
                             buffer=self._si.buf)
        self._y = np.ndarray((self.rows, self.hidden), dtype=np.float32,
                             buffer=self._so.buf)
        ctx = mp.get_context("spawn")
        self._cmd_recv, self._cmd_send = ctx.Pipe(duplex=False)
        self._res_recv, self._res_send = ctx.Pipe(duplex=False)
        self._proc = ctx.Process(
            target=_worker,
            args=(self._si.name, self._so.name, self._cmd_recv,
                  self._res_send, self.rows, self.hidden),
        )
        self._proc.start()
        if not self._res_recv.poll(180) or self._res_recv.recv() != "ready":
            self._proc.terminate()
            self.close()
            raise RuntimeError("ANE worker did not become ready")
        atexit.register(self.close)

    def _model_path(self, key: str, w1: mx.array, b1: mx.array,
                    w2: mx.array, b2: mx.array) -> str:
        path = self._paths.get(key)
        if path is not None:
            return path
        inter, hidden = w1.shape
        if hidden != self.hidden:
            raise ValueError(f"expected hidden {self.hidden}, got {hidden}")
        filename = f"{self.tag}_{key}_{self.rows}x{hidden}x{inter}.mlpackage"
        target = self.cache / filename
        if not target.exists():
            _build_mlp(self.rows, hidden, inter, target,
                       np.array(w1.astype(mx.float32)),
                       np.array(b1.astype(mx.float32)),
                       np.array(w2.astype(mx.float32)),
                       np.array(b2.astype(mx.float32)))
        path = str(target)
        self._paths[key] = path
        return path

    def __call__(self, key: str, x: mx.array,
                 w1: mx.array, b1: mx.array, w2: mx.array, b2: mx.array,
                 gpu_fn: Callable[[mx.array], mx.array]) -> mx.array:
        if self._closed:
            return gpu_fn(x)
        if x.ndim != 2 or x.shape[0] <= self.rows:
            return gpu_fn(x)
        path = self._model_path(key, w1, b1, w2, b2)
        self._x[...] = np.array(x[:self.rows].astype(mx.float32))
        self._cmd_send.send(path)
        y_gpu = gpu_fn(x[self.rows:])
        mx.eval(y_gpu)
        status = self._res_recv.recv()
        if status is not None:
            raise RuntimeError(f"ANE worker failed: {status}")
        y = mx.concatenate(
            [mx.array(self._y), y_gpu.astype(mx.float32)], axis=0)
        mx.eval(y)
        return y.astype(x.dtype)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._cmd_send.send("stop")
        except Exception:
            pass
        self._proc.join(timeout=30)
        if self._proc.is_alive():
            self._proc.terminate()
        for shm in (self._si, self._so):
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


def maybe_create(hidden: int, tag: str) -> ANEFusedFFN | None:
    if not enabled():
        return None
    try:
        return ANEFusedFFN(hidden, tag=tag)
    except Exception as exc:
        warnings.warn(f"ANE offload disabled: {exc}", RuntimeWarning,
                      stacklevel=2)
        return None
