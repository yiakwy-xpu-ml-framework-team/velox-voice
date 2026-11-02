"""Standalone benchmark for the Metal JIT kernels.

This is a thin wrapper around ``veloxvoice.kernels.ops.metal_ops`` that keeps
the shapes close to the real SheetSage2 workloads, so a speedup number is not
hidden by an unrepresentative 512-element shape.  It compares the Metal JIT
implementation with the native MLX fallback for every kernel.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from veloxvoice.kernels.ops import metal_ops as mo


@dataclass
class BenchRow:
    name: str
    jit_ms: float
    jit_min_ms: float
    ref_ms: float
    ref_min_ms: float
    speedup: float


def bench(fn, *, warmup: int = 20, rep: int = 100) -> tuple[float, float]:
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(rep):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples), min(samples)


def _row(name: str, jit_fn, ref_fn, *, warmup: int, rep: int) -> BenchRow:
    jit_ms, jit_min = bench(jit_fn, warmup=warmup, rep=rep)
    ref_ms, ref_min = bench(ref_fn, warmup=warmup, rep=rep)
    return BenchRow(name=name, jit_ms=jit_ms, jit_min_ms=jit_min,
                    ref_ms=ref_ms, ref_min_ms=ref_min,
                    speedup=ref_ms / max(jit_ms, 1e-9))


def _ln_case(m: int, d: int):
    x = mx.random.normal((m, d)).astype(mx.bfloat16)
    w = mx.random.normal((d,)).astype(mx.bfloat16)
    b = mx.random.normal((d,)).astype(mx.bfloat16)
    return (
        f"fused_layernorm {m}x{d}",
        lambda x=x, w=w, b=b: mo.fused_layernorm(x, w, b),
        lambda x=x, w=w, b=b: mx.fast.layer_norm(x, w, b, 1e-5),
    )


def _glu_case(m: int, c: int):
    x = mx.random.normal((m, 2 * c)).astype(mx.bfloat16)
    a, b = mx.split(x, 2, axis=-1)
    return (
        f"silu_glu {m}x{c}",
        lambda x=x: mo.silu_glu(x),
        lambda a=a, b=b: a * mx.sigmoid(b),
    )


def _mel_case(t: int, m: int):
    power = mx.abs(mx.random.normal((t, m))).astype(mx.float32) + 1e-3
    mean = mx.random.normal((m,)).astype(mx.float32)
    std = mx.abs(mx.random.normal((m,))).astype(mx.float32) + 0.5
    floor = mx.array(1e-10, dtype=power.dtype)
    std_floor = mx.array(1e-5, dtype=power.dtype)

    def ref(power=power, mean=mean, std=std, floor=floor, std_floor=std_floor):
        y = 10.0 * mx.log10(mx.maximum(power, floor))
        return (y[:-1] - mean) / mx.maximum(std, std_floor)

    return (f"mel_log_norm {t}x{m}", lambda power=power, mean=mean, std=std:
            mo.mel_log_norm(power, mean, std), ref)


def _gemv_case(n: int, k: int):
    x = mx.random.normal((k,)).astype(mx.bfloat16)
    w = mx.random.normal((n, k)).astype(mx.bfloat16)
    return (
        f"gemv {n}x{k}",
        lambda x=x, w=w: mo.gemv(x, w),
        lambda x=x, w=w: (x[None] @ w.T)[0],
    )


def _gemv_batch_case(b: int, n: int, k: int):
    x = mx.random.normal((b, k)).astype(mx.bfloat16)
    w = mx.random.normal((n, k)).astype(mx.bfloat16)
    return (
        f"gemv_batch {b}x{n}x{k}",
        lambda x=x, w=w: mo.gemv_batch(x, w),
        lambda x=x, w=w: x @ w.T,
    )


def _argmax_cases(v: int = 31678, k: int = 512):
    logits = mx.random.normal((v,)).astype(mx.float32)
    allowed = mx.random.normal((v,)) > 0.5
    yield (
        "masked_argmax",
        f"masked_argmax {v}",
        lambda logits=logits, allowed=allowed: mx.array(
            mo.masked_argmax(logits, allowed)),
        lambda logits=logits, allowed=allowed: mx.argmax(
            mx.where(allowed, logits,
                     mx.array(float("-inf"), dtype=logits.dtype))),
    )

    x = mx.random.normal((k,)).astype(mx.bfloat16)
    w = mx.random.normal((v, k)).astype(mx.bfloat16)
    yield (
        "lm_head_argmax",
        f"lm_head_argmax {v}x{k}",
        lambda x=x, w=w, allowed=allowed: mx.array(
            mo.lm_head_argmax(x, w, allowed)),
        lambda x=x, w=w, allowed=allowed: mx.argmax(
            mx.where(allowed, (x[None] @ w.T)[0],
                     mx.array(float("-inf"), dtype=mx.bfloat16))),
    )


def cases(ops: list[str]) -> list[tuple[str, object, object]]:
    out: list[tuple[str, object, object]] = []
    if "fused_layernorm" in ops:
        out.extend(_ln_case(m, d) for m, d in ((512, 1024), (7500, 1024)))
    if "silu_glu" in ops:
        out.extend(_glu_case(m, c) for m, c in ((512, 1024), (7500, 1024)))
    if "mel_log_norm" in ops:
        out.extend(_mel_case(t, m) for t, m in ((500, 128), (7500, 128)))
    if "gemv" in ops:
        out.extend(_gemv_case(n, k) for n, k in ((512, 512), (31678, 512)))
    if "gemv_batch" in ops:
        out.extend(_gemv_batch_case(b, n, k)
                   for b, n, k in ((8, 512, 512), (8, 31678, 512)))
    wanted = set(ops)
    for op, name, jit_fn, ref_fn in _argmax_cases():
        if op in wanted:
            out.append((name, jit_fn, ref_fn))
    return out


ALL_OPS = ["fused_layernorm", "silu_glu", "mel_log_norm", "gemv",
           "gemv_batch", "masked_argmax", "lm_head_argmax"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ops", default=",".join(ALL_OPS),
                    help="comma separated subset of " + ",".join(ALL_OPS))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rep", type=int, default=100)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    wanted = [op.strip() for op in args.ops.split(",") if op.strip()]
    unknown = [op for op in wanted if op not in ALL_OPS]
    if unknown:
        raise SystemExit(f"unknown ops: {unknown}")

    rows: list[BenchRow] = []
    for name, jit_fn, ref_fn in cases(wanted):
        row = _row(name, jit_fn, ref_fn, warmup=args.warmup, rep=args.rep)
        rows.append(row)
        print(f"{row.name:26s} jit {row.jit_ms:8.3f} ms "
              f"(min {row.jit_min_ms:8.3f})  "
              f"mlx {row.ref_ms:8.3f} ms (min {row.ref_min_ms:8.3f})  "
              f"speedup {row.speedup:5.2f}x")

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(row) for row in rows], f, indent=2)
        print(f"report {args.json}")


if __name__ == "__main__":
    main()
