"""Concurrent ANE + Metal split for one GEMM

GPU rows run on MLX; ANE rows run on a Core ML mlprogram in a background
thread.  

The split is chosen from measured per-row throughput, then the actual 
overlap is measured.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

import mlx.core as mx
import numpy as np

from bench_ane_gemm_roofline import bench, build_ane_linear, roofline


def _cached_weight(path: Path, K: int, N: int) -> tuple[np.ndarray, bool]:
    weight_path = path.with_suffix(".weight.npy")
    if weight_path.exists():
        return np.load(weight_path), True
    w_np = np.random.default_rng(0).standard_normal((N, K)).astype(np.float32)
    np.save(weight_path, w_np)
    return w_np, False


def run_split(M, K, N, ane_rows, reps=10):
    import coremltools as ct

    path = Path(f"/tmp/ane_split_{ane_rows}x{K}x{N}.mlpackage")
    w_np, weight_cached = _cached_weight(path, K, N)
    if not weight_cached or not path.exists():
        build_ane_linear(ane_rows, K, N, path, weight=w_np)
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)

    x = mx.random.normal((M, K)).astype(mx.float16)
    w = mx.array(w_np.astype(np.float16))
    mx.eval(x, w)
    x_np = np.array(x.astype(mx.float32))

    def once():
        out = np.empty((M, N), dtype=np.float32)
        err = {}

        def ane_job():
            try:
                err["ane"] = model.predict({"x": x_np[:ane_rows]})["y"]
            except Exception as exc:  # pragma: no cover
                err["error"] = exc

        t = threading.Thread(target=ane_job)
        t.start()
        y_gpu = x[ane_rows:] @ w.T
        mx.eval(y_gpu)
        t.join()
        if "error" in err:
            raise err["error"]
        out[:ane_rows] = err["ane"]
        out[ane_rows:] = np.array(y_gpu.astype(mx.float32))
        return out

    ref = np.array((x @ w.T).astype(mx.float32))
    got = once()
    max_err = float(np.max(np.abs(got - ref)))
    ms, mn = bench(once, 3, reps)
    return ms, mn, max_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--splits", default="0.0,0.25,0.37,0.5")
    args = ap.parse_args()

    r = roofline(args.M, args.K, args.N)
    print(f"{args.M}x{args.K}x{args.N}  GPU SOL {r['t_sol_ms']:.3f} ms "
          f"({r['bound']})  AI {r['ai']:.1f}")

    for s in [float(v) for v in args.splits.split(",")]:
        ane_rows = max(1, min(args.M - 1, int(round(args.M * s))))
        if s <= 0:
            x = mx.random.normal((args.M, args.K)).astype(mx.float16)
            w = mx.random.normal((args.N, args.K)).astype(mx.float16)
            mx.eval(x, w)
            ms, mn = bench(lambda: mx.eval(x @ w.T))
            print(f"  ANE  0.0%  GPU-only {ms:.3f} ms (min {mn:.3f})")
            continue
        ms, mn, err = run_split(args.M, args.K, args.N, ane_rows)
        print(f"  ANE {ane_rows/args.M:5.1%}  split {ms:.3f} ms "
              f"(min {mn:.3f})  max|err| {err:.4f}  "
              f"vs SOL {r['t_sol_ms']/ms:.1%}")


if __name__ == "__main__":
    main()
