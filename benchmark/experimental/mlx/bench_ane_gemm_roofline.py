"""Roofline + GPU/ANE split study for fp16 GEMM on Apple silicon.

The script is to find the *SOL* for a shape and the split ratio that minimizes 
max(t_gpu(M_g), t_ane(M_a)). 

The ANE path is a Core ML mlprogram (CPU_AND_NE); the GPU path is MLX.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np

GPU_TFLOPS = 9.0       # M4 Pro, conservative dense fp16
GPU_GBS = 273.0
ANE_TOPS = 38.0
RIDGE = GPU_TFLOPS * 1e12 / (GPU_GBS * 1e9)


def roofline(M, K, N, peak=GPU_TFLOPS, bw=GPU_GBS):
    flops = 2 * M * N * K
    nbytes = 2 * (M * K + K * N + M * N)
    t_c = flops / (peak * 1e12)
    t_m = nbytes / (bw * 1e9)
    return {"M": M, "K": K, "N": N, "flops": flops, "bytes": nbytes,
            "ai": flops / nbytes, "t_compute_ms": t_c * 1e3,
            "t_mem_ms": t_m * 1e3, "t_sol_ms": max(t_c, t_m) * 1e3,
            "bound": "compute" if t_c > t_m else "memory"}


def bench(fn, warmup=10, rep=30):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3, ts[0] * 1e3


def build_ane_linear(M, K, N, path, weight=None):
    import torch
    import coremltools as ct

    lin = torch.nn.Linear(K, N, bias=False)
    with torch.no_grad():
        if weight is None:
            lin.weight.normal_(0, 0.02)
        else:
            lin.weight.copy_(torch.from_numpy(
                np.asarray(weight, dtype=np.float32)))
    ex = torch.randn(M, K)
    traced = torch.jit.trace(lin.eval(), ex)
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(M, K))],
        outputs=[ct.TensorType(name="y")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    model.save(str(path))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="4096x4096x4096,4096x16384x4096,8x512x2048,8x512x31678")
    ap.add_argument("--ane", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = []
    for spec in args.shapes.split(","):
        M, K, N = (int(v) for v in spec.lower().split("x"))
        r = roofline(M, K, N)
        r_ane = roofline(M, K, N, peak=ANE_TOPS)
        print(f"\n{M}x{K}x{N}  AI {r['ai']:.1f} FLOP/B  "
              f"GPU SOL {r['t_sol_ms']:.3f} ms ({r['bound']})  "
              f"ANE SOL {r_ane['t_compute_ms']:.3f} ms  "
              f"mem {r['t_mem_ms']:.3f}")

        x = mx.random.normal((M, K)).astype(mx.float16)
        w = mx.random.normal((N, K)).astype(mx.float16)
        mx.eval(x, w)
        gpu_ms, gpu_min = bench(lambda: mx.eval(x @ w.T))
        print(f"  MLX GPU {gpu_ms:.3f} ms (min {gpu_min:.3f})  "
              f"SOL util {r['t_sol_ms']/gpu_ms:.1%}")
        row = {**r, "gpu_ms": gpu_ms, "gpu_sol_util": r["t_sol_ms"] / gpu_ms}

        if args.ane:
            import coremltools as ct
            from pathlib import Path
            path = Path(f"/tmp/ane_gemm_{M}x{K}x{N}.mlpackage")
            if not path.exists():
                build_ane_linear(M, K, N, path)
            model = ct.models.MLModel(str(path),
                                      compute_units=ct.ComputeUnit.CPU_AND_NE)
            x_np = np.array(x.astype(mx.float32))
            ane_ms, ane_min = bench(lambda: model.predict({"x": x_np}), 3, 10)
            print(f"  ANE {ane_ms:.3f} ms (min {ane_min:.3f})  "
                  f"ANE SOL util {r_ane['t_compute_ms']/ane_ms:.1%} "
                  f"(end-to-end, incl. copies)")
            row.update(ane_ms=ane_ms, ane_sol_util=r_ane["t_compute_ms"] / ane_ms)

            # optimal split: rows/ms throughput, overlap => min max
            g_r = M / gpu_ms
            a_r = M / ane_ms
            split = a_r / (a_r + g_r) if (a_r + g_r) else 0.5
            t_opt = max((1 - split) * M / g_r, split * M / a_r)
            print(f"  optimal ANE share {split:.1%}  overlapped {t_opt:.3f} ms "
                  f"(vs GPU-only {gpu_ms:.3f})")
            row.update(ane_share=split, overlapped_ms=t_opt)
        rows.append(row)

    print(f"\nridge point {RIDGE:.1f} FLOP/byte")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
