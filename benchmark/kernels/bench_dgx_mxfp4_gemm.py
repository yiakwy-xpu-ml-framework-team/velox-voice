"""benchmark for `veloxvoice.kernels.ops.dgx_mxfp4_gemm` (mxfp4 lower level engine).

Run: `python benchmark/kernels/bench_dgx_mxfp4_gemm.py`.
"""

import time

import numpy as np
import torch

from veloxvoice.kernels.ops import dgx_mxfp4_gemm
from veloxvoice.kernels.ops.triton_ops import per_row_col_quantize


def measure_performance(flops_fn, flops, iters=8, warmup=3):
    for _ in range(warmup):
        flops_fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        flops_fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - start) / iters
    return dt, flops / dt / 1e12


def bench():
    shapes = [
        (2048, 2048, 2048),
        (4096, 2048, 2048),
        (8192, 2048, 2048),
        (2048, 1024, 8192),
        (1024, 1024, 16384),
    ]
    print(f"{'shape':>16s} {'wall':>9s} {'TFLOPS':>8s}")
    for M, N, K in shapes:
        if (M % 128) or (N % 128) or (K % 64):
            print(f"({M},{N},{K}) -> skip (alignment requirement)")
            continue

        # Generate real random data and quantize properly
        a_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b_fp32 = torch.randn(N, K, device="cuda", dtype=torch.float32)

        # Per-row MXFP4 quantize: codes [M, K/2], scales [M]
        a_codes, a_scales = per_row_col_quantize(a_fp32)
        b_codes, b_scales = per_row_col_quantize(b_fp32)

        f = lambda: dgx_mxfp4_gemm(a_codes, b_codes, a_scales, b_scales)
        dt, tfl = measure_performance(f, 2 * M * N * K)
        ms = dt * 1e3
        print(f"{M:>6d}x{N:>6d}x{K:>6d}  {ms:8.2f}ms {tfl:8.2f} TFLOPS")


if __name__ == "__main__":
    bench()
