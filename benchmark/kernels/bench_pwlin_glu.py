"""Benchmark for veloxvoice CUDA kernel: pwlin_glu (fused 1x1-conv + silu-GLU).

Phase 1: correctness vs fp64 reference (bf16 output rounding is the only error)
Phase 2: performance vs F.linear+glu (cuBLAS)
         + %SOL vs H800 bf16 dense peak (989 TFLOPS) and HBM bandwidth

Run: python benchmark/kernels/bench_pwlin_glu.py
"""

import time

import torch
import torch.nn.functional as F

from veloxvoice.kernels.ops import pwlin_glu

# TODO (yiakwy) : add support to DGX Spark

PEAK_H800_TF = 989.0  # H800 bf16 dense TFLOPS
BW_H800_GBS = 3.35e12  # H800 HBM bytes/s

N_HALF = 512
N_DIM = 2 * N_HALF


def _reference(x, w, b):
    return F.glu(F.linear(x, w, b), dim=-1)


def _verify(shapes, atol=3e-2):
    print("Phase 1: correctness (CUDA vs fp64 reference, bf16)")
    ok = True
    for M in shapes:
        torch.manual_seed(0)
        x = torch.randn(M, N_HALF, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N_DIM, N_HALF, device="cuda", dtype=torch.bfloat16) * 0.05
        b = torch.randn(N_DIM, device="cuda", dtype=torch.bfloat16)
        ref = F.glu(F.linear(x.double(), w.double(), b.double()), dim=-1)
        out = pwlin_glu(x, w, b)
        if out is None:
            print(f"  M={M:5d}  kernel n/a on this arch [torch fallback]")
            continue
        diff = (out.double() - ref).abs().max().item()
        status = "PASS" if diff < atol else "FAIL"
        ok &= diff < atol
        print(f"  M={M:5d}  max|diff| vs fp64={diff:.3e}  [{status}]")
    return ok


def _bench(fn, iters=200, warmup=20):
    """CUDA-graph replay timing: removes launch overhead that otherwise
    dominates short kernels"""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()

    return (time.perf_counter() - t0) / iters * 1e6  # us


def _benchmark(shapes):
    print("\nPhase 2: performance (CUDA-graph replay GPU time)")
    print(
        f"{'shape':<18s} {'pwlin_glu':>10s} {'linear+glu':>11s} "
        f"{'speedup':>8s} {'GB/s':>7s} {'%memSOL':>7s} {'TFLOPS':>7s} "
        f"{'%cmpSOL':>8s} {'kernel':>10s}"
    )
    for M in shapes:
        torch.manual_seed(0)
        x = torch.randn(M, N_HALF, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N_DIM, N_HALF, device="cuda", dtype=torch.bfloat16) * 0.05
        b = torch.randn(N_DIM, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(M, N_HALF, device="cuda", dtype=torch.bfloat16)

        shape = f"M={M:<5d} K={N_HALF:<4d} N={N_HALF:<4d}"
        cuda_us = _bench(lambda: pwlin_glu(x, w, b, out=out))
        cublas_us = _bench(lambda: _reference(x, w, b))

        traffic = (M * (N_HALF + N_HALF) + N_DIM * N_HALF + N_DIM) * 2
        gbs = traffic / (cuda_us * 1e-6) / 1e9
        tflops = 2 * M * N_DIM * N_HALF / (cuda_us * 1e-6) / 1e12

        # TODO (yiakwy) : select from detected arch
        mem_sol = 100 * gbs * 1e9 / BW_H800_GBS
        cmp_sol = 100 * tflops / PEAK_H800_TF

        print(
            f"{shape:<18s} {cuda_us:8.2f}us {cublas_us:8.2f}us "
            f"{cublas_us / cuda_us:7.2f}x "
            f"{gbs:7.1f} {mem_sol:6.1f}% {tflops:7.1f} "
            f"{cmp_sol:7.1f}% {'pwlin_glu':>10s}"
        )


def main():
    shapes = [64, 128, 256, 512, 1024, 2488, 3523]
    if not _verify(shapes):
        print("\nCorrectness FAILED — skipping benchmark.")
        return
    _benchmark(shapes)


if __name__ == "__main__":
    main()
