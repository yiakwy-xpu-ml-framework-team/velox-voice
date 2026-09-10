"""Benchmark for veloxvoice CUDA kernel: fused_qkv projection.

Phase 1: correctness vs F.linear (fp32, TF32 disabled)
Phase 2: performance vs F.linear (cuBLAS) baseline + %SOL (memory-bound at small M)

Run: python benchmark/kernels/bench_fused_qkv.py
"""

import torch

from veloxvoice.kernels.ops import fused_qkv, reference_fused_qkv

BW = 273.056e9  # GB10 theoretical LPDDR5x bytes/s


def _verify(shapes, atol=2e-4):
    print("Phase 1: correctness (CUDA vs F.linear, fp32)")
    ok = True
    for M, K, N in shapes:
        torch.manual_seed(0)
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        w = torch.randn(3 * N, K, device="cuda", dtype=torch.float32)
        bias = torch.randn(3 * N, device="cuda", dtype=torch.float32)
        ref = reference_fused_qkv(x, w, bias)
        out = fused_qkv(x, w, bias)
        diff = (out - ref).abs().max().item()
        status = "PASS" if diff < atol else "FAIL"
        ok &= diff < atol
        print(f"  M={M:3d} K={K:5d} N={N:4d}  max|diff|={diff:.3e}  [{status}]")
    return ok


def _bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def _benchmark(shapes):
    print("\nPhase 2: performance (CUDA-event GPU time)")
    print(
        f"{'shape':>18s} {'cuda':>9s} {'cuBLAS':>9s} {'speedup':>8s} {'GB/s':>8s} {'%SOL':>6s}"
    )
    for M, K, N in shapes:
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        w = torch.randn(3 * N, K, device="cuda", dtype=torch.float32)
        bias = torch.randn(3 * N, device="cuda", dtype=torch.float32)

        cuda_us = _bench(lambda: fused_qkv(x, w, bias))
        torch_us = _bench(lambda: reference_fused_qkv(x, w, bias))

        traffic = (M * K + 3 * N * K + 3 * N + M * 3 * N) * 4  # read x,w,b + write out
        gbs = traffic / (cuda_us * 1e-6) / 1e9
        print(
            f"M={M:3d} K={K:5d} N={N:4d} {cuda_us:8.2f}us {torch_us:8.2f}us "
            f"{torch_us/cuda_us:7.2f}x {gbs:7.1f} {100 * gbs*1e9/BW:5.1f}%"
        )


def main():
    # streaming shapes (M <= 32) are the kernel's target regime
    shapes = [
        (1, 512, 128),
        (4, 512, 128),
        (16, 512, 128),
        (32, 512, 128),
        (1, 1024, 256),
        (16, 1024, 256),
    ]
    if not _verify(shapes):
        print("\nCorrectness FAILED — skipping benchmark.")
        return
    _benchmark(shapes)


if __name__ == "__main__":
    main()
