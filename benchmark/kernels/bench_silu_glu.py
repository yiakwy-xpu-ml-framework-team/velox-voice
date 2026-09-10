"""Benchmark for veloxvoice CUDA kernel: fused silu_glu.

Phase 1: correctness vs F.glu (fp32)
Phase 2: performance vs F.glu baseline + %SOL (memory-bound)

Run: python benchmark/kernels/bench_silu_glu.py
"""

import torch
import torch.nn.functional as F

from veloxvoice.kernels.ops import silu_glu

BW = 273.056e9  # GB10 theoretical LPDDR5x bytes/s


def _reference(x):
    """F.glu(x, dim=-1) == x[:, :C] * sigmoid(x[:, C:])."""
    return F.glu(x, dim=-1)


def _verify(shapes, atol=2e-5):
    print("Phase 1: correctness (CUDA vs F.glu, fp32)")
    ok = True
    for T, C in shapes:
        torch.manual_seed(0)
        x = torch.randn(T, 2 * C, device="cuda", dtype=torch.float32)
        ref = _reference(x)
        out = silu_glu(x)
        diff = (out - ref).abs().max().item()
        status = "PASS" if diff < atol else "FAIL"
        ok &= diff < atol
        print(f"  T={T:5d} C={C:5d}  max|diff|={diff:.3e}  [{status}]")
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
        f"{'shape':>14s} {'cuda':>9s} {'torch':>9s} {'speedup':>8s} {'GB/s':>8s} {'%SOL':>6s}"
    )
    for T, C in shapes:
        x = torch.randn(T, 2 * C, device="cuda", dtype=torch.float32)

        cuda_us = _bench(lambda: silu_glu(x))
        torch_us = _bench(lambda: _reference(x))

        traffic = (T * 2 * C + T * C) * 4  # read x + write out
        gbs = traffic / (cuda_us * 1e-6) / 1e9
        print(
            f"{T:>6d}x{C:<6d} {cuda_us:8.2f}us {torch_us:8.2f}us {torch_us/cuda_us:7.2f}x "
            f"{gbs:7.1f} {100 * gbs*1e9/BW:5.1f}%"
        )


def main():
    shapes = [(1, 128), (64, 128), (256, 256), (1024, 512), (4096, 1024)]
    if not _verify(shapes):
        print("\nCorrectness FAILED — skipping benchmark.")
        return
    _benchmark(shapes)


if __name__ == "__main__":
    main()
