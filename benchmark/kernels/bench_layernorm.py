"""Benchmark for veloxvoice CUDA kernel: fused layernorm.

Phase 1: correctness vs F.layer_norm (fp32, TF32 disabled)
Phase 2: performance vs F.layer_norm baseline + %SOL (memory-bound)

Run: python benchmark/kernels/bench_layernorm.py
"""

import torch

from veloxvoice.kernels.ops import fused_layernorm, reference_layernorm

BW = 273.056e9  # GB10 theoretical LPDDR5x bytes/s


def _verify(shapes, atol=2e-4):
    print("Phase 1: correctness (CUDA vs F.layer_norm, fp32)")
    ok = True
    for T, D in shapes:
        torch.manual_seed(0)
        x = torch.randn(T, D, device="cuda", dtype=torch.float32)
        w = torch.randn(D, device="cuda", dtype=torch.float32)
        b = torch.randn(D, device="cuda", dtype=torch.float32)
        ref = reference_layernorm(x, w, b, eps=1e-5)
        out = fused_layernorm(x, w, b, eps=1e-5)
        diff = (out - ref).abs().max().item()
        status = "PASS" if diff < atol else "FAIL"
        ok &= diff < atol
        print(f"  T={T:5d} D={D:5d}  max|diff|={diff:.3e}  [{status}]")
    # eps stability: constant input -> zeros
    x = torch.ones(8, 64, device="cuda") * 5.0
    out = fused_layernorm(
        x, torch.ones(64, device="cuda"), torch.zeros(64, device="cuda")
    )
    diff = out.abs().max().item()
    status = "PASS" if diff < 1e-4 else "FAIL"
    ok &= diff < 1e-4
    print(f"  eps stability (const input)  max|out|={diff:.3e}  [{status}]")
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
    for T, D in shapes:
        x = torch.randn(T, D, device="cuda", dtype=torch.float32)
        w = torch.randn(D, device="cuda", dtype=torch.float32)
        b = torch.randn(D, device="cuda", dtype=torch.float32)

        cuda_us = _bench(lambda: fused_layernorm(x, w, b))
        torch_us = _bench(lambda: reference_layernorm(x, w, b))

        traffic = (2 * T * D + 2 * D) * 4  # read x + write out + params
        gbs = traffic / (cuda_us * 1e-6) / 1e9
        print(
            f"{T:>6d}x{D:<6d} {cuda_us:8.2f}us {torch_us:8.2f}us {torch_us/cuda_us:7.2f}x "
            f"{gbs:7.1f} {100 * gbs*1e9/BW:5.1f}%"
        )


def main():
    shapes = [(1, 256), (64, 256), (256, 512), (1024, 512), (4096, 1024)]
    if not _verify(shapes):
        print("\nCorrectness FAILED — skipping benchmark.")
        return
    _benchmark(shapes)


if __name__ == "__main__":
    main()
