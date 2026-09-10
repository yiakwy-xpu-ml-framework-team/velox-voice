"""Benchmark for veloxvoice CUDA kernel: dw_causal_conv1d (depthwise causal conv with cache).

Phase 1: correctness vs F.conv1d(groups=C) (fp32, cuDNN TF32 disabled)
Phase 2: performance vs F.conv1d(groups=C) baseline + %SOL (memory-bound)

Run: python benchmark/kernels/bench_dw_causal_conv1d.py
"""

import torch
import torch.nn.functional as F

from veloxvoice.kernels.ops import dw_causal_conv1d

BW = 273.056e9  # GB10 theoretical LPDDR5x bytes/s


def _reference(x, cache, weight, bias):
    """Optimized torch baseline: single F.conv1d depthwise call (NOT a python loop).

    x [T,C], cache [K-1,C], weight [C,K], bias [C]
    -> out [T,C], new_cache [K-1,C]
    """
    T, C = x.shape
    K = weight.shape[1]
    x_cat = torch.cat([cache, x], dim=0)  # [T+K-1, C]
    xc = x_cat.t().unsqueeze(0)  # [1, C, T+K-1]
    out = F.conv1d(xc, weight.unsqueeze(1), bias, groups=C)  # [1, C, T]
    new_cache = x_cat[T:]  # last K-1 rows
    return out.squeeze(0).t(), new_cache


def _verify(shapes, atol=2e-4):
    print("Phase 1: correctness (CUDA vs F.conv1d groups=C, fp32)")
    prev = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        ok = True
        for T, C, K in shapes:
            torch.manual_seed(0)
            x = torch.randn(T, C, device="cuda", dtype=torch.float32)
            cache = torch.randn(K - 1, C, device="cuda", dtype=torch.float32)
            weight = torch.randn(C, K, device="cuda", dtype=torch.float32)
            bias = torch.randn(C, device="cuda", dtype=torch.float32)

            ref_out, ref_cache = _reference(x, cache, weight, bias)
            out, new_cache = dw_causal_conv1d(x, cache, weight, bias)

            d_out = (out - ref_out).abs().max().item()
            d_cache = (new_cache - ref_cache).abs().max().item()
            status = "PASS" if (d_out < atol and d_cache < atol) else "FAIL"
            ok &= d_out < atol and d_cache < atol
            print(
                f"  T={T:5d} C={C:4d} K={K}  max|out-ref|={d_out:.3e}  "
                f"max|cache-ref|={d_cache:.3e}  [{status}]"
            )
    finally:
        torch.backends.cudnn.allow_tf32 = prev
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
        f"{'config':>18s} {'cuda':>9s} {'conv1d':>9s} {'speedup':>8s} {'GB/s':>8s} {'%SOL':>6s}"
    )
    for T, C, K in shapes:
        x = torch.randn(T, C, device="cuda", dtype=torch.float32)
        cache = torch.randn(K - 1, C, device="cuda", dtype=torch.float32)
        weight = torch.randn(C, K, device="cuda", dtype=torch.float32)
        bias = torch.randn(C, device="cuda", dtype=torch.float32)

        cuda_us = _bench(lambda: dw_causal_conv1d(x, cache, weight, bias))
        torch_us = _bench(lambda: _reference(x, cache, weight, bias))

        traffic = (
            T * C + T * C + (K - 1) * C + C * K
        ) * 4  # read x,cache,w + write out
        gbs = traffic / (cuda_us * 1e-6) / 1e9
        print(
            f"T={T:5d} C={C:4d} K={K:2d} {cuda_us:8.2f}us {torch_us:8.2f}us "
            f"{torch_us/cuda_us:7.2f}x {gbs:7.1f} {100 * gbs*1e9/BW:5.1f}%"
        )


def main():
    shapes = [(1, 64, 5), (64, 64, 5), (256, 128, 5), (1024, 256, 7), (2048, 512, 5)]
    if not _verify(shapes):
        print("\nCorrectness FAILED — skipping benchmark.")
        return
    _benchmark(shapes)


if __name__ == "__main__":
    main()
