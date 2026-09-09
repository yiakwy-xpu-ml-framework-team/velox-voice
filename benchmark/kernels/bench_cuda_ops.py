"""Benchmarks for veloxvoice CUDA kernels: layernorm, silu_glu, fused_qkv,
chunk_rel_pos_attn, dw_causal_conv1d.

Run: python benchmark/kernels/bench_cuda_ops.py
"""

import time

import torch

from veloxvoice.kernels.ops import (
    chunk_rel_pos_attn,
    dw_causal_conv1d,
    fused_layernorm,
    fused_qkv,
    reference_fused_qkv,
    reference_layernorm,
    silu_glu,
)


def bench(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times)


def bench_layernorm():
    print("\n=== LayerNorm ===")
    print(f"{'Shape':>12s} {'CUDA':>8s} {'cuBLAS':>8s} {'Speedup':>8s}")
    for T, D in [(64, 256), (256, 512), (1024, 512), (4096, 1024)]:
        x = torch.randn(T, D, device="cuda", dtype=torch.float32)
        w = torch.randn(D, device="cuda", dtype=torch.float32)
        b = torch.randn(D, device="cuda", dtype=torch.float32)

        cuda_ms = bench(lambda: fused_layernorm(x, w, b))
        ref_ms = bench(lambda: reference_layernorm(x, w, b))
        print(
            f"{T:>5d}x{D:<5d} {cuda_ms:7.3f}ms {ref_ms:7.3f}ms {ref_ms/cuda_ms:7.2f}x"
        )


def bench_silu_glu():
    print("\n=== SiLU-GLU ===")
    print(f"{'Shape':>12s} {'CUDA':>8s} {'PyTorch':>8s} {'Speedup':>8s}")
    for T, C in [(64, 128), (256, 256), (1024, 512), (4096, 1024)]:
        x = torch.randn(T, 2 * C, device="cuda", dtype=torch.float32)

        cuda_ms = bench(lambda: silu_glu(x))
        ref_ms = bench(lambda: x[:, :C] * torch.sigmoid(x[:, C:]))
        print(
            f"{T:>5d}x{C:<5d} {cuda_ms:7.3f}ms {ref_ms:7.3f}ms {ref_ms/cuda_ms:7.2f}x"
        )


def bench_fused_qkv():
    print("\n=== Fused QKV ===")
    print(f"{'Shape':>16s} {'CUDA':>8s} {'cuBLAS':>8s} {'Speedup':>8s}")
    for M, K, N in [
        (1, 512, 128),
        (4, 512, 128),
        (16, 512, 128),
        (1, 1024, 256),
        (4, 1024, 256),
        (16, 1024, 256),
    ]:
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        w = torch.randn(3 * N, K, device="cuda", dtype=torch.float32)
        bias = torch.randn(3 * N, device="cuda", dtype=torch.float32)

        cuda_ms = bench(lambda: fused_qkv(x, w, bias))
        ref_ms = bench(lambda: reference_fused_qkv(x, w, bias))
        print(
            f"{M:>3d}x{K:<5d}x{N:<5d} {cuda_ms:7.3f}ms {ref_ms:7.3f}ms {ref_ms/cuda_ms:7.2f}x"
        )


def bench_chunk_rel_pos_attn():
    print("\n=== Chunk RelPos Attn ===")
    print(f"{'Config':>24s} {'CUDA':>8s}")
    for H, Tq, L, DK, SPAN in [
        (8, 64, 64, 64, 32),
        (8, 128, 128, 64, 64),
        (16, 256, 256, 128, 128),
    ]:
        q_u = torch.randn(H, Tq, DK, device="cuda", dtype=torch.float32)
        q_v = torch.randn(H, Tq, DK, device="cuda", dtype=torch.float32)
        k = torch.randn(H, L, DK, device="cuda", dtype=torch.float32)
        v = torch.randn(H, L, DK, device="cuda", dtype=torch.float32)
        pe = torch.randn(SPAN, H * DK, device="cuda", dtype=torch.float32)
        idx = torch.zeros(Tq, L, device="cuda", dtype=torch.int32)
        for i in range(Tq):
            start = max(0, i - L // 2)
            end = min(L, start + L)
            idx[i, start:end] = torch.arange(
                start, end, device="cuda", dtype=torch.int32
            )
        valid = torch.full((H, Tq), L, device="cuda", dtype=torch.int32)

        cuda_ms = bench(
            lambda: chunk_rel_pos_attn(q_u, q_v, k, pe, idx, v, valid), iters=50
        )
        cfg = f"H={H} Tq={Tq} L={L} DK={DK}"
        print(f"{cfg:>24s} {cuda_ms:7.3f}ms")


def bench_dw_causal_conv1d():
    print("\n=== Dw Causal Conv1d ===")
    print(f"{'Config':>20s} {'CUDA':>8s} {'PyTorch':>8s} {'Speedup':>8s}")
    for T, C, K in [(64, 64, 5), (256, 128, 5), (1024, 256, 7)]:
        x = torch.randn(T, C, device="cuda", dtype=torch.float32)

        cache = torch.randn(K - 1, C, device="cuda", dtype=torch.float32)

        weight = torch.randn(C, K, device="cuda", dtype=torch.float32)
        bias = torch.randn(C, device="cuda", dtype=torch.float32)

        def ref_conv():
            T_, C_ = x.shape
            x_cat = torch.cat([cache, x], dim=0)
            out = torch.zeros(T_, C_, device=x.device, dtype=x.dtype)
            for t in range(T_):
                out[t] = (x_cat[t : t + K] * weight.T).sum(dim=0) + bias
            return out

        cuda_ms = bench(lambda: dw_causal_conv1d(x, cache, weight, bias))
        ref_ms = bench(lambda: ref_conv())
        cfg = f"T={T} C={C} K={K}"
        print(f"{cfg:>20s} {cuda_ms:7.3f}ms {ref_ms:7.3f}ms {ref_ms/cuda_ms:7.2f}x")


if __name__ == "__main__":
    bench_layernorm()
    bench_silu_glu()
    bench_fused_qkv()
    bench_chunk_rel_pos_attn()
    bench_dw_causal_conv1d()
