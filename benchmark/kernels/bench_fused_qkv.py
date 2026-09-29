"""Benchmark for veloxvoice CUDA kernel: fused_qkv projection.

Phase 1: correctness vs F.linear (fp32, TF32 disabled)
Phase 2: performance vs F.linear (cuBLASLt) baseline / %SOL

Kernel Dispath Path :
- small-M fp32 : `fused_qkv` (sm_120a SIMT mma) for GB10 and Hopper;
- bf16 M >= 64 : `fused_qkv_wgmma` (sm_90a 1p2c wgmma) for Hopper;

Oops! Note for small M fused_qkv is faster than cudaBlasLt in our tests.

Run: python benchmark/kernels/bench_fused_qkv.py
"""

import os
import time

import torch

from veloxvoice.kernels.ops import (  # TODO (yiakwy) :; - fused_qkv_wgmma is slower than cudablasLt in our tests, but; - we can fuse it with cu_packing to make it faster than cudablasLt + torch packing in our tests
    fused_qkv,
    fused_qkv_wgmma,
    reference_fused_qkv,
)

# Per-architecture SOL constants: (memory BW bytes/s, bf16 dense TFLOPS).
#   sm_90a  H800 (Hopper):     HBM 3.35 TB/s, bf16 dense 989 TFLOP/s
#   sm_121 GB10 (DGX Spark):  LPDDR5x ~273 GB/s, bf16 dense ~125 TFLOP/s
#
_ARCH_SOL = {
    (9, 0): (3.35e12, 989.0),
    (12, 1): (273.056e9, 125.0),
}


_SHAPE_W = 30


def _dtype_to_str(datatype):
    return "bf16" if datatype == torch.bfloat16 else "fp32"


def _shape_str(M, K, N, datatype):
    return f"M={M:<5d} K={K:<5d} N={N:<4d} {_dtype_to_str(datatype):<4s}"


# ---------------------------------------------------------------- dispatch
# Kernel support predicates — MUST mirror the wrappers' own guards; the
# decision is made from arch/dtype/shape up front (no trial calls), and the
# chosen path is memoized per (arch, dtype, shape) so it is probed once.
_SIMT_MAX_M = 32  # velox_fused_qkv.cu MAX_M
_WGMMA_ARCH = (9, 0)  # sm_90a wgmma: Hopper only (GB10 sm_121 -> torch)
_WGMMA_K = 512  # K_BLOCK
_WGMMA_N = 1536  # N_OUT (q|k|v)

_KERNELS = {
    "simt": fused_qkv,  # fp32 small-M SIMT (GB10 lane)
    "wgmma": fused_qkv_wgmma,  # bf16 sm_90a 1p2c wgmma
    "torch": reference_fused_qkv,  # cuBLAS
}

_ARCH = None
_PATH_CACHE = {}  # (arch, dtype, M, K, N) -> path
_TEST_4096 = os.environ.get("VELOXVOICE_FQKV_TEST_4096", "0") == "1"


def _plain_4096(M, K, N):
    """Temporary probe: benchmark one [M,K] @ [N,K]^T GEMM as [4096]^3."""
    return _TEST_4096 and (M, K, N) == (4096, 4096, 4096)


def _make_gemm_tensors(M, K, N, dt):
    """Return (x, w, b).  The fused-qkv benchmark normally stores `w` as
    [3N,K]; the temporary 4096 probe uses a plain [N,K] GEMM."""
    x = torch.randn(M, K, device="cuda", dtype=dt)
    if _plain_4096(M, K, N):
        w = torch.randn(N, K, device="cuda", dtype=dt) * 0.05
        b = torch.randn(N, device="cuda", dtype=dt) * 0.05
    else:
        w = torch.randn(3 * N, K, device="cuda", dtype=dt) * 0.05
        b = torch.randn(3 * N, device="cuda", dtype=dt) * 0.05
    return x, w, b


def _replicas(M, K, N):
    return 1 if _plain_4096(M, K, N) else 3


def _sol():
    """(mem_BW_bytes_s, bf16_dense_TFLOPS) for the current arch, or None."""
    return _ARCH_SOL.get(_arch())


def _arch():
    global _ARCH
    if _ARCH is None:
        _ARCH = (
            torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        )
    return _ARCH


def _pick_path(dt, M, K, N):
    """Static decision by architecture + dtype + shape (mirrors the kernel
    wrappers' guards): fp32 M<=32 -> SIMT (any arch); bf16 M>=64 on sm_90 ->
    wgmma; everything else -> cuBLAS."""
    if dt == torch.float32 and M <= _SIMT_MAX_M and K % 4 == 0:
        return "simt"
    if _plain_4096(M, K, N):
        return "wgmma"
    if _TEST_4096:
        return "torch"
    if (
        dt == torch.bfloat16
        and _arch() == _WGMMA_ARCH
        and M >= 64
        and K == _WGMMA_K
        and N == _WGMMA_N
    ):
        return "wgmma"
    return "torch"


def _resolve(x, w, b):
    """Pick the kernel for this shape -> (path, callable(current tensors)).

    The PATH is decided from arch/dtype/shape and memoized; the callable is
    rebuilt each call so it always binds the tensors passed in."""
    M = x.reshape(-1, x.shape[-1]).shape[0]
    K = x.shape[-1]

    N = w.shape[0]

    key = (_arch(), str(x.dtype), M, K, N)
    path = _PATH_CACHE.get(key)
    if path is None:
        path = _pick_path(x.dtype, M, K, N)
        _PATH_CACHE[key] = path
    fn = _KERNELS[path]
    return path, (lambda: fn(x, w, b))


def _verify(shapes, atol=2e-2):
    torch.manual_seed(0)

    print(
        f"Phase 1: correctness vs F.linear (cuBLAS)  "
        f"[atol=2e-2, arch=sm_{_arch()[0]}{_arch()[1]}]"
    )
    ok = True
    for M, K, N, dt in shapes:
        x, w_qkv, bias_qkv = _make_gemm_tensors(M, K, N, dt)
        path, call = _resolve(x, w_qkv, bias_qkv)
        s = _shape_str(M, K, N, dt)
        if path == "torch":
            print(f"  {s:<{_SHAPE_W}s} kernel n/a -> cuBLAS only")
            continue
        ref = reference_fused_qkv(x, w_qkv, bias_qkv)
        out = call()
        diff = (out.float() - ref.float()).abs().max().item()
        status = "PASS" if diff < atol else "FAIL"
        ok &= diff < atol
        print(f"  {s:<{_SHAPE_W}s} kernel={path:<5s} max|diff|={diff:.3e}  [{status}]")
    return ok


def _bench(fn, iters=200, warmup=20):
    """CUDA-graph replay timing: removes the per-call launch+sync latency
    (~14 us) that otherwise dominates short kernels and makes small-M rows
    meaningless (and matches the model's --use-graphs execution)."""
    try:
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
    except Exception:
        # fall back to plain event timing when capture is unavailable
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
        return s.elapsed_time(e) / iters * 1e3


def _benchmark(shapes):
    sol = _sol()
    if sol is None:
        note = f"arch sm_{_arch()[0]}{_arch()[1]}: SOL not registered -> n/a"
    else:
        note = (
            f"arch sm_{_arch()[0]}{_arch()[1]}: SOL mem={sol[0] / 1e12:.2f} TB/s, "
            f"bf16 dense={sol[1]:.0f} TFLOPS"
        )
    print(f"\nPhase 2: performance (CUDA-graph replay GPU time)  [{note}]")
    print(
        f"{'shape':<{_SHAPE_W}s} {'cuda':>10s} {'cuBLAS':>10s} {'speedup':>8s} "
        f"{'GB/s':>7s} {'%memSOL':>7s} {'TFLOPS':>6s} {'%cmpSOL':>7s} {'kernel':>6s}"
    )
    for M, K, N, dt in shapes:
        x, w_qkv, bias_qkv = _make_gemm_tensors(M, K, N, dt)

        path, call = _resolve(x, w_qkv, bias_qkv)
        cuda_us = _bench(call)
        torch_us = _bench(lambda: reference_fused_qkv(x, w_qkv, bias_qkv))

        esize = 2 if dt == torch.bfloat16 else 4
        rep = _replicas(M, K, N)
        traffic = (M * K + rep * N * K + rep * N + M * rep * N) * esize
        gbs = traffic / (cuda_us * 1e-6) / 1e9
        tflops = 2 * M * K * rep * N / (cuda_us * 1e-6) / 1e12

        sol = _sol()
        if sol is None:
            mem_sol_s, cmp_sol_s = f"{'n/a':>7s}", f"{'n/a':>7s}"
        else:
            mem_bw, peak_tf = sol
            mem_sol_s = f"{100 * gbs * 1e9 / mem_bw:>6.1f}%"
            cmp_sol_s = f"{100 * tflops / peak_tf:>6.1f}%"
        print(
            f"{_shape_str(M, K, N, dt):<{_SHAPE_W}s} "
            f"{cuda_us:>8.2f}us {torch_us:>8.2f}us {torch_us / cuda_us:>7.2f}x "
            f"{gbs:>7.1f} {mem_sol_s} {tflops:>6.1f} {cmp_sol_s} "
            f"{path:>6s}"
        )


def main():
    # streaming shapes (M <= 32, fp32 SIMT kernel — GB10 target regime)
    streaming = [
        (1, 512, 128, torch.float32),
        (4, 512, 128, torch.float32),
        (16, 512, 128, torch.float32),
        (32, 512, 128, torch.float32),
        (1, 1024, 256, torch.float32),
        (16, 1024, 256, torch.float32),
    ]

    # M represents sampled frames of the bench audios
    # - 99.58s : 2488
    # - long spans (1 hrs) : 3523
    full_context = [
        (64, 512, 512, torch.bfloat16),
        (128, 512, 512, torch.bfloat16),
        (512, 512, 512, torch.bfloat16),
        (2488, 512, 512, torch.bfloat16),
        (3523, 512, 512, torch.bfloat16),
        # NOTE (yiakwy) : we can easily beat cudaBlasLt at this scale
        # (4096, 4096, 4096, torch.bfloat16),
    ]

    if _TEST_4096:
        full_context.append((4096, 4096, 4096, torch.bfloat16))

    if not _verify(streaming + full_context):
        print("\nCorrectness FAILED — skipping benchmark.")
        return
    _benchmark(streaming)
    _benchmark(full_context)


if __name__ == "__main__":
    main()
