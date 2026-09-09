"""Benchmark for fused power-mel-log CUDA kernel (TF32 MMA pipeline).

Correctness contract (see tests/kernels/test_power_mel_log.py):
  mma.sync m16n8k8 tf32 TRUNCATES unconverted f32 operands (10-bit mantissa,
  RZ). The strict reference emulates RZ operand rounding with BK=64-chunked
  fp32 accumulation. vs a plain fp32 GEMM, random-mel inputs show large
  log-domain deviations only at catastrophic-cancellation outputs (~0.15%
  of elements); with real non-negative mel filterbanks the kernel matches
  fp32 to ~1.5e-3.

Phase 1: correctness (RZ-emulation for the CUDA path, fp32 for the cuBLAS
         fallback; real-fbank fp32 check)
Phase 2: performance (CUDA-event GPU time, %SOL from minimal DRAM traffic)

Usage:
  python benchmark/kernels/bench_power_mel_log.py
"""

from __future__ import annotations

import time

import torch

from veloxvoice.kernels.ops import power_mel_log, reference_power_mel_log

BW = 273.056e9  # GB10 theoretical LPDDR5x bytes/s


# ---------------------------------------------------------------------------
# References (mirror tests/kernels/test_power_mel_log.py)
# ---------------------------------------------------------------------------


def _tf32_rz(x):
    xi = x.contiguous().view(torch.int32)
    return (xi & ~0x1FFF).view(torch.float32)


def emulated_power_mel_log(spec, mel, cmvn_mean=None, cmvn_istd=None):
    T, F = spec.shape
    M = mel.shape[0]
    BK = 64
    F_pad = ((F + BK - 1) // BK) * BK
    if (F_pad // BK) % 2:
        F_pad += BK
    power = spec.real.float() ** 2 + spec.imag.float() ** 2
    P = torch.zeros(T, F_pad, device=spec.device)
    P[:, :F] = _tf32_rz(power)
    Mt = torch.zeros(F_pad, M, device=mel.device)
    Mt[:F] = _tf32_rz(mel.t())
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        acc = torch.zeros(T, M, device=spec.device)
        for k0 in range(0, F_pad, BK):
            acc += P[:, k0 : k0 + BK] @ Mt[k0 : k0 + BK, :]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    out = torch.log(torch.clamp_min(acc, 1.1920929e-7))
    if cmvn_mean is not None:
        out = (out - cmvn_mean) * cmvn_istd
    return out


def reference_for_dispatch(spec, mel, cmvn_mean=None, cmvn_istd=None):
    T, F = spec.shape
    M = mel.shape[0]
    if M <= 80 and T >= 64:
        return emulated_power_mel_log(spec, mel, cmvn_mean, cmvn_istd)
    return reference_power_mel_log(spec.real, spec.imag, mel, cmvn_mean, cmvn_istd)


def _real_mel_fbank(F, M):
    import torchaudio.functional as AF

    return AF.melscale_fbanks(F, 0.0, 8000.0, M, 16000).cuda().t().contiguous()


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _gpu_time(fn, iters=200, warmup=30):
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


def _wall_time(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def _flops(T, F, M):
    return 2 * T * F * M + 2 * T * F + T * M


# ---------------------------------------------------------------------------
# Phase 1: correctness
# ---------------------------------------------------------------------------


def verify_correctness(shapes):
    print("Phase 1: correctness (noise-scale-aware: |exp(out)-exp(ref)| < 0.35 =")
    print("         4.5-sigma RZ-truncation bound; 99.9pct log-diff < 5e-3 guard)")
    all_ok = True
    for T, F, M in shapes:
        torch.manual_seed(42)
        spec = torch.randn(T, F, device="cuda", dtype=torch.cfloat)
        mel = torch.randn(M, F, device="cuda", dtype=torch.float32)
        ref = reference_for_dispatch(spec, mel)
        out = power_mel_log(spec, mel, None, None)
        out_f, ref_f = out.flatten().float(), ref.flatten().float()
        lin = (torch.exp(out_f) - torch.exp(ref_f)).abs().max().item()
        meaningful = ref_f > -9.0
        d = (out_f - ref_f).abs()[meaningful]
        q999 = torch.quantile(d, 0.999).item() if d.numel() else 0.0
        ok = lin < 0.35 and q999 < 5e-3
        all_ok &= ok
        path = "MMA" if (M <= 80 and T >= 64) else "cuBLAS"
        print(
            f"  T={T:5d} F={F:3d} M={M:2d} [{path:6s}] prelog-lin={lin:.4f} "
            f"99.9%log={q999:.5f} [{'PASS' if ok else 'FAIL'}]"
        )

    print("  (b) real mel filterbank vs fp32 (production regime, no cancellation)")
    for T, F, M in [(128, 257, 80), (256, 513, 80)]:
        torch.manual_seed(42)
        spec = torch.randn(T, F, device="cuda", dtype=torch.cfloat)
        mel = _real_mel_fbank(F, M)
        ref = reference_power_mel_log(spec.real, spec.imag, mel, None, None)
        out = power_mel_log(spec, mel, None, None)
        mx = (out - ref).abs().max().item()
        ok = mx < 5e-3
        all_ok &= ok
        print(
            f"  T={T:5d} F={F:3d} M={M:2d}          max={mx:.4e} [{'PASS' if ok else 'FAIL'}]"
        )
    return all_ok


# ---------------------------------------------------------------------------
# Phase 2: performance
# ---------------------------------------------------------------------------


def main():
    shapes = [
        (64, 257, 80),
        (128, 257, 80),
        (256, 257, 80),
        (512, 257, 80),
        (1024, 257, 80),
        (2048, 257, 80),
        (128, 513, 80),
        (256, 513, 80),
        (2048, 513, 80),
    ]

    if not verify_correctness(shapes):
        print("\nCorrectness FAILED — skipping benchmark.")
        return

    print()
    print("Phase 2: performance (CUDA-event GPU time; %SOL vs 273 GB/s DRAM;")
    print("         sizes fitting in L2 can exceed DRAM SOL — see eff. GB/s)")
    print(
        f"{'shape':>16s} {'cuda_gpu':>9s} {'cuda_wall':>10s} {'torch_gpu':>9s} "
        f"{'speedup':>8s} {'TFLOPS':>7s} {'%SOL':>6s}"
    )
    for T, F, M in shapes:
        torch.manual_seed(42)
        spec = torch.randn(T, F, device="cuda", dtype=torch.cfloat)
        mel = torch.randn(M, F, device="cuda", dtype=torch.float32)
        flops = _flops(T, F, M)

        g_cuda = _gpu_time(lambda: power_mel_log(spec, mel, None, None))
        w_cuda = _wall_time(lambda: power_mel_log(spec, mel, None, None))
        g_torch = _gpu_time(
            lambda: reference_power_mel_log(spec.real, spec.imag, mel, None, None)
        )

        F_pad = ((F + 63) // 64) * 64
        traffic = T * F * 8 + F_pad * M * 4 + T * M * 4
        sol_us = traffic / BW * 1e6

        print(
            f"T={T:5d}xF{F:3d}xM{M:2d} "
            f"{g_cuda:8.2f}us {w_cuda:9.2f}us {g_torch:8.2f}us "
            f"{g_torch / g_cuda:7.2f}x {flops / g_cuda / 1e6:6.2f} "
            f"{100 * sol_us / g_cuda:5.1f}%"
        )

    print()
    print("Notes:")
    print("  - ~9us TVM-FFI dispatch floor dominates small-T wall time (1 launch")
    print("    vs 4+ for the torch pipeline, so end-to-end streaming still wins).")
    print("  - %SOL uses minimal DRAM traffic (spec + mel + out); mel is re-read")
    print("    per tile but L2-resident (~82-184KB working set).")


if __name__ == "__main__":
    main()
