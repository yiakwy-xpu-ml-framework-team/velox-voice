"""Tests for fused power-mel-log CUDA kernel (TF32 tensor-core pipeline).

Precision contract (measured on GB10, mma.sync.aligned.m16n8k8 tf32):
  The MMA instruction TRUNCATES unconverted f32 operands to tf32 (10-bit
  mantissa, round-toward-zero); PTX requires explicit cvt.rn.tf32.f32 for
  nearest rounding, which this kernel does not use. The correct reference
  therefore emulates RZ operand rounding with BK=64-chunked fp32
  accumulation (validated to max|diff| ~ 7e-3 in the log domain).

  Comparing against a plain fp32 GEMM shows large log-domain deviations
  ONLY at catastrophic-cancellation outputs (near-zero GEMM results where
  +-10-magnitude partial sums cancel), which the clamped log amplifies
  (~0.15% of random-mel elements). With real non-negative mel filterbanks
  there is no cancellation and the kernel matches fp32 to ~1.5e-3.
"""

from __future__ import annotations

import pytest
import torch

from veloxvoice.kernels.ops import power_mel_log, reference_power_mel_log

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _tf32_rz(x: torch.Tensor) -> torch.Tensor:
    """Truncate fp32 mantissa to 10 bits (what mma.sync tf32 does to operands)."""
    xi = x.contiguous().view(torch.int32)
    return (xi & ~0x1FFF).view(torch.float32)


def emulated_power_mel_log(spec, mel, cmvn_mean=None, cmvn_istd=None):
    """Emulate the kernel arithmetic: RZ-tf32 operands, BK=64-chunked fp32 GEMM."""
    T, F = spec.shape
    M = mel.shape[0]
    BK = 64
    F_pad = ((F + BK - 1) // BK) * BK
    if (F_pad // BK) % 2:
        F_pad += BK  # match wrapper padding rule
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


def _real_mel_fbank(F, M, device="cuda"):
    """Non-negative triangular mel filterbank [M, F] (production regime)."""
    import torchaudio.functional as AF

    fbank = AF.melscale_fbanks(F, 0.0, 8000.0, M, 16000).to(device)
    return fbank.t().contiguous()


def reference_for_dispatch(spec, mel, cmvn_mean=None, cmvn_istd=None):
    """Mirror power_mel_log's dispatch: CUDA MMA path -> RZ emulation;
    cuBLAS fallback (T < 64 or M > 80) -> plain fp32 reference."""
    T, F = spec.shape
    M = mel.shape[0]
    if M <= 80 and T >= 64:
        return emulated_power_mel_log(spec, mel, cmvn_mean, cmvn_istd)
    return reference_power_mel_log(spec.real, spec.imag, mel, cmvn_mean, cmvn_istd)


def _assert_close(out, ref, lin_tol=0.35, q999_tol=5e-3):
    """Noise-scale-aware comparison.

    The kernel's pre-log GEMM carries absolute RZ-truncation noise
    sigma ~ sqrt(K_terms) * 2e-3 (~0.08 for F=640); its max over many
    elements reaches ~4.5 sigma regardless of magnitude, and at
    cancellation points (pre-log -> 0) the log domain is unbounded.
    Therefore:
    - everywhere: |exp(out) - exp(ref)| < 0.35 (4.5-sigma absolute pre-log
      noise bound, magnitude-independent),
    - 99.9th percentile of |out - ref| over meaningful values
      (ref > -9, i.e. pre-log > 1.2e-4) < 5e-3 (measured ~7e-4; catches
      any systematic mapping error, which the emulation contract makes
      ~100x tighter than a generic TF32-vs-fp32 comparison).
    """
    out_f = out.flatten().float()
    ref_f = ref.flatten().float()
    lin = (torch.exp(out_f) - torch.exp(ref_f)).abs()
    mx = lin.max().item()
    assert mx < lin_tol, f"pre-log |exp diff|={mx:.4e} >= {lin_tol}"
    meaningful = ref_f > -9.0
    assert meaningful.any(), "reference has no meaningful (non-floor) values"
    d = (out_f - ref_f).abs()[meaningful]
    q999 = torch.quantile(d, 0.999).item()
    assert q999 < q999_tol, f"99.9pct|log diff|={q999:.4e} >= {q999_tol}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestPowerMelLogBf16:

    def test_matches_tf32_truncation_emulation(self):
        torch.manual_seed(42)
        spec = torch.randn(128, 257, device="cuda", dtype=torch.cfloat)
        mel = torch.randn(80, 257, device="cuda", dtype=torch.float32)
        ref = emulated_power_mel_log(spec, mel)
        out = power_mel_log(spec, mel, None, None)
        assert out.shape == ref.shape
        _assert_close(out, ref)

    def test_matches_emulation_various_sizes(self):
        for seed, T, F, M in [
            (0, 64, 257, 80),
            (1, 256, 513, 80),
            (7, 2048, 257, 80),
            (3, 98, 129, 64),
        ]:
            torch.manual_seed(seed)
            spec = torch.randn(T, F, device="cuda", dtype=torch.cfloat)
            mel = torch.randn(M, F, device="cuda", dtype=torch.float32)
            ref = emulated_power_mel_log(spec, mel)
            out = power_mel_log(spec, mel, None, None)
            _assert_close(out, ref)

    def test_with_cmvn(self):
        torch.manual_seed(42)
        spec = torch.randn(128, 257, device="cuda", dtype=torch.cfloat)
        mel = torch.randn(80, 257, device="cuda", dtype=torch.float32)
        mean = torch.randn(80, device="cuda")
        istd = torch.rand(80, device="cuda") + 0.5
        ref = emulated_power_mel_log(spec, mel, mean, istd)
        out = power_mel_log(spec, mel, mean, istd)
        _assert_close(out, ref)

    def test_real_mel_fbank_matches_fp32(self):
        """Production regime: non-negative filterbank -> no cancellation -> fp32-grade."""
        for T, F, M in [(128, 257, 80), (256, 513, 80)]:
            torch.manual_seed(42)
            spec = torch.randn(T, F, device="cuda", dtype=torch.cfloat)
            mel = _real_mel_fbank(F, M)
            ref = reference_power_mel_log(spec.real, spec.imag, mel, None, None)
            out = power_mel_log(spec, mel, None, None)
            diff = (out - ref).abs().max().item()
            assert diff < 5e-3, f"T={T},F={F}: max|diff|={diff:.4e} vs fp32"

    def test_tiny_input(self):
        torch.manual_seed(0)
        spec = torch.randn(1, 64, device="cuda", dtype=torch.cfloat)
        mel = torch.randn(1, 64, device="cuda", dtype=torch.float32)
        ref = reference_for_dispatch(spec, mel)
        out = power_mel_log(spec, mel, None, None)
        diff = (out - ref).abs().max().item()
        assert diff < 5e-3, f"T=1 max|diff|={diff}"

    def test_various_T_values(self):
        torch.manual_seed(2)
        F, M = 257, 80
        for T in [1, 8, 16, 32, 64, 98, 127, 128, 256]:
            spec = torch.randn(T, F, device="cuda", dtype=torch.cfloat)
            mel = torch.randn(M, F, device="cuda", dtype=torch.float32)
            ref = reference_for_dispatch(spec, mel)
            out = power_mel_log(spec, mel, None, None)
            assert out.shape == (T, M), f"T={T}: shape {out.shape} != ({T}, {M})"
            diff = (out - ref).abs().max().item()
            assert diff < 3e-2, f"T={T}: max|diff|={diff}"

    def test_output_shape(self):
        torch.manual_seed(42)
        spec = torch.randn(128, 257, device="cuda", dtype=torch.cfloat)
        mel = torch.randn(80, 257, device="cuda", dtype=torch.float32)
        out = power_mel_log(spec, mel, None, None)
        assert out.shape == (spec.shape[0], mel.shape[0])
        assert out.dtype == torch.float32

    def test_numerical_stability(self):
        """log(max(acc, 1.19e-7)) should not produce NaN/inf."""
        torch.manual_seed(3)
        spec = torch.randn(128, 257, device="cuda", dtype=torch.cfloat) * 100
        mel = torch.randn(80, 257, device="cuda", dtype=torch.float32)
        out = power_mel_log(spec, mel, None, None)
        assert torch.isfinite(out).all(), "output contains NaN or inf"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestDenseLinearPrecision:

    def test_mxfp4_forward(self):
        from veloxvoice.models.wenet.torch_conformer import DenseLinear

        torch.manual_seed(10)
        w = torch.randn(64, 128, device="cuda")
        dl = DenseLinear(w).cuda()
        dl.precision = "mxfp4"
        x = torch.randn(32, 128, device="cuda")
        out = dl(x)
        assert out.shape == (32, 64)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_nvfp4_forward(self):
        from veloxvoice.models.wenet.torch_conformer import DenseLinear

        torch.manual_seed(11)
        w = torch.randn(64, 128, device="cuda")
        dl = DenseLinear(w).cuda()
        dl.precision = "nvfp4"
        x = torch.randn(32, 128, device="cuda")
        out = dl(x)
        assert out.shape == (32, 64)
        assert torch.isfinite(out).all()

    def test_fp32_forward(self):
        from veloxvoice.models.wenet.torch_conformer import DenseLinear

        torch.manual_seed(12)
        w = torch.randn(64, 128, device="cuda")
        dl = DenseLinear(w).cuda()
        x = torch.randn(32, 128, device="cuda")
        out = dl(x)
        ref = torch.nn.functional.linear(x, dl.weight, dl.bias)
        diff = (out - ref).abs().max().item()
        assert diff == 0.0, f"fp32 should be exact, diff={diff}"

    def test_mxfp4_3d_input(self):
        from veloxvoice.models.wenet.torch_conformer import DenseLinear

        torch.manual_seed(13)
        w = torch.randn(64, 128, device="cuda")
        dl = DenseLinear(w).cuda()
        dl.precision = "mxfp4"
        x = torch.randn(2, 16, 128, device="cuda")
        out = dl(x)
        assert out.shape == (2, 16, 64)
        assert torch.isfinite(out).all()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
