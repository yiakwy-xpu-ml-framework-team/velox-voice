"""Tests for `veloxvoice.kernels.ops.dgx_mxfp4_gemm`.

Methodology (user-mandated):
  1. Generate torch bf16 data: x [M, K], w [N, K]
  2. Quantize x -> (xq, xs), w -> (wq, ws)
  3. Compute GEMM: out = dgx_mxfp4_gemm(xq, wq, xs, ws)
  4. Reference: ref[i][j] = (unpack(xq) @ unpack(wq).T)[i][j] * xs[i] * ws[j]
     (kernel applies e8m0 scales in epilogue, not element-wise before dot)
  5. Compare out vs ref
"""

from __future__ import annotations

import pytest
import torch

from veloxvoice.kernels.ops import dgx_mxfp4_gemm
from veloxvoice.models.wenet.nvfp4_linear import (
    CODE_LUT,
    _encode_codes,
    dequantize_w,
    pack_codes,
    quantize_w,
    ue8m0_from_scale,
    ue8m0_to_float,
)


def unpack(p):
    """packed u8 [R, K/2] -> signed magnitudes [R, K] (e2m1 with sign bit 3)."""
    lo = p & 0xF
    hi = p >> 4
    c = torch.stack([lo, hi], -1).reshape(p.shape[0], p.shape[1] * 2)
    mag = CODE_LUT.to(p.device)[(c & 7).long()]
    sign = (c & 8) > 0  # bit 3 = sign (bool)
    return torch.where(sign, -mag, mag)


def torch_matmul_ref(xq, wq, sa_u8, sb_u8):
    """Reference GEMM matching kernel semantics:
    raw = unpack(xq) @ unpack(wq).T   (MMA uses unit scales)
    ref = raw * row_scales * col_scales (epilogue applies e8m0 scales)
    """
    xu = unpack(xq)  # [M, K] signed magnitudes
    wu = unpack(wq)  # [N, K] signed magnitudes
    raw = xu @ wu.T  # [M, N] fp32 (unit-scale dot)

    # vectorized e8m0 -> fp32
    row_s = (2.0 ** (sa_u8.to(torch.int16).to(torch.float32) - 127)).unsqueeze(
        1
    )  # [M, 1]
    col_s = (2.0 ** (sb_u8.to(torch.int16).to(torch.float32) - 127)).unsqueeze(
        0
    )  # [1, N]
    return raw * row_s * col_s  # [M, N] fp32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestDgxMxfp4Gemm:

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (128, 128, 64),  # min tile
            (128, 256, 64),
            (256, 128, 128),
            (512, 512, 2048),  # multi-tile
            (1024, 512, 1024),
            (2048, 2048, 2048),  # large
        ],
    )
    def test_bit_exact_methodology(self, M, N, K):
        torch.manual_seed(0)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

        xq, xs = quantize_w(x)
        wq, ws = quantize_w(w)

        out = dgx_mxfp4_gemm(xq, wq, xs, ws)
        ref = torch_matmul_ref(xq, wq, xs, ws)

        assert out.shape == (M, N), f"shape mismatch: {out.shape}"
        assert out.dtype == torch.float32
        diff = (out - ref).abs().max().item()
        assert diff == 0.0, f"bit-exact fail: max|diff|={diff}"

    def test_same_shape_new_tensors(self):
        """Same shapes, different tensors -> must not reuse stale TMA descriptors."""
        torch.manual_seed(1)
        M, N, K = 512, 512, 2048
        x1 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        xq1, xs1 = quantize_w(x1)
        wq1, ws1 = quantize_w(w1)
        out1 = dgx_mxfp4_gemm(xq1, wq1, xs1, ws1)
        ref1 = torch_matmul_ref(xq1, wq1, xs1, ws1)
        assert (out1 - ref1).abs().max().item() == 0.0

        x2 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        xq2, xs2 = quantize_w(x2)
        wq2, ws2 = quantize_w(w2)
        out2 = dgx_mxfp4_gemm(xq2, wq2, xs2, ws2)
        ref2 = torch_matmul_ref(xq2, wq2, xs2, ws2)
        assert (out2 - ref2).abs().max().item() == 0.0
        assert (out1 - out2).abs().max().item() > 0.0

    def test_unit_scale_parity(self):
        """Both sides 0x7F (unit) -> should match raw unpack matmul."""
        torch.manual_seed(2)
        M, N, K = 256, 256, 128
        pa = torch.randint(0, 8, (M, K), device="cuda", dtype=torch.uint8)
        pb = torch.randint(0, 8, (N, K), device="cuda", dtype=torch.uint8)
        sa = torch.full((M,), 0x7F, dtype=torch.uint8, device="cuda")
        sb = torch.full((N,), 0x7F, dtype=torch.uint8, device="cuda")
        out = dgx_mxfp4_gemm(pack_codes(pa), pack_codes(pb), sa, sb)
        ref = unpack(pack_codes(pa)) @ unpack(pack_codes(pb)).T
        assert (out - ref).abs().max().item() == 0.0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
