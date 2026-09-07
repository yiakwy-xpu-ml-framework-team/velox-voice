"""NVFP4/MXFP4 block-scale quantization-fused GEMM.

Formats:
  nvfp4:  per-row e8m0 scales, kernel applies 2^(xs-127) * 2^(ws-127) in epilogue.
  mxfp4:  same packing, MX format scales (block-level), MMA handles scaling.

Usage pattern (production):
  1. Pre-quantize weight once: wq, ws = quantize_w(w)
  2. On-line quantize activation per call: xq, xs = quantize_w(x)
  3. GEMM: out = dgx_mxfp4_gemm(xq, wq, xs, ws)
"""

import torch

from veloxvoice.kernels.ops.cuda_ops import dgx_mxfp4_gemm
from veloxvoice.kernels.ops.triton_ops import per_row_col_quantize

BLOCK = 16
CODE_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")


def _encode_codes(w):
    """Vectorized encode: float -> uint8 codes (sign|mag3).
    Uses arithmetic sum of comparisons — fully torch.compile compatible."""
    ax = w.abs()
    sign = (w < 0).to(torch.uint8) << 3
    # Each comparison adds 1 if true → magnitude code 0..6
    c = (
        (ax > 0.25).to(torch.uint8)
        + (ax > 0.75).to(torch.uint8)
        + (ax > 1.25).to(torch.uint8)
        + (ax > 1.75).to(torch.uint8)
        + (ax > 2.5).to(torch.uint8)
        + (ax > 3.5).to(torch.uint8)
    )
    return c | sign


def pack_codes(c):
    return ((c[:, 0::2] & 0xF) | ((c[:, 1::2] & 0xF) << 4)).cuda()


def ue8m0_from_scale(s):
    if s <= 0.0:
        return 0x80
    import math

    e = math.trunc(math.log2(s)) + 127
    return max(0, min(255, e))


def ue8m0_to_float(u):
    return float(2.0 ** (int(u) - 127))


def _ue8m0_from_scale_vec(s):
    """[R] float -> [R] uint8 e8m0, fully vectorized (no in-place ops).
    Uses trunc (round toward zero) to match math.trunc(math.log2(s))."""
    positive = s.clamp(min=1e-20)
    log2_s = torch.trunc(torch.log2(positive)).to(torch.int16)
    e = (log2_s + 127).clamp(0, 255).to(torch.uint8)
    e = torch.where(s <= 0.0, torch.tensor(0x80, dtype=torch.uint8, device=s.device), e)
    return e


def quantize_w(w, block_size=BLOCK, format="nvfp4"):
    """[M, K] bf16/fp32 quantize (packed u8 [M, K/2], per-row ue8m0 u8 [M])."""
    return per_row_col_quantize(w, block_size)


_quantize_stream = None


def _get_quantize_stream():
    global _quantize_stream
    if _quantize_stream is None:
        _quantize_stream = torch.cuda.Stream()
    return _quantize_stream


def quantize_and_gemm(x, wq, ws, P=1, C=8):
    """PDL: quantize x on a separate stream, then GEMM on default stream.

    Overlaps quantization compute with GEMM
    """
    quant_stream = _get_quantize_stream()

    # Quantize x on separate stream
    with torch.cuda.stream(quant_stream):
        xq, xs = quantize_w(x)

    # Record event after quantization
    quant_done = torch.cuda.Event()
    quant_done.record(quant_stream)

    # NOTE (yiakwy) : Switch back to default stream, wait for quantization
    torch.cuda.current_stream().wait_event(quant_done)

    out = dgx_mxfp4_gemm(xq, wq, xs, ws, P, C)
    return out


def decode_codes(packed):
    """packed u8 [R, K/2] → codes [R, K] (uint8, sign|mag3)."""
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    return torch.stack([lo, hi], -1).reshape(packed.shape[0], packed.shape[1] * 2)


def _dequantize_impl(q, s_u8):
    """packed codes + per-row ue8m0 u8 → [R, K] bf16."""
    codes = decode_codes(q)
    mag = CODE_LUT.to(q.device)[(codes & 7).long()]
    sign = (codes & 8) > 0
    dx = mag * torch.where(sign, -1.0, 1.0).to(q.device)
    sv = _ue8m0_from_scale_vec(
        torch.tensor(
            [ue8m0_to_float(int(u)) for u in s_u8], device=q.device, dtype=torch.float32
        )
    ).to(torch.bfloat16)
    return dx.to(torch.bfloat16) * sv.view(-1, 1)


dequantize_w = _dequantize_impl
unquantize_w = _dequantize_impl


class Nvfp4Linear(torch.nn.Module):
    """Pre-quantized weight, online activation quantization."""

    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        w = torch.randn(out_features, in_features, device=device, dtype=torch.bfloat16)
        wq, ws = quantize_w(w)
        self.register_buffer("weight_q", wq)
        self.register_buffer("weight_s", ws)
        # Pre-dequantize for cuBLAS fallback on small M
        from veloxvoice.kernels.ops.cuda_ops import _dequantize_to_bf16

        self.register_buffer("weight_bf16", _dequantize_to_bf16(wq, ws))
        if bias:
            self.register_buffer(
                "bias", torch.zeros(out_features, device=device, dtype=torch.float32)
            )
        else:
            self.bias = None

    def forward(self, x):
        from veloxvoice.kernels.ops.cuda_ops import nvfp4_linear

        orig_shape = x.shape
        x = x.view(-1, self.in_features)
        out = nvfp4_linear(x, self.weight_q, self.weight_s, w_bf16=self.weight_bf16)
        if self.bias is not None:
            out = out + self.bias
        return out.view(*orig_shape[:-1], self.out_features)


def nvfp4_gemm(x, w, b=None):
    """x [M, K] @ w [N, K]^T — full online quantize (both sides)."""
    xq, xs = quantize_w(x)
    wq, ws = quantize_w(w)
    out = dgx_mxfp4_gemm(xq, wq, xs, ws)
    if b is not None:
        out = out + b
    return out
