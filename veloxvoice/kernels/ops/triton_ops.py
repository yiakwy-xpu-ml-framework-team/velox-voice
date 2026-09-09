"""Triton kernels for veloxvoice quantization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _per_row_col_quantize_kernel(
    X,
    OutQ,
    OutS,
    M,
    K,
    stride_x_m,
    stride_x_k,
    stride_q_m,
    stride_q_k,
    BLOCK_KP2: tl.constexpr,
):
    pid = tl.program_id(0)

    cols = tl.arange(0, BLOCK_KP2)
    mask = cols < K // 2

    even = tl.load(
        X + pid * stride_x_m + (cols * 2) * stride_x_k, mask=mask, other=0.0
    ).to(tl.float32)

    odd = tl.load(
        X + pid * stride_x_m + (cols * 2 + 1) * stride_x_k, mask=mask, other=0.0
    ).to(tl.float32)

    # TODO (yiakwy) : rewrite to compute row_max
    abs_even = tl.abs(even)
    abs_odd = tl.abs(odd)
    local_max = tl.maximum(abs_even, abs_odd)
    row_max = tl.max(local_max, axis=0)

    # NOTE (yiakwy) : clamp per row/col max to fp4 range.
    # The stored scale must be the EXACT power of two used for normalization
    # (kernel decodes ue8m0 = 2^(e-127)); ceil keeps scale >= row_max/6 so
    # codes never saturate the e2m1 range.
    s = row_max / 6.0
    s_safe = tl.maximum(s, 1e-6)
    e = tl.ceil(tl.log2(s_safe))
    scale = tl.exp2(e)

    # Normalize by the stored scale
    n_even = even / scale
    n_odd = odd / scale

    ax_even = tl.abs(n_even)
    ax_odd = tl.abs(n_odd)

    # TODO (yiakwy) : vectorize compare (least sqrt distance) with CODE_LUT

    # Encode even columns: sequential thresholds, last write wins
    c_even = tl.zeros([BLOCK_KP2], dtype=tl.uint8)
    c_even = tl.where(ax_even > 0.25, 1, c_even)
    c_even = tl.where(ax_even > 0.75, 2, c_even)
    c_even = tl.where(ax_even > 1.25, 3, c_even)
    c_even = tl.where(ax_even > 1.75, 4, c_even)
    c_even = tl.where(ax_even > 2.5, 5, c_even)
    c_even = tl.where(ax_even > 3.5, 6, c_even)

    sign_e = tl.where(n_even < 0.0, 8, 0).to(tl.uint8)
    c_even = c_even | sign_e

    # Encode odd columns
    c_odd = tl.zeros([BLOCK_KP2], dtype=tl.uint8)
    c_odd = tl.where(ax_odd > 0.25, 1, c_odd)
    c_odd = tl.where(ax_odd > 0.75, 2, c_odd)
    c_odd = tl.where(ax_odd > 1.25, 3, c_odd)
    c_odd = tl.where(ax_odd > 1.75, 4, c_odd)
    c_odd = tl.where(ax_odd > 2.5, 5, c_odd)
    c_odd = tl.where(ax_odd > 3.5, 6, c_odd)

    sign_o = tl.where(n_odd < 0.0, 8, 0).to(tl.uint8)
    c_odd = c_odd | sign_o

    # Pack: lo 4 bits = even, hi 4 bits = odd
    packed = c_even | (c_odd << 4)
    tl.store(OutQ + pid * stride_q_m + cols * stride_q_k, packed, mask=mask)

    # Per-row scale as ue8m0 (e + 127 bias); decode = 2^e == normalization scale
    e_byte = tl.clamp(e + 127.0, 0.0, 255.0).to(tl.uint8)

    tl.store(OutS + pid, e_byte)


def per_row_col_quantize(w, block_size=None):
    """[M, K] bf16/fp32 quantize (packed u8 [M, K/2], per-row ue8m0 u8 [M, 1])."""
    assert w.ndim == 2

    M, K = w.shape

    assert K % 2 == 0, "K must be even"
    assert K <= 65536, f"K={K} exceeds max block size"

    # TODO (yiakwy) : remove
    w_c = w.contiguous().view(M, K).to(w.dtype)

    out_q = torch.empty(M, K // 2, device=w.device, dtype=torch.uint8)

    # TODO (yiakwy) : add support
    out_s = torch.empty(M, device=w.device, dtype=torch.uint8)

    BLOCK_KP2 = triton.next_power_of_2(K // 2)

    _per_row_col_quantize_kernel[(M,)](
        w_c,
        out_q,
        out_s,
        M,
        K,
        w_c.stride(0),
        w_c.stride(1),
        out_q.stride(0),
        out_q.stride(1),
        BLOCK_KP2=BLOCK_KP2,
    )

    return out_q, out_s


# Backward-compat alias used by cuda_ops.nvfp4_linear / bench_nvfp4_linear.
def triton_quantize_w(w, block_size=None):
    return per_row_col_quantize(w, block_size)


MXFP4_CODE_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")


# ── mxfp4: per (row-pair | row) × 16-column ue8m0 block quantization ─────────
#
# Hardware contract (probed on sm_121a):
#   scale_vec::2X thread slot = own 16-k slice (two 8-wide halves); the A-side
#   scale is shared across the MMA row-pair (2g,2g+1); B-side is per column.
# Therefore:
#   mode="A": block = (row pair) x 16 cols  -> ue8m0 grid [K/16][R/2]
#   mode="B": block = (row     ) x 16 cols  -> ue8m0 grid [K/16][R]


@triton.jit
def _fp4_encode(n):
    """float32 -> uint8 code (sign|mag3): round-to-nearest thresholds."""
    ax = tl.abs(n)
    c = tl.zeros(n.shape, dtype=tl.uint8)
    c = tl.where(ax > 0.25, 1, c)
    c = tl.where(ax > 0.75, 2, c)
    c = tl.where(ax > 1.25, 3, c)
    c = tl.where(ax > 1.75, 4, c)
    c = tl.where(ax > 2.5, 5, c)
    c = tl.where(ax > 3.5, 6, c)
    return c | tl.where(n < 0.0, 8, 0).to(tl.uint8)


@triton.jit
def _ue8m0_vec(bs):
    """[NB] float32 block scales -> ue8m0 u8 (trunc log2 + 127)."""
    safe = tl.maximum(bs, 1e-20)
    log2v = tl.log2(safe)
    trunc = tl.where(log2v >= 0, tl.floor(log2v), tl.ceil(log2v))
    e = tl.clamp(trunc + 127.0, 0.0, 255.0).to(tl.uint8)
    return tl.where(bs > 1e-20, e, 0x80).to(tl.uint8)


@triton.jit
def _mxfp4_quantize_kernel(
    X,
    OutQ,
    OutS,
    R2,
    K,
    RP: tl.constexpr,
    stride_x_r,
    stride_x_k,
    stride_q_r,
    stride_q_k,
    BLOCK_KP2: tl.constexpr,
):
    """One program per quantization row-group.

    RP=2: program pid quantizes rows (2pid, 2pid+1) with SHARED per-block scale.
    RP=1: program pid quantizes row pid.
    Scale output layout (kb-major): OutS[kb * R2 + pid].
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_KP2)  # e/o pair index along K
    mask = cols < K // 2

    if RP == 2:
        a_e = tl.load(
            X + (2 * pid) * stride_x_r + (cols * 2) * stride_x_k, mask=mask, other=0.0
        ).to(tl.float32)
        a_o = tl.load(
            X + (2 * pid) * stride_x_r + (cols * 2 + 1) * stride_x_k,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        b_e = tl.load(
            X + (2 * pid + 1) * stride_x_r + (cols * 2) * stride_x_k,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        b_o = tl.load(
            X + (2 * pid + 1) * stride_x_r + (cols * 2 + 1) * stride_x_k,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        am = tl.maximum(
            tl.maximum(tl.abs(a_e), tl.abs(a_o)), tl.maximum(tl.abs(b_e), tl.abs(b_o))
        )
        b_n_e = b_e
        b_n_o = b_o
    else:
        a_e = tl.load(
            X + pid * stride_x_r + (cols * 2) * stride_x_k, mask=mask, other=0.0
        ).to(tl.float32)
        a_o = tl.load(
            X + pid * stride_x_r + (cols * 2 + 1) * stride_x_k, mask=mask, other=0.0
        ).to(tl.float32)
        am = tl.maximum(tl.abs(a_e), tl.abs(a_o))
        b_n_e = a_e
        b_n_o = a_e

    # Per 16-col block absmax: one block = 8 e/o pairs -> grid [K/16, 8]
    # Stored scale = 2^ceil(log2(block_max/6)) >= block_max/6 (no saturation);
    # codes are normalized by that exact power of two so decode == encode.
    ab = tl.reshape(am, (BLOCK_KP2 // 8, 8))
    bs = tl.max(ab, axis=1) / 6.0  # [K/16] block scales
    bs = tl.maximum(bs, 1e-6)
    e_blk = tl.ceil(tl.log2(bs))  # [K/16]
    scale_blk = tl.exp2(e_blk)
    bcols = tl.reshape(
        tl.broadcast_to(scale_blk[:, None], (BLOCK_KP2 // 8, 8)), (BLOCK_KP2,)
    )

    # Row a: normalize + encode + pack
    ca_e = _fp4_encode(a_e / bcols)
    ca_o = _fp4_encode(a_o / bcols)
    packed_a = ca_e | (ca_o << 4)
    q_row = (2 * pid) if RP == 2 else pid
    tl.store(OutQ + q_row * stride_q_r + cols * stride_q_k, packed_a, mask=mask)
    if RP == 2:
        cb_e = _fp4_encode(b_n_e / bcols)
        cb_o = _fp4_encode(b_n_o / bcols)
        packed_b = cb_e | (cb_o << 4)
        tl.store(
            OutQ + (2 * pid + 1) * stride_q_r + cols * stride_q_k, packed_b, mask=mask
        )

    # Per-block scale: ue8m0 (e + 127), kb-major layout [K/16][R2]
    kb = tl.arange(0, BLOCK_KP2 // 8)
    k_mask = kb < K // 16
    e_byte = tl.clamp(e_blk + 127.0, 0.0, 255.0).to(tl.uint8)
    tl.store(OutS + kb * R2 + pid, e_byte, mask=k_mask)


def mxfp4_block_quantize(w, mode: str = "A"):
    """[R, K] bf16/fp32 -> (packed u8 [R, K/2], ue8m0 u8 [K/16][R2], kb-major).

    mode: "A" row-pair coupled (GEMM lhs); "B" per-row (GEMM rhs columns).
    """
    assert w.ndim == 2
    R, K = w.shape
    assert K % 16 == 0, "K must be divisible by 16"
    RP = 2 if mode == "A" else 1
    assert mode != "A" or R % 2 == 0, "mode A requires even R (row pairs)"
    R2 = R // RP

    w_c = w.contiguous().view(R, K).to(w.dtype)
    out_q = torch.empty(R, K // 2, device=w.device, dtype=torch.uint8)
    out_s = torch.empty(K // 16, R2, device=w.device, dtype=torch.uint8)

    BLOCK_KP2 = triton.next_power_of_2(K // 2)

    _mxfp4_quantize_kernel[(R2,)](
        w_c,
        out_q,
        out_s,
        R2,
        K,
        RP,
        w_c.stride(0),
        w_c.stride(1),
        out_q.stride(0),
        out_q.stride(1),
        BLOCK_KP2=BLOCK_KP2,
    )

    return out_q, out_s


def mxfp4_dequant(q, s, mode: str = "A"):
    """Reference expand: (packed [R, K/2], kb-major ue8m0 [K/16][R2]) -> [R, K]."""
    R = q.shape[0]
    K = q.shape[1] * 2
    lo = q & 0xF
    hi = q >> 4
    c = torch.stack([lo, hi], -1).reshape(R, K)
    lut = MXFP4_CODE_LUT.to(q.device)
    mag = lut[(c & 7).long()]
    codes = torch.where((c & 8) > 0, -mag, mag)
    sv = 2.0 ** (s.to(torch.int16).to(torch.float32) - 127)
    if mode == "A":
        sv = sv.repeat_interleave(2, dim=1)  # [K/16, R] per row-pair
    return codes * sv.T.repeat_interleave(16, dim=1)  # [K/16][K/16->K] per block
