"""Python wrappers for velox CUDA ops (TVM-FFI loaded, disk-cached).

Every call runs inside `tvm_ffi.use_torch_stream()` so kernels use torch's
current stream (CUDA-graph capture safe).
"""

from __future__ import annotations

import functools

import tvm_ffi


@functools.cache
def _audio_mod():
    from veloxvoice.kernels.utils import CSRC, build_cuda_module

    return build_cuda_module(
        "audio_ops",
        ("dgx/dgx_power_mel_log.cu",),
        ("power_log_bf16",),
        extra_cuda_cflags=(
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            f"-I{CSRC}",
            f"-I{CSRC}/dgx",
        ),
        arch_override="12.1a",
    )


@functools.cache
def _norm_mod():
    from veloxvoice.kernels.utils import build_cuda_module

    return build_cuda_module("norm_ops", ("velox_layernorm.cu",), ("layernorm",))


@functools.cache
def _conv_mod():
    from veloxvoice.kernels.utils import build_cuda_module

    return build_cuda_module(
        "conv_ops",
        ("velox_silu_glu.cu", "velox_depthwise_causal_conv1d.cu"),
        ("silu_glu", "dw_causal_conv1d"),
    )


def power_mel_log(spec, mel, cmvn_mean, cmvn_istd):
    """spec [T,F] complex (torch.fft.rfft output, interleaved storage), mel [M,F]
    -> out [T,M]. cmvn_mean/istd: [M] tensors or None.

    Dispatch:
      - M <= 80 AND T >= 64: CUDA fused TMA pipeline (tf32 persistent GEMM)
      - Otherwise: cuBLAS GEMM (power→matmul→log)
    """
    import torch

    T, F = spec.shape
    M = mel.shape[0]

    BK, BM, BN = 64, 64, 80
    F_padded = ((F + BK - 1) // BK) * BK
    # Ensure F_padded / BK is even for double-buffer
    if (F_padded // BK) % 2 != 0:
        F_padded += BK

    # Fused TMA kernel path: M <= BN=80 AND T >= BM=64
    if M <= BN and T >= BM:
        BK, BM, BN = 64, 64, 80
        F_padded = ((F + BK - 1) // BK) * BK
        if (F_padded // BK) % 2 != 0:
            F_padded += BK

        # mel_g must be [F_padded, M] for kernel: mel_g[f * M + n]
        # Original mel is [M, F], transpose to [F, M], pad to [F_padded, M]
        mel_g = torch.zeros(F_padded, M, device=mel.device, dtype=mel.dtype)
        mel_g[:F, :M] = mel.t()  # [F, M] → pad to [F_padded, M]

        out = torch.empty(T, M, device=spec.device, dtype=torch.float32)
        zero = torch.zeros(M, device=mel.device, dtype=mel.dtype)
        has = 1 if cmvn_mean is not None else 0
        with tvm_ffi.use_torch_stream():
            _audio_mod().power_log_bf16(
                spec.contiguous(),
                mel_g,
                cmvn_mean if has else zero,
                cmvn_istd if has else zero,
                out,
                has,
            )
        return out


def _power_mel_log_cublas(spec, mel, cmvn_mean, cmvn_istd):
    """cuBLAS fast path: power → matmul → log. No padding overhead."""
    import torch

    power = spec.real**2 + spec.imag**2  # [T, F]
    feat = power @ mel.T  # [T, M] — cuBLAS GEMM
    feat = torch.log(torch.clamp_min(feat, 1.1920929e-7))
    if cmvn_mean is not None and cmvn_istd is not None:
        feat = (feat - cmvn_mean.unsqueeze(0)) * cmvn_istd.unsqueeze(0)
    return feat


def power_mel_log_mxfp4(
    spec, mel_codes, mel_scales, cmvn_mean, cmvn_istd, F_padded, M_padded
):
    """Composed fallback: power(torch) → mxfp4_block_quantize → gemm → log → CMVN.

    spec: [T, F] complex interleaved (torch.fft.rfft output).
    mel_codes: [M_padded, K/2] u8 pre-quantized mel codes (F_padded = K = multiple of 64).
    mel_scales: [M_padded] u8 ue8m0 per-col scales.
    F_padded: F padded to multiple of BK=64.
    M_padded: M padded to multiple of BN=128.
    -> out [T, M_padded] f32 (only first M columns are valid).
    """
    import torch

    from veloxvoice.kernels.ops.cuda_ops import dgx_mxfp4_blkscale_gemm
    from veloxvoice.kernels.ops.triton_ops import mxfp4_block_quantize

    T, F = spec.shape

    # Step 1: Compute power (re² + im²) — online, not pre-quantized
    power = spec.real**2 + spec.imag**2  # [T, F]

    # Step 2: Quantize power activations (online quantize A)
    power_padded = torch.zeros(T, F_padded, device=spec.device, dtype=power.dtype)
    power_padded[:, :F] = power
    a_codes, a_scales = mxfp4_block_quantize(power_padded, mode="A")

    # Step 3: GEMM via mxfp4 block-scale kernel
    out_padded = dgx_mxfp4_blkscale_gemm(a_codes, mel_codes, a_scales, mel_scales)

    # Step 4: Log + CMVN
    out_padded = torch.log(torch.clamp_min(out_padded, 1.1920929e-7))
    if cmvn_mean is not None and cmvn_istd is not None:
        out_padded = (out_padded - cmvn_mean.unsqueeze(0)) * cmvn_istd.unsqueeze(0)

    # Slice to valid M columns
    return out_padded[:, :M_padded]


def reference_power_mel_log(re, im, mel, cmvn_mean, cmvn_istd):
    power = re.float() ** 2 + im.float() ** 2
    feat = power @ mel.T
    feat = feat.clamp_min(1.1920929e-7).log()
    if cmvn_mean is not None:
        feat = (feat - cmvn_mean) * cmvn_istd
    return feat


def fused_layernorm(x, w, b, eps=1e-5, out=None):
    import torch

    out = out if out is not None else torch.empty_like(x)
    with tvm_ffi.use_torch_stream():
        _norm_mod().layernorm(x, w, b, out, float(eps))
    return out


def reference_layernorm(x, w, b, eps=1e-5):
    import torch.nn.functional as F

    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def silu_glu(x, out=None):
    import torch

    T, C2 = x.shape
    out = (
        out
        if out is not None
        else torch.empty(T, C2 // 2, device=x.device, dtype=x.dtype)
    )
    with tvm_ffi.use_torch_stream():
        _conv_mod().silu_glu(x, out)
    return out


@functools.cache
def _attn_mod():
    from veloxvoice.kernels.utils import CSRC, build_cuda_module

    return build_cuda_module(
        "attn_ops",
        ("velox_chunk_rel_pos_attn.cu", "velox_fused_qkv.cu"),
        ("chunk_rel_pos_attn", "fused_qkv"),
    )


def dw_causal_conv1d(x, cache, weight, bias, outs=None):
    """x [T,C]; cache [K-1,C]; weight [K,C]; bias [C] -> (out [T,C], new_cache)."""
    import torch

    out = torch.empty_like(x) if outs is None else outs[0]
    new_cache = torch.empty_like(cache) if outs is None else outs[1]
    with tvm_ffi.use_torch_stream():
        _conv_mod().dw_causal_conv1d(x, cache, weight, bias, out, new_cache)
    return out, new_cache


def chunk_rel_pos_attn(q_u, q_v, k, pe, idx, v, valid=None, outs=None):
    """q_u/q_v [H,Tq,DK]; k/v [H,L,DK]; pe [SPAN,H*DK]; idx [Tq,L] int32;
    valid [H,Tq] int32 counts (default = full L for every (h,i)).
    -> (probs [H,Tq,L], out [H,Tq,DK])"""
    import torch

    H, Tq, DK = q_u.shape
    L = v.shape[1]
    if valid is None:
        valid = torch.full((H, Tq), L, device=v.device, dtype=torch.int32)
    probs = (
        torch.empty(H, Tq, L, device=v.device, dtype=torch.float32)
        if outs is None
        else outs[0]
    )
    out = (
        torch.empty(H, Tq, DK, device=v.device, dtype=torch.float32)
        if outs is None
        else outs[1]
    )
    with tvm_ffi.use_torch_stream():
        _attn_mod().chunk_rel_pos_attn(q_u, q_v, k, pe, idx, valid, probs, v, out)
    return probs, out


def fused_qkv(x, w, bias, out=None):
    """x [M,K]; w [3N,K]; bias [3N] -> out [M, 3N]."""
    import torch

    M, K = x.shape
    out = (
        out
        if out is not None
        else torch.empty(M, w.shape[0], device=x.device, dtype=x.dtype)
    )
    with tvm_ffi.use_torch_stream():
        _attn_mod().fused_qkv(x.contiguous(), w.contiguous(), bias.contiguous(), out)
    return out


def reference_fused_qkv(x, w, bias):
    import torch.nn.functional as F

    return F.linear(x, w, bias)


def _build_dgx_mod(
    num_producer_warps=1, num_consumer_warps=8, group_size_m=16, cluster_size_m=1
):
    """Build dgx_mxfp4_gemm with P producer warps and C consumer warps."""
    from veloxvoice.kernels.utils import CSRC, build_cuda_module

    return build_cuda_module(
        f"dgx_ops_p{num_producer_warps}_c{num_consumer_warps}_g{group_size_m}_cl{cluster_size_m}",
        ("dgx/dgx_mxfp4_gemm.cu",),
        (),
        extra_cuda_cflags=(
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            f"-I{CSRC}",
            f"-I{CSRC}/dgx",
            f"-DNUM_PRODUCER_WARPS={num_producer_warps}",
            f"-DNUM_CONSUMER_WARPS={num_consumer_warps}",
            f"-DK_GROUP_SIZE_M={group_size_m}",
            f"-DK_CLUSTER_SIZE_M={cluster_size_m}",
        ),
        arch_override="12.1a",
    )


@functools.cache
def _dgx_mod(
    num_producer_warps=1, num_consumer_warps=8, group_size_m=16, cluster_size_m=1
):
    return _build_dgx_mod(
        num_producer_warps, num_consumer_warps, group_size_m, cluster_size_m
    )


def dgx_mxfp4_gemm(
    pack_a,
    pack_b,
    scale_a,
    scale_b,
    num_producer_warps=1,
    num_consumer_warps=8,
    group_size_m=16,
    cluster_size_m=1,
):
    """pack_a [M, K/2] u8; pack_b [N, K/2] u8; scale tensor u8; -> out [M, N] f32."""
    import torch

    M, K2 = pack_a.shape
    out = torch.empty(M, pack_b.shape[0], device=pack_a.device, dtype=torch.float32)
    with tvm_ffi.use_torch_stream():
        _dgx_mod(
            num_producer_warps, num_consumer_warps, group_size_m, cluster_size_m
        ).dgx_mxfp4_gemm(pack_a, pack_b, scale_a, scale_b, out)
    return out


def nvfp4_linear(x, wq, ws, P=4, C=8, w_bf16=None):
    """Smart dispatch: use cuBLAS for small shapes, nvfp4 GEMM for large.

    For small M, the GPU is underutilized (fewer blocks than SMs) and
    cuBLAS's highly-tuned small-M kernels win. For large M, nvfp4 GEMM
    dominates due to the packed format advantage.

    w_bf16: optional pre-dequantized bf16 weight [N, K] for cuBLAS fallback.
             If None, dequantizes on the fly (slower).
    """
    import torch

    from veloxvoice.kernels.ops.triton_ops import triton_quantize_w

    M = x.shape[0]
    K = x.shape[-1]
    N = wq.shape[0]

    if M <= 512:
        # Small M: GPU underutilized, cuBLAS wins
        if w_bf16 is None:
            w_bf16 = _dequantize_to_bf16(wq, ws)
        out = torch.mm(x.view(-1, K).to(torch.bfloat16), w_bf16.T)
        return out.view(*x.shape[:-1], N)

    x_2d = x.view(-1, K)
    xq, xs = triton_quantize_w(x_2d)
    out = dgx_mxfp4_gemm(xq, wq, xs, ws, P, C)
    return out.view(*x.shape[:-1], N)


def dgx_mxfp4_blkscale_gemm(
    pack_a,
    pack_b,
    scale_a,
    scale_b,
    num_producer_warps=1,
    num_consumer_warps=8,
    group_size_m=16,
    cluster_size_m=1,
):
    """mxfp4 block-scale GEMM (per-16-col ue8m0, scales consumed inside MMA).

    pack_a [M, K/2] u8; pack_b [N, K/2] u8;
    scale_a: ue8m0 u8 [K/16][M/2]; scale_b: ue8m0 u8 [K/16][N]  -> out [M, N] f32.
    """
    import torch

    M, _ = pack_a.shape
    out = torch.empty(M, pack_b.shape[0], device=pack_a.device, dtype=torch.float32)
    with tvm_ffi.use_torch_stream():
        _dgx_mod(
            num_producer_warps, num_consumer_warps, group_size_m, cluster_size_m
        ).dgx_mxfp4_blkscale_gemm(pack_a, pack_b, scale_a, scale_b, out)
    return out


def mxfp4_gemm(xq, wq, xs, ws, P=1, C=8, group_size_m=16):
    """Convenience: block-scaled mxfp4 gemm of pre-quantized tensors."""
    return dgx_mxfp4_blkscale_gemm(xq, wq, xs, ws, P, C, group_size_m)


def _dequantize_to_bf16(packed, scale_u8):
    """packed u8 [N, K/2] + ue8m0 u8 [N] → bf16 [N, K]."""
    import torch

    from veloxvoice.models.wenet.nvfp4_linear import CODE_LUT

    lo = packed & 0xF
    hi = packed >> 4
    c = torch.stack([lo, hi], -1).reshape(packed.shape[0], packed.shape[1] * 2)
    mag = CODE_LUT.to(packed.device)[(c & 7).long()]
    sign = (c & 8) > 0
    w = torch.where(sign, -mag, mag)
    s = (2.0 ** (scale_u8.to(torch.int16).to(torch.float32) - 127)).unsqueeze(1)
    return (w * s).to(torch.bfloat16)
