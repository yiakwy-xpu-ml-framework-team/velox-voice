"""Python wrappers for velox CUDA ops (TVM-FFI loaded, disk-cached).

Every call runs inside `tvm_ffi.use_torch_stream()` so kernels use torch's
current stream (CUDA-graph capture safe).
"""

from __future__ import annotations

import functools

import tvm_ffi


@functools.cache
def _audio_mod():
    from veloxvoice.kernels.utils import build_cuda_module

    return build_cuda_module(
        "audio_ops", ("velox_power_mel_log.cu",), ("power_mel_log",)
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


def power_mel_log(spec, mel, cmvn_mean, cmvn_istd, out=None):
    """spec [T,F] complex (torch.fft.rfft output, interleaved storage), mel [M,F]
    -> out [T,M]. cmvn_mean/istd: [M] tensors or None."""
    import torch

    T, F = spec.shape
    M = mel.shape[0]
    out = (
        out
        if out is not None
        else torch.empty(T, M, device=spec.device, dtype=mel.dtype)
    )
    zero = torch.zeros(M, device=mel.device, dtype=mel.dtype)  # dummy when no CMVN
    has = 1 if cmvn_mean is not None else 0
    with tvm_ffi.use_torch_stream():
        _audio_mod().power_mel_log(
            spec, mel, cmvn_mean if has else zero, cmvn_istd if has else zero, out, has
        )
    return out


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
    from veloxvoice.kernels.utils import build_cuda_module

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
