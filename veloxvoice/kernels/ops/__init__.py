try:  # CUDA path; unavailable on Apple silicon / CPU-only hosts
    from .cuda_ops import (
        chunk_rel_pos_attn,
        dgx_mxfp4_blkscale_gemm,
        dgx_mxfp4_gemm,
        dw_causal_conv1d,
        fused_layernorm,
        fused_qkv,
        fused_qkv_wgmma,
        mxfp4_gemm,
        nvfp4_linear,
        power_mel_log,
        power_mel_log_mxfp4,
        pwlin_glu,
        qkv_pack,
        reference_fused_qkv,
        reference_layernorm,
        reference_power_mel_log,
        silu_glu,
    )
except ModuleNotFoundError:
    pass

try:  # MLX/Metal path; unavailable when mlx is not installed
    from . import metal_ops
except ModuleNotFoundError:
    metal_ops = None

__all__ = [name for name in (
    "metal_ops",
    "power_mel_log",
    "power_mel_log_mxfp4",
    "reference_power_mel_log",
    "fused_layernorm",
    "reference_layernorm",
    "silu_glu",
    "dw_causal_conv1d",
    "chunk_rel_pos_attn",
    "pwlin_glu",
    "qkv_pack",
    "fused_qkv",
    "fused_qkv_wgmma",
    "reference_fused_qkv",
    "dgx_mxfp4_gemm",
    "dgx_mxfp4_blkscale_gemm",
    "mxfp4_gemm",
    "nvfp4_linear",
) if name in globals() and globals()[name] is not None]
