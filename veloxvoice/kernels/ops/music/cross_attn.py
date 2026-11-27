"""NOTE (yiakwy) : flashinfer decoding attn impl best suit our needs in mega_deocde_step"""
from __future__ import annotations

import torch


def cross_attn_decode(q: torch.Tensor, k_cross: torch.Tensor,
                      v_cross: torch.Tensor, layer: int,
                      *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-attention for decoder ``layer``: ``q`` [H, D] against the
    layer's ``[H, T, D]`` slice of the ``[L, H, T, D]`` cross-KV cache."""
    import flashinfer

    k = k_cross[layer]
    v = v_cross[layer]
    if not k.is_contiguous():
        k = k.contiguous()
        v = v.contiguous()
    
    return flashinfer.single_decode_with_kv_cache(q, k, v, kv_layout="HND")


def cross_attn_decode_all(q_all: torch.Tensor, k_cross: torch.Tensor,
                          v_cross: torch.Tensor) -> list[torch.Tensor]:
    """Batched convenience: ``q_all`` [L, H, D] -> list of [H, D].

    NOTE: in the true decode the layers are strictly sequential, so callers
    use :func:`cross_attn_decode` per layer; this is for profiling the
    batch-everything upper bound.
    """
    return [cross_attn_decode(q_all[l], k_cross, v_cross, l)
            for l in range(q_all.shape[0])]
