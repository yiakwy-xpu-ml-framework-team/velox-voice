"""CUDA ops for SheetSage2 fused decoder step.

Kept separate from ``kernels/ops/cuda_ops.py`` to avoid circular imports,
and is imported lazily by ``models/sheetsage/v2/graph_decoder.py``.

Split a Mega Decode Step into 3 stages of captures is 26x faster and 
more than 2x faster than whole cuda graph capture.
"""

from __future__ import annotations

import functools


# NOTE (yiakwy) : only used for profling 
@functools.cache
def _sheetsage_mod():
    from veloxvoice.kernels.helper_cuda import velox_arch_str
    from veloxvoice.kernels.utils import build_cuda_module

    from veloxvoice.kernels.utils import CSRC

    return build_cuda_module(
        "music_sheetsage_ops",
        ("velox_sheetsage_mega_step.cu",),
        ("velox_sheetsage_mega_decode_step", "velox_ss_probe", "velox_ss_sum_probe"),
        extra_cuda_cflags=(f"-I{CSRC}",),
        arch_override=velox_arch_str(),
    )


def sheetsage_step(x_in, pos, allowed, w_blob, token_emb, pos_table,
                   k_self, v_self, k_cross, v_cross, lm_head,
                   scratch, red, logits, token_out, dbg, S: int, T: int):
    """One fused decoder step.

    ``w_blob`` packs all per-layer weights (layout documented in the .cu);
    ``scratch`` = h|qkv|ffn|attn|tmp (bf16); ``red`` = pm2|pacc2|cmax|cidx
    (fp32, sized for up to 2048 blocks).
    """
    mod = _sheetsage_mod()
    mod.velox_sheetsage_mega_decode_step(
        x_in, pos, allowed, w_blob, token_emb, pos_table,
        k_self, v_self, k_cross, v_cross, lm_head,
        scratch, red, logits, token_out, dbg, int(S), int(T))


def ss_probe(out):
    """Minimal cooperative-grid probe (grid.sync + write)."""
    _sheetsage_mod().velox_ss_probe(out)


def ss_sum_probe(out):
    """block_sum isolation probe: out[i] must all equal sum(1..512)."""
    _sheetsage_mod().velox_ss_sum_probe(out)


@functools.cache
def _step_kernels_mod():
    """Multi-SM phase kernels for the K2/K3 stages."""
    from veloxvoice.kernels.helper_cuda import velox_arch_str
    from veloxvoice.kernels.utils import build_cuda_module, CSRC

    return build_cuda_module(
        "music_sheetsage_step_ops",
        ("velox_sheetsage_step_kernels.cu",),
        ("ss_gemv", "ss_ln_res", "ss_self_attn", "ss_kvstore", "ss_lmhead"),
        extra_cuda_cflags=(f"-I{CSRC}",),
        arch_override=velox_arch_str(),
    )


def ss_gemv(w, b, x, y, N: int, K: int, blocks: int = 64):
    """Multi-SM GEMV y = b + W @ x (grid strides over the N rows)."""
    _step_kernels_mod().ss_gemv(w, b if b is not None else x, x, y,
                                int(N), int(K), int(blocks))


def ss_ln_res(h, res, w, b, eps: float = 1e-5):
    """residual add (if res is not h) + LayerNorm over the [512] vector."""
    _step_kernels_mod().ss_ln_res(h, res if res is not None else h, w, b,
                                  float(eps))


def ss_self_attn(q, K, V, out, S: int, rows_dev, scale: float, heads: int):
    _step_kernels_mod().ss_self_attn(q, K, V, out, rows_dev, float(scale),
                                     int(heads), int(S))


def ss_kvstore(qkv, K, V, S: int, pos_dev):
    _step_kernels_mod().ss_kvstore(qkv, K, V, int(S), pos_dev)


def ss_lmhead(lm, h, allowed, pbest, pidx, V: int, K: int, blocks: int = 64):
    _step_kernels_mod().ss_lmhead(lm, h, allowed, pbest, pidx, int(V), int(K),
                                  int(blocks))


def ss_gemv_ln(w, b, h, res, lnw, lnb, y, N: int, K: int,
               eps: float = 1e-5, do_gelu: bool = False, blocks: int = 64):
    """Fused ``h = LN(h + res); y = W @ h + b`` (+ optional gelu) — the
    residual+LayerNorm is folded into the gemv prologue."""
    _step_kernels_mod().ss_gemv_ln(w, b, h, res if res is not None else h,
                                   lnw, lnb, y, int(N), int(K), float(eps),
                                   1 if do_gelu else 0, int(blocks))


def ss_self_attn_split(q, K, V, out, pm, pl, pacc, S: int, pos_dev,
                       split: int, heads: int, scale: float):
    """Split-K flash-decoding self-attention over rows [0, pos]."""
    _step_kernels_mod().ss_self_attn_split(q, K, V, pm, pl, pacc, out,
                                           pos_dev, int(S), int(split),
                                           int(heads), float(scale))


def ss_gemv_row(w, b, x, y, N: int, K: int):
    """Block-per-row GEMV for wide-K shapes, e.g. ffn2 [512,2048]."""
    _step_kernels_mod().ss_gemv_row(w, b, x, y, int(N), int(K))
