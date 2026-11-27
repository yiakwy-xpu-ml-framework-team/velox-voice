"""Batched MLX decoder for paged-window serving.

The CUDA lane batches windows in ``WindowDecodePool`` / ``_GraphedStep``.
This is the MLX analogue: one static-KV decoder over ``B`` independent
windows, so the 6-layer body and the LM head amortize weight reads across
windows.  It is Apple-only and imported lazily by the MLX path.
"""

from __future__ import annotations

import os

import mlx.core as mx

from ....kernels.ops import metal_ops
from .mlx_model import BF16, _gelu, _linear, _ln


class MLXBatchedDecoder:
    """Static-KV Bart decoder for ``B`` paged windows."""

    def __init__(self, model, max_seq: int, t_mem: int, batch: int,
                 dtype=BF16):
        self.m = model
        self.cfg = model.cfg
        self.B = int(batch)
        self.L = model.cfg.decoder_layers
        self.H = model.cfg.num_attention_heads
        self.D = model.cfg.hidden_size // self.H
        self.S = int(max_seq)
        self.T = int(t_mem)
        self.dtype = dtype
        self.k_self = mx.zeros((self.L, self.B, self.H, self.S, self.D),
                               dtype=dtype)
        self.v_self = mx.zeros_like(self.k_self)
        self.k_cross = mx.zeros((self.L, self.B, self.H, self.T, self.D),
                                dtype=dtype)
        self.v_cross = mx.zeros_like(self.k_cross)
        self.pos = mx.zeros((self.B,), dtype=mx.int32)
        self._prefill_len = 0

    # -- cross KV ----------------------------------------------------------
    def set_memory(self, memory):
        if memory.ndim == 2:
            memory = memory[None]
        if memory.shape[0] == 1 and self.B > 1:
            memory = mx.broadcast_to(memory, (self.B,) + memory.shape[1:])
        if memory.shape[0] != self.B:
            raise ValueError(f"memory batch {memory.shape[0]} != {self.B}")
        for li, l in enumerate(self.m.layers):
            k = _linear(memory, l["encoder_attn_k_proj_w"],
                        l["encoder_attn_k_proj_b"])
            v = _linear(memory, l["encoder_attn_v_proj_w"],
                        l["encoder_attn_v_proj_b"])
            k = k.reshape(self.B, self.T, self.H, self.D).transpose(0, 2, 1, 3)
            v = v.reshape(self.B, self.T, self.H, self.D).transpose(0, 2, 1, 3)
            self.k_cross[li] = k
            self.v_cross[li] = v
        mx.eval(self.k_cross, self.v_cross)

    # -- prefill -----------------------------------------------------------
    def prefill(self, tokens):
        toks = mx.array(list(tokens), dtype=mx.int32)
        p = toks.shape[0]
        h = self.m.tok_emb[toks]
        h = h + self.m.pos_emb[mx.arange(2, 2 + p)]
        h = _ln(h, self.m.ln_emb_w, self.m.ln_emb_b, 1e-5)
        h = mx.repeat(h[None], self.B, axis=0)
        scale = self.D ** -0.5
        for li, l in enumerate(self.m.layers):
            q = _linear(h, l["self_attn_q_proj_w"], l["self_attn_q_proj_b"])
            k = _linear(h, l["self_attn_k_proj_w"], l["self_attn_k_proj_b"])
            v = _linear(h, l["self_attn_v_proj_w"], l["self_attn_v_proj_b"])
            q = q.reshape(self.B, p, self.H, self.D).transpose(0, 2, 1, 3)
            k = k.reshape(self.B, p, self.H, self.D).transpose(0, 2, 1, 3)
            v = v.reshape(self.B, p, self.H, self.D).transpose(0, 2, 1, 3)
            self.k_self[li, :, :, :p] = k
            self.v_self[li, :, :, :p] = v
            a = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask="causal")
            a = a.transpose(0, 2, 1, 3).reshape(self.B, p, -1)
            h = self._lin(a, l["self_attn_out_proj_w"],
                          l["self_attn_out_proj_b"]) + h
            h = _ln(h, l["self_attn_ln_w"], l["self_attn_ln_b"], 1e-5)

            q = self._lin(h, l["encoder_attn_q_proj_w"],
                          l["encoder_attn_q_proj_b"])
            q = q.reshape(self.B, p, self.H, self.D).transpose(0, 2, 1, 3)
            a = mx.fast.scaled_dot_product_attention(
                q, self.k_cross[li], self.v_cross[li], scale=scale)
            a = a.transpose(0, 2, 1, 3).reshape(self.B, p, -1)
            h = self._lin(a, l["encoder_attn_out_proj_w"],
                          l["encoder_attn_out_proj_b"]) + h
            h = _ln(h, l["encoder_attn_ln_w"], l["encoder_attn_ln_b"], 1e-5)

            h = self._lin(_gelu(self._lin(h, l["fc1_w"], l["fc1_b"])),
                          l["fc2_w"], l["fc2_b"]) + h
            h = _ln(h, l["final_ln_w"], l["final_ln_b"], 1e-5)
        self._prefill_len = p
        self.pos = mx.full((self.B,), p, dtype=mx.int32)
        return h[:, -1] @ self.m.lm_head.T

    def _lin(self, x, w, b):
        if (metal_ops.jit_enabled() and os.environ.get("VELOX_GEMV_BATCH") == "1"
                and x.ndim == 3 and x.shape[1] == 1
                and x.shape[0] <= 8 and w.shape[0] <= 2048
                and x.dtype == mx.bfloat16):
            return metal_ops.gemv_batch(x[:, 0, :], w)[:, None, :] + b
        return _linear(x, w, b)

    # -- one batched token -------------------------------------------------
    def step(self, tokens, allowed):
        toks = mx.array(list(tokens), dtype=mx.int32)[:, None]
        h = self.m.tok_emb[toks]
        h = h + self.m.pos_emb[self.pos + 2][:, None, :]
        h = _ln(h, self.m.ln_emb_w, self.m.ln_emb_b, 1e-5)
        scale = self.D ** -0.5
        maxp = int(mx.max(self.pos)) + 1
        cols = mx.arange(maxp)
        mask = mx.where(cols[None, None, None, :] <= self.pos[:, None, None, None],
                        0.0, -1e9).astype(h.dtype)
        for li, l in enumerate(self.m.layers):
            q = self._lin(h, l["self_attn_q_proj_w"], l["self_attn_q_proj_b"])
            k = self._lin(h, l["self_attn_k_proj_w"], l["self_attn_k_proj_b"])
            v = self._lin(h, l["self_attn_v_proj_w"], l["self_attn_v_proj_b"])
            q = q.reshape(self.B, 1, self.H, self.D).transpose(0, 2, 1, 3)
            k = k.reshape(self.B, 1, self.H, self.D).transpose(0, 2, 1, 3)
            v = v.reshape(self.B, 1, self.H, self.D).transpose(0, 2, 1, 3)
            for b in range(self.B):
                p = int(self.pos[b])
                self.k_self[li, b, :, p, :] = k[b, :, 0, :]
                self.v_self[li, b, :, p, :] = v[b, :, 0, :]
            kk = self.k_self[li, :, :, :maxp]
            vv = self.v_self[li, :, :, :maxp]
            a = mx.fast.scaled_dot_product_attention(
                q, kk, vv, scale=scale, mask=mask)
            a = a.transpose(0, 2, 1, 3).reshape(self.B, 1, -1)
            h = self._lin(a, l["self_attn_out_proj_w"],
                          l["self_attn_out_proj_b"]) + h
            h = _ln(h, l["self_attn_ln_w"], l["self_attn_ln_b"], 1e-5)

            q = self._lin(h, l["encoder_attn_q_proj_w"],
                          l["encoder_attn_q_proj_b"])
            q = q.reshape(self.B, 1, self.H, self.D).transpose(0, 2, 1, 3)
            a = mx.fast.scaled_dot_product_attention(
                q, self.k_cross[li], self.v_cross[li], scale=scale)
            a = a.transpose(0, 2, 1, 3).reshape(self.B, 1, -1)
            h = self._lin(a, l["encoder_attn_out_proj_w"],
                          l["encoder_attn_out_proj_b"]) + h
            h = _ln(h, l["encoder_attn_ln_w"], l["encoder_attn_ln_b"], 1e-5)

            h = self._lin(_gelu(self._lin(h, l["fc1_w"], l["fc1_b"])),
                          l["fc2_w"], l["fc2_b"]) + h
            h = _ln(h, l["final_ln_w"], l["final_ln_b"], 1e-5)

        self.pos = self.pos + 1
        logits = h[:, 0] @ self.m.lm_head.T
        masked = mx.where(allowed, logits,
                          mx.array(float("-inf"), dtype=logits.dtype))
        return mx.argmax(masked, axis=-1).tolist()
