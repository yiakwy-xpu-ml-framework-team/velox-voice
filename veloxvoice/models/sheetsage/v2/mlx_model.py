"""SheetSage2 / MERT-v2 inference on Apple MLX (Metal + optional ANE).

This is the Apple-silicon sibling of ``mert.py`` + ``sheetsage.py`` +
``graph_decoder.py``.  Weights are read straight from the two safetensors
checkpoints, the encoder LoRA adapters are merged in fp32, and the hot path is
executed with ``mlx.core`` on the unified-memory GPU.  The decoder keeps a
static KV cache so each AR step is a handful of GEMVs.

Optional Metal JIT kernels live in ``veloxvoice/kernels/csrc/metal`` and are
used when ``VELOXVOICE_METAL_JIT=1`` (default on Apple silicon).
"""

from __future__ import annotations

import os
from typing import Iterable

import mlx.core as mx

from .config import MERT2Config, SheetSage2Config
from ....kernels.ops import metal_ops

BF16 = mx.bfloat16
FP32 = mx.float32


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _linear(x, w, b=None):
    y = x @ w.T
    return y if b is None else y + b


def _ln(x, w, b, eps):
    return mx.fast.layer_norm(x, w, b, eps)


def _gelu(x):
    c = mx.array(0.7071067811865476, dtype=x.dtype)
    half = mx.array(0.5, dtype=x.dtype)
    one = mx.array(1.0, dtype=x.dtype)
    return half * x * (one + mx.erf(x * c))


def _softmax(x, axis=-1):
    return mx.softmax(x, axis=axis)


def _conv1d(x, w, stride=1, padding=0, groups=1):
    return mx.conv1d(x, w, stride=stride, padding=padding, groups=groups)


def _take_t(x, dtype):
    return x.astype(dtype)


# ---------------------------------------------------------------------------
# frontend + ConvNext subsampling
# ---------------------------------------------------------------------------
class MERT2MLX:
    def __init__(self, cfg: MERT2Config):
        self.cfg = cfg
        self.model_tag = "default"
        self._ane = None
        self._ane_failed = False
        self.window = None
        self.fb = None
        self.mel_mean = None
        self.mel_std = None
        self.sub = []
        self.layers = []
        self.inv_freq = None
        self._cos = None
        self._sin = None
        self.dtype = BF16

    # -- loading -----------------------------------------------------------
    @classmethod
    def from_tensors(cls, cfg: MERT2Config, t: dict, adapter: dict | None,
                     dtype=BF16, lora_scale: float = 1.0,
                     model_tag: str = "default") -> "MERT2MLX":
        self = cls(cfg)
        self.model_tag = model_tag
        self.dtype = dtype
        self.window = t["feature_extractor.spectrogram.window"].astype(FP32)
        self.fb = t["feature_extractor.mel_scale.fb"].astype(FP32)
        self.mel_mean = t["feature_extractor.mel_mean"].astype(FP32)
        self.mel_std = t["feature_extractor.mel_std"].astype(FP32)

        channels = [cfg.num_mel_bins] + list(cfg.subsampling_channels)
        for bi in range(3):
            pre = f"subsampling_module.{bi}"
            blk = {"res": None, "layers": []}
            if f"{pre}.resampling_layer.0.weight" in t:
                blk["res"] = {
                    "ln_w": t[f"{pre}.resampling_layer.0.weight"].astype(dtype),
                    "ln_b": t[f"{pre}.resampling_layer.0.bias"].astype(dtype),
                    "conv_w": t[f"{pre}.resampling_layer.2.weight"].astype(dtype)
                    .transpose(0, 2, 1),
                    "conv_b": t[f"{pre}.resampling_layer.2.bias"].astype(dtype),
                }
            for li in range(cfg.subsampling_depths[bi]):
                p = f"{pre}.convnext_layers.{li}"
                blk["layers"].append({
                    "dw_w": t[f"{p}.depthwise_block.1.weight"].astype(dtype)
                    .transpose(0, 2, 1),
                    "dw_b": t[f"{p}.depthwise_block.1.bias"].astype(dtype),
                    "ln_w": t[f"{p}.pointwise_block.0.weight"].astype(dtype),
                    "ln_b": t[f"{p}.pointwise_block.0.bias"].astype(dtype),
                    "l1_w": t[f"{p}.pointwise_block.1.weight"].astype(dtype),
                    "l1_b": t[f"{p}.pointwise_block.1.bias"].astype(dtype),
                    "grn_w": t[f"{p}.pointwise_block.3.weight"].astype(dtype)
                    .reshape(-1),
                    "grn_b": t[f"{p}.pointwise_block.3.bias"].astype(dtype)
                    .reshape(-1),
                    "l2_w": t[f"{p}.pointwise_block.4.weight"].astype(dtype),
                    "l2_b": t[f"{p}.pointwise_block.4.bias"].astype(dtype),
                })
            self.sub.append(blk)

        scale = lora_scale
        for li in range(cfg.num_hidden_layers):
            p = f"layers.{li}"
            layer = {}
            for name in ("ffn1", "ffn2"):
                layer[f"{name}_ln_w"] = t[f"{p}.{name}_layer_norm.weight"].astype(dtype)
                layer[f"{name}_ln_b"] = t[f"{p}.{name}_layer_norm.bias"].astype(dtype)
                layer[f"{name}_w1"] = t[f"{p}.{name}.w_1.weight"].astype(dtype)
                layer[f"{name}_b1"] = t[f"{p}.{name}.w_1.bias"].astype(dtype)
                layer[f"{name}_w2"] = t[f"{p}.{name}.w_2.weight"].astype(dtype)
                layer[f"{name}_b2"] = t[f"{p}.{name}.w_2.bias"].astype(dtype)
            layer["attn_ln_w"] = t[f"{p}.attn_layer_norm.weight"].astype(dtype)
            layer["attn_ln_b"] = t[f"{p}.attn_layer_norm.bias"].astype(dtype)
            for proj in ("query_proj", "key_proj", "value_proj", "out_proj"):
                w = t[f"{p}.attn.{proj}.weight"].astype(FP32)
                if adapter is not None:
                    a = adapter.get(f"adapter.layers.{li}.attn.{proj}.lora_A.weight")
                    b = adapter.get(f"adapter.layers.{li}.attn.{proj}.lora_B.weight")
                    if a is not None and b is not None:
                        w = w + (b.astype(FP32) @ a.astype(FP32)) * scale
                layer[f"attn_{proj}_w"] = w.astype(dtype)
                layer[f"attn_{proj}_b"] = t[f"{p}.attn.{proj}.bias"].astype(dtype)
            layer["conv_ln_w"] = t[f"{p}.conv_module.layer_norm.weight"].astype(dtype)
            layer["conv_ln_b"] = t[f"{p}.conv_module.layer_norm.bias"].astype(dtype)
            layer["conv_pw_w"] = t[f"{p}.conv_module.conv_block.1.weight"].astype(dtype)
            layer["conv_dw_w"] = t[f"{p}.conv_module.conv_block.3.weight"].astype(dtype)
            layer["conv_ln2_w"] = t[f"{p}.conv_module.conv_block.4.1.weight"].astype(dtype)
            layer["conv_ln2_b"] = t[f"{p}.conv_module.conv_block.4.1.bias"].astype(dtype)
            layer["conv_pw2_w"] = t[f"{p}.conv_module.conv_block.6.weight"].astype(dtype)
            layer["final_ln_w"] = t[f"{p}.final_layer_norm.weight"].astype(dtype)
            layer["final_ln_b"] = t[f"{p}.final_layer_norm.bias"].astype(dtype)
            self.layers.append(layer)

        head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.inv_freq = 1.0 / (
            cfg.rotary_embedding_base
            ** (mx.arange(0, head_dim, 2, dtype=FP32) / head_dim))
        return self

    # -- math --------------------------------------------------------------
    def _mel(self, wav):
        """Waveform [B, samples] fp32 -> [B, frames, 128] fp32.

        The mel frontend is the only batch-unsafe stage (``as_strided`` over
        one waveform), so batch>1 is handled by a small host loop; B=1 keeps
        the original single-shot path.
        """
        if wav.ndim == 1:
            wav = wav[None]
        if wav.shape[0] == 1:
            return self._mel_one(wav[0])[None]
        return mx.stack([self._mel_one(wav[i]) for i in range(wav.shape[0])])

    def _mel_one(self, wav):
        cfg = self.cfg
        x = wav.reshape(-1).astype(FP32)
        pad = cfg.n_fft // 2
        left = x[1:pad + 1][::-1]
        right = x[-pad - 1:-1][::-1]
        xp = mx.concatenate([left, x, right], axis=0)
        n_frames = 1 + (x.shape[0] + 2 * pad - cfg.n_fft) // cfg.hop_length
        frames = mx.as_strided(xp, shape=(n_frames, cfg.n_fft),
                               strides=(cfg.hop_length, 1))
        spec = mx.fft.rfft(frames * self.window, n=cfg.n_fft, axis=-1)
        power = spec.real * spec.real + spec.imag * spec.imag
        mel = power @ self.fb
        if metal_ops.jit_enabled():
            mel = metal_ops.mel_log_norm(mel, self.mel_mean, self.mel_std)
        else:
            mel = 10.0 * mx.log10(mx.maximum(mel, mx.array(1e-10, dtype=FP32)))
            mel = mel[:-1]
            mel = (mel - self.mel_mean) / mx.maximum(
                self.mel_std, mx.array(1e-5, dtype=FP32))
        return mel

    def _convnext(self, x):
        for blk in self.sub:
            if blk["res"] is not None:
                r = blk["res"]
                x = _ln(x, r["ln_w"], r["ln_b"], 1e-6)
                x = _conv1d(x, r["conv_w"], stride=2)
                x = x + r["conv_b"]
            for l in blk["layers"]:
                y = _conv1d(x, l["dw_w"], padding=3, groups=x.shape[-1])
                y = y + l["dw_b"]
                y = _ln(y, l["ln_w"], l["ln_b"], 1e-6)
                y = _linear(y, l["l1_w"], l["l1_b"])
                y = _gelu(y)
                mag = mx.sqrt(mx.sum(y * y, axis=1, keepdims=True))
                norm = mag / (mx.mean(mag, axis=-1, keepdims=True) + 1e-6)
                y = l["grn_w"] * (y * norm) + l["grn_b"] + y
                y = _linear(y, l["l2_w"], l["l2_b"])
                x = x + y
        return x

    def _rope(self, t, dtype):
        if self._cos is None or self._cos.shape[1] < t:
            pos = mx.arange(t, dtype=FP32)
            freqs = mx.outer(pos, self.inv_freq)
            ang = mx.concatenate([freqs, freqs], axis=-1)
            self._cos = mx.cos(ang)[None, :, None, :]
            self._sin = mx.sin(ang)[None, :, None, :]
        return (self._cos[:, :t].astype(dtype), self._sin[:, :t].astype(dtype))

    @staticmethod
    def _rotate_half(v):
        a, b = mx.split(v, 2, axis=-1)
        return mx.concatenate([-b, a], axis=-1)

    def _attn(self, x, l, cos, sin):
        b, t, _ = x.shape
        h = self.cfg.num_attention_heads
        d = self.cfg.hidden_size // h
        q = _linear(x, l["attn_query_proj_w"], l["attn_query_proj_b"]).reshape(b, t, h, d)
        k = _linear(x, l["attn_key_proj_w"], l["attn_key_proj_b"]).reshape(b, t, h, d)
        v = _linear(x, l["attn_value_proj_w"], l["attn_value_proj_b"]).reshape(b, t, h, d)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        chunk = int(os.environ.get("VELOX_ATTN_CHUNK", "0"))
        if chunk and t > chunk:
            outs = []
            kt = k.transpose(0, 1, 3, 2)
            for s0 in range(0, t, chunk):
                qc = q[:, :, s0:s0 + chunk]
                if os.environ.get("VELOX_ATTN_CHUNK_FP32"):
                    sc = (qc.astype(FP32) @ kt.astype(FP32)) * (d ** -0.5)
                    pr = mx.softmax(sc, axis=-1)
                    outs.append((pr @ v.astype(FP32)).astype(x.dtype))
                else:
                    sc = (qc @ kt) * (d ** -0.5)
                    pr = mx.softmax(sc.astype(FP32), axis=-1).astype(x.dtype)
                    outs.append(pr @ v)
            a = mx.concatenate(outs, axis=2)
        elif os.environ.get("VELOX_ATTN_FP32"):
            a = mx.fast.scaled_dot_product_attention(
                q.astype(FP32), k.astype(FP32), v.astype(FP32), scale=d ** -0.5)
            a = a.astype(x.dtype)
        else:
            a = mx.fast.scaled_dot_product_attention(q, k, v, scale=d ** -0.5)
        a = a.transpose(0, 2, 1, 3).reshape(b, t, h * d)
        return _linear(a, l["attn_out_proj_w"], l["attn_out_proj_b"])

    def _ane_split(self, li, name, x, l):
        if self._ane_failed:
            return None
        if self._ane is None:
            from ....runtime.ane.ops.ffn import (enabled, maybe_create,
                                                 offload_rows)
            if not enabled():
                self._ane_failed = True
                return None
            if x.reshape(-1, self.cfg.hidden_size).shape[0] <= offload_rows():
                return None
            self._ane = maybe_create(self.cfg.hidden_size, self.model_tag)
            if self._ane is None:
                self._ane_failed = True
                return None
        flat = x.reshape(-1, self.cfg.hidden_size)
        w1 = l[f"{name}_w1"]
        b1 = l[f"{name}_b1"]
        w2 = l[f"{name}_w2"]
        b2 = l[f"{name}_b2"]

        def gpu(z):
            return _linear(_gelu(_linear(z, w1, b1)), w2, b2)

        try:
            y = self._ane(f"{name}_{li:02d}", flat, w1, b1, w2, b2, gpu)
            return y.reshape(x.shape)
        except Exception as exc:
            import warnings
            warnings.warn(f"ANE offload fell back to GPU: {exc}",
                          RuntimeWarning, stacklevel=2)
            self._ane_failed = True
            return None

    def _conformer(self, x, l, cos, sin, li=0):
        eps = self.cfg.layer_norm_eps
        x_ln = _ln(x, l["ffn1_ln_w"], l["ffn1_ln_b"], eps)
        y = self._ane_split(li, "ffn1", x_ln, l)
        if y is None:
            y = _linear(_gelu(_linear(x_ln, l["ffn1_w1"], l["ffn1_b1"])),
                        l["ffn1_w2"], l["ffn1_b2"])
        x = x + 0.5 * y
        x = x + self._attn(_ln(x, l["attn_ln_w"], l["attn_ln_b"], eps),
                           l, cos, sin)

        y = _ln(x, l["conv_ln_w"], l["conv_ln_b"], eps)
        y = _conv1d(y, l["conv_pw_w"].transpose(0, 2, 1))
        y = metal_ops.silu_glu(y) if metal_ops.jit_enabled() else (
            lambda a, b: a * mx.sigmoid(b))(*mx.split(y, 2, axis=-1))
        y = _conv1d(y, l["conv_dw_w"].transpose(0, 2, 1), padding=15,
                    groups=y.shape[-1])
        y = _ln(y, l["conv_ln2_w"], l["conv_ln2_b"], eps)
        y = _gelu(y)
        y = _conv1d(y, l["conv_pw2_w"].transpose(0, 2, 1))
        x = x + y

        x_ln = _ln(x, l["ffn2_ln_w"], l["ffn2_ln_b"], eps)
        y = self._ane_split(li, "ffn2", x_ln, l)
        if y is None:
            y = _linear(_gelu(_linear(x_ln, l["ffn2_w1"], l["ffn2_b1"])),
                        l["ffn2_w2"], l["ffn2_b2"])
        x = x + 0.5 * y
        return _ln(x, l["final_ln_w"], l["final_ln_b"], eps)

    def __call__(self, wav, return_states=False):
        mel = self._mel(wav)
        x = self._convnext(mel.astype(self.dtype))
        cos, sin = self._rope(x.shape[1], self.dtype)
        states = [x] if return_states else None
        for li, l in enumerate(self.layers):
            x = self._conformer(x, l, cos, sin, li)
            if states is not None:
                states.append(x)
        return x, states


# ---------------------------------------------------------------------------
# head + decoder
# ---------------------------------------------------------------------------
class Sheetsage2MLX:
    def __init__(self, cfg: SheetSage2Config):
        self.cfg = cfg
        self.encoder = MERT2MLX(cfg.mert)
        self.layer_weight = None
        self.enc_proj_w = None
        self.enc_proj_b = None
        self.tok_emb = None
        self.pos_emb = None
        self.ln_emb_w = None
        self.ln_emb_b = None
        self.layers = []
        self.lm_head = None
        self.dtype = BF16
        self.model_dir = cfg.model_dir

    @classmethod
    def load(cls, model_dir, base_model_dir, dtype=BF16) -> "Sheetsage2MLX":
        cfg = SheetSage2Config.from_json(os.path.join(model_dir, "config.json"))
        mert = mx.load(os.path.join(base_model_dir, "model.safetensors"))
        ss = mx.load(os.path.join(model_dir, "model.safetensors"))
        self = cls(cfg)
        self.dtype = dtype
        self.encoder = MERT2MLX.from_tensors(
            cfg.mert, mert, ss, dtype, cfg.lora_alpha / cfg.lora_rank,
            model_tag=os.path.basename(os.path.normpath(model_dir)))
        self.layer_weight = ss["layer_weight"].astype(FP32)
        self.enc_proj_w = ss["encoder_projection.weight"].astype(dtype)
        self.enc_proj_b = ss["encoder_projection.bias"].astype(dtype)
        self.tok_emb = ss["token_embedding.weight"].astype(dtype)
        self.pos_emb = ss["decoder.embed_positions.weight"].astype(dtype)
        self.ln_emb_w = ss["decoder.layernorm_embedding.weight"].astype(dtype)
        self.ln_emb_b = ss["decoder.layernorm_embedding.bias"].astype(dtype)
        for li in range(cfg.decoder_layers):
            p = f"decoder.layers.{li}"
            l = {}
            for name in ("self_attn", "encoder_attn"):
                for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    l[f"{name}_{proj}_w"] = ss[f"{p}.{name}.{proj}.weight"].astype(dtype)
                    l[f"{name}_{proj}_b"] = ss[f"{p}.{name}.{proj}.bias"].astype(dtype)
                l[f"{name}_ln_w"] = ss[f"{p}.{name}_layer_norm.weight"].astype(dtype)
                l[f"{name}_ln_b"] = ss[f"{p}.{name}_layer_norm.bias"].astype(dtype)
            l["fc1_w"] = ss[f"{p}.fc1.weight"].astype(dtype)
            l["fc1_b"] = ss[f"{p}.fc1.bias"].astype(dtype)
            l["fc2_w"] = ss[f"{p}.fc2.weight"].astype(dtype)
            l["fc2_b"] = ss[f"{p}.fc2.bias"].astype(dtype)
            l["final_ln_w"] = ss[f"{p}.final_layer_norm.weight"].astype(dtype)
            l["final_ln_b"] = ss[f"{p}.final_layer_norm.bias"].astype(dtype)
            self.layers.append(l)
        self.lm_head = self.tok_emb
        # drop the fp32 copies early
        del mert, ss
        return self

    # -- encoder -----------------------------------------------------------
    def encode(self, wav):
        x, states = self.encoder(wav, return_states=True)
        w = mx.softmax(self.layer_weight, axis=0).astype(self.dtype)
        mixed = states[0] * w[0]
        for wi, s in zip(w[1:], states[1:]):
            mixed = mixed + s * wi
        return _linear(mixed, self.enc_proj_w, self.enc_proj_b)


class MLXDecoder:
    """Static-KV Bart decoder for greedy grammar-constrained AR."""

    def __init__(self, model: Sheetsage2MLX, max_seq: int, t_mem: int,
                 dtype=BF16):
        self.m = model
        self.cfg = model.cfg
        self.L = model.cfg.decoder_layers
        self.H = model.cfg.num_attention_heads
        self.D = model.cfg.hidden_size // self.H
        self.S = int(max_seq)
        self.T = int(t_mem)
        self.dtype = dtype
        self.k_self = mx.zeros((self.L, 1, self.H, self.S, self.D), dtype=dtype)
        self.v_self = mx.zeros_like(self.k_self)
        self.k_cross = mx.zeros((self.L, 1, self.H, self.T, self.D), dtype=dtype)
        self.v_cross = mx.zeros_like(self.k_cross)
        self._prefill_len = 0

    # -- cross KV ----------------------------------------------------------
    def set_memory(self, memory):
        if memory.ndim == 2:
            memory = memory[None]
        if memory.shape[0] != 1:
            raise ValueError("MLX decoder currently supports batch size 1")
        for li, l in enumerate(self.m.layers):
            k = _linear(memory, l["encoder_attn_k_proj_w"],
                        l["encoder_attn_k_proj_b"])
            v = _linear(memory, l["encoder_attn_v_proj_w"],
                        l["encoder_attn_v_proj_b"])
            k = k.reshape(1, self.T, self.H, self.D).transpose(0, 2, 1, 3)
            v = v.reshape(1, self.T, self.H, self.D).transpose(0, 2, 1, 3)
            self.k_cross[li] = k
            self.v_cross[li] = v
        mx.eval(self.k_cross, self.v_cross)

    # -- prefill -----------------------------------------------------------
    def prefill(self, tokens: Iterable[int]):
        toks = mx.array(list(tokens), dtype=mx.int32)[None]
        p = toks.shape[1]
        h = self.m.tok_emb[toks]
        h = h + self.m.pos_emb[mx.arange(2, 2 + p)][None]
        h = _ln(h, self.m.ln_emb_w, self.m.ln_emb_b, 1e-5)
        scale = self.D ** -0.5
        for li, l in enumerate(self.m.layers):
            q = _linear(h, l["self_attn_q_proj_w"], l["self_attn_q_proj_b"])
            k = _linear(h, l["self_attn_k_proj_w"], l["self_attn_k_proj_b"])
            v = _linear(h, l["self_attn_v_proj_w"], l["self_attn_v_proj_b"])
            q = q.reshape(1, p, self.H, self.D).transpose(0, 2, 1, 3)
            k = k.reshape(1, p, self.H, self.D).transpose(0, 2, 1, 3)
            v = v.reshape(1, p, self.H, self.D).transpose(0, 2, 1, 3)
            self.k_self[li, :, :, :p] = k
            self.v_self[li, :, :, :p] = v
            a = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask="causal")
            a = a.transpose(0, 2, 1, 3).reshape(1, p, -1)
            h = _linear(a, l["self_attn_out_proj_w"], l["self_attn_out_proj_b"]) + h
            h = _ln(h, l["self_attn_ln_w"], l["self_attn_ln_b"], 1e-5)

            q = _linear(h, l["encoder_attn_q_proj_w"], l["encoder_attn_q_proj_b"])
            q = q.reshape(1, p, self.H, self.D).transpose(0, 2, 1, 3)
            a = mx.fast.scaled_dot_product_attention(
                q, self.k_cross[li], self.v_cross[li], scale=scale)
            a = a.transpose(0, 2, 1, 3).reshape(1, p, -1)
            h = _linear(a, l["encoder_attn_out_proj_w"],
                        l["encoder_attn_out_proj_b"]) + h
            h = _ln(h, l["encoder_attn_ln_w"], l["encoder_attn_ln_b"], 1e-5)

            h = _linear(_gelu(_linear(h, l["fc1_w"], l["fc1_b"])),
                        l["fc2_w"], l["fc2_b"]) + h
            h = _ln(h, l["final_ln_w"], l["final_ln_b"], 1e-5)
        self._prefill_len = p
        return self._logits(h[:, -1])

    def _logits(self, h):
        return h @ self.m.lm_head.T

    # -- one token ---------------------------------------------------------
    def step(self, token: int, pos: int, allowed):
        toks = mx.array([[int(token)]], dtype=mx.int32)
        h = self.m.tok_emb[toks]
        h = h + self.m.pos_emb[mx.array([pos + 2])][None]
        h = _ln(h, self.m.ln_emb_w, self.m.ln_emb_b, 1e-5)
        scale = self.D ** -0.5
        for li, l in enumerate(self.m.layers):
            q = _linear(h, l["self_attn_q_proj_w"], l["self_attn_q_proj_b"])
            k = _linear(h, l["self_attn_k_proj_w"], l["self_attn_k_proj_b"])
            v = _linear(h, l["self_attn_v_proj_w"], l["self_attn_v_proj_b"])
            q = q.reshape(1, 1, self.H, self.D).transpose(0, 2, 1, 3)
            k = k.reshape(1, 1, self.H, self.D).transpose(0, 2, 1, 3)
            v = v.reshape(1, 1, self.H, self.D).transpose(0, 2, 1, 3)
            self.k_self[li, :, :, pos:pos + 1] = k
            self.v_self[li, :, :, pos:pos + 1] = v
            kk = self.k_self[li, :, :, :pos + 1]
            vv = self.v_self[li, :, :, :pos + 1]
            a = mx.fast.scaled_dot_product_attention(q, kk, vv, scale=scale)
            a = a.transpose(0, 2, 1, 3).reshape(1, 1, -1)
            h = _linear(a, l["self_attn_out_proj_w"], l["self_attn_out_proj_b"]) + h
            h = _ln(h, l["self_attn_ln_w"], l["self_attn_ln_b"], 1e-5)

            q = _linear(h, l["encoder_attn_q_proj_w"], l["encoder_attn_q_proj_b"])
            q = q.reshape(1, 1, self.H, self.D).transpose(0, 2, 1, 3)
            a = mx.fast.scaled_dot_product_attention(
                q, self.k_cross[li], self.v_cross[li], scale=scale)
            a = a.transpose(0, 2, 1, 3).reshape(1, 1, -1)
            h = _linear(a, l["encoder_attn_out_proj_w"],
                        l["encoder_attn_out_proj_b"]) + h
            h = _ln(h, l["encoder_attn_ln_w"], l["encoder_attn_ln_b"], 1e-5)

            h = _linear(_gelu(_linear(h, l["fc1_w"], l["fc1_b"])),
                        l["fc2_w"], l["fc2_b"]) + h
            h = _ln(h, l["final_ln_w"], l["final_ln_b"], 1e-5)

        logits = self._logits(h)[0]
        masked = mx.where(allowed, logits, mx.array(float("-inf"), dtype=logits.dtype))
        return int(mx.argmax(masked))
