"""SheetSage2 model head : layer-weighted encoder mix, the
512-d projection, and a 6-layer post-LN Bart decoder with tied embeddings.

Checkpoint = MERT-v2 encoder tensors + a 219 MB SheetSage2 safetensors
containing the attention LoRA adapters, the Bart decoder,
``layer_weight``, ``encoder_projection`` and the tied token embedding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import SheetSage2Config


class BartDecoderLayer(torch.nn.Module):
    """Post-LN decoder layer: self-attn -> +res -> LN; cross-attn -> +res ->
    LN; FFN(gelu) -> +res -> LN."""

    def __init__(self, cfg: SheetSage2Config):
        super().__init__()
        d = cfg.hidden_size
        self.self_attn = _MultiheadAttention(d, cfg.num_attention_heads)
        self.self_attn_layer_norm = torch.nn.LayerNorm(d)
        self.encoder_attn = _MultiheadAttention(d, cfg.num_attention_heads)
        self.encoder_attn_layer_norm = torch.nn.LayerNorm(d)
        self.fc1 = torch.nn.Linear(d, cfg.intermediate_size)
        self.fc2 = torch.nn.Linear(cfg.intermediate_size, d)
        self.final_layer_norm = torch.nn.LayerNorm(d)

    def forward(self, hidden, self_kv=None, cross_kv=None, causal_mask=None):
        residual = hidden
        x, present = self.self_attn(hidden, kv=self_kv, attn_mask=causal_mask)
        hidden = self.self_attn_layer_norm(residual + x)

        residual = hidden
        x, cross_present = self.encoder_attn(hidden, kv=cross_kv)
        hidden = self.encoder_attn_layer_norm(residual + x)

        residual = hidden
        x = self.fc2(F.gelu(self.fc1(hidden)))
        hidden = self.final_layer_norm(residual + x)
        return hidden, present, cross_present


class _MultiheadAttention(torch.nn.Module):
    """q/k/v/out projections with an optional static KV cache row."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.num_heads = heads
        self.head_dim = dim // heads
        self.q_proj = torch.nn.Linear(dim, dim, bias=True)
        self.k_proj = torch.nn.Linear(dim, dim, bias=True)
        self.v_proj = torch.nn.Linear(dim, dim, bias=True)
        self.out_proj = torch.nn.Linear(dim, dim, bias=True)

    def _shape(self, x):
        bsz, seq, _ = x.shape
        return x.view(bsz, seq, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden, kv=None, key_value_states=None, attn_mask=None):
        """Self-attention when ``key_value_states`` is None, cross-attention
        otherwise.  ``kv`` = (K, V) cache tensors [B, H, T, D]."""
        q = self._shape(self.q_proj(hidden))
        if key_value_states is not None:
            k = self._shape(self.k_proj(key_value_states))
            v = self._shape(self.v_proj(key_value_states))
        elif kv is not None:
            k, v = kv
        else:
            k = self._shape(self.k_proj(hidden))
            v = self._shape(self.v_proj(hidden))
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        bsz, _, seq, _ = attended.shape
        attended = attended.transpose(1, 2).reshape(bsz, seq, -1)
        return self.out_proj(attended), (k, v)


class BartDecoder(torch.nn.Module):
    def __init__(self, cfg: SheetSage2Config, embed_tokens):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [BartDecoderLayer(cfg) for _ in range(cfg.decoder_layers)])
        self.layernorm_embedding = torch.nn.LayerNorm(cfg.hidden_size)
        self.embed_tokens = embed_tokens
        # learned positional embedding [S+2, D] with the Bart +2 offset
        self.embed_positions = torch.nn.Embedding(
            cfg.max_output_seq_len + 2, cfg.hidden_size)

    def forward(self, inputs_embeds, past_len, memory, kv_self=None,
                kv_cross=None, causal_mask=None):
        positions = torch.arange(past_len, past_len + inputs_embeds.shape[1],
                                 device=inputs_embeds.device)
        hidden = self.embed_tokens(inputs_embeds) \
            + self.embed_positions(positions + 2)[None]
        hidden = self.layernorm_embedding(hidden)

        new_self, new_cross = [], []
        for i, layer in enumerate(self.layers):
            skv = None if kv_self is None else (
                kv_self[0][i], kv_self[1][i])
            ckv = None if kv_cross is None else (
                kv_cross[0][i], kv_cross[1][i])
            hidden, s_pres, c_pres = layer(hidden, skv, ckv,
                                           causal_mask=causal_mask)
            new_self.append(s_pres)
            new_cross.append(c_pres)
        return hidden, (new_self, new_cross)


class Sheetsage2Model(torch.nn.Module):
    """Encoder (MERT2, LoRA-merged) + layer-weighted mix + Bart decoder."""

    def __init__(self, cfg: SheetSage2Config):
        super().__init__()
        from .mert import MERT2Model

        self.cfg = cfg
        self.encoder = MERT2Model(cfg.mert)
        self.layer_weight = torch.nn.Parameter(
            torch.zeros(cfg.mert.num_hidden_layers + 1))
        self.encoder_projection = torch.nn.Linear(
            cfg.mert.hidden_size, cfg.hidden_size)
        self.token_embedding = torch.nn.Embedding(
            cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.decoder = BartDecoder(cfg, self.token_embedding)
        self.output_projection = torch.nn.Linear(
            cfg.hidden_size, cfg.vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

    # aliases used by the graphed decode step (HF-style attribute names)
    @property
    def config(self):
        return self.cfg

    @property
    def max_output_seq_len(self):
        return self.cfg.max_output_seq_len

    # -- checkpoint --------------------------------------------------------
    @torch.no_grad()
    def load_checkpoint(self, sheetsage_path: str, mert_path: str, device):
        """Load + LoRA-merge (fp32, CPU) + cast bf16 onto ``device``."""
        from safetensors.torch import load_file

        mert_tensors = load_file(mert_path)
        ss_tensors = load_file(sheetsage_path)

        # 1) encoder weights
        mert_sd = {k: v for k, v in mert_tensors.items()
                   if not k.startswith("__metadata__")}
        self.encoder.load_checkpoint(mert_tensors, "cpu")

        # 2) LoRA adapters merged into the encoder attention projections
        scale = self.cfg.lora_alpha / self.cfg.lora_rank
        proj_map = {"query_proj": "query_proj", "key_proj": "key_proj",
                    "value_proj": "value_proj", "out_proj": "out_proj"}
        import re

        for name, param in self.encoder.named_parameters():
            m = re.match(r"layers\.(\d+)\.attn\.(\w+)\.weight", name)
            if not m:
                continue
            li, proj = m.groups()
            a = ss_tensors.get(
                f"adapter.layers.{li}.attn.{proj}.lora_A.weight")
            b = ss_tensors.get(
                f"adapter.layers.{li}.attn.{proj}.lora_B.weight")
            if a is not None and b is not None:
                param.data += (b.float() @ a.float()) * scale

        # 3) head weights
        self.layer_weight.data.copy_(ss_tensors["layer_weight"])
        self.encoder_projection.weight.data.copy_(
            ss_tensors["encoder_projection.weight"])
        self.encoder_projection.bias.data.copy_(
            ss_tensors["encoder_projection.bias"])
        self.token_embedding.weight.data.copy_(
            ss_tensors["token_embedding.weight"])
        dec_sd = {}
        for k, v in ss_tensors.items():
            if k.startswith("decoder."):
                dec_sd[k[len("decoder."):]] = v

        for name, param in self.decoder.named_parameters():
            if name == "embed_tokens.weight":
                continue
            if name in dec_sd:
                param.data.copy_(dec_sd[name])
        missing = [n for n, _ in self.decoder.named_parameters()
                   if n not in dec_sd and n != "embed_tokens.weight"]
        if missing:
            raise ValueError(f"missing decoder tensors: {missing[:4]}")

        self.to(device=device, dtype=torch.bfloat16)
        # the encoder frontend buffers stay fp32
        self.encoder.feature_extractor.spectrogram = (
            self.encoder.feature_extractor.spectrogram.float())
        self.encoder.feature_extractor.mel_scale = (
            self.encoder.feature_extractor.mel_scale.float())
        self.encoder.feature_extractor.mel_mean = (
            self.encoder.feature_extractor.mel_mean.float().to(device))
        self.encoder.feature_extractor.mel_std = (
            self.encoder.feature_extractor.mel_std.float().to(device))

    @torch.inference_mode()
    def get_audio_features(self, input_values, output_hidden_states=False):
        """Encoder forward + softmax(layer_weight)-weighted mix + projection."""
        _, states = self.encoder(input_values, True)  # [input, layer0..23]
        weights = torch.softmax(self.layer_weight.float(), dim=0).to(states[0].dtype)
        mixed = states[0] * weights[0]
        for weight, state in zip(weights[1:], states[1:]):
            mixed = mixed + state * weight
        return self.encoder_projection(mixed)

    def encode(self, audio):
        return self.get_audio_features(audio)

    def prefill_cross_kv(self, memory):
        """Compute the decoder's cross-attention K/V once per window."""
        kvs = []
        for layer in self.decoder.layers:
            k = layer.encoder_attn._shape(
                layer.encoder_attn.k_proj(memory))
            v = layer.encoder_attn._shape(
                layer.encoder_attn.v_proj(memory))
            kvs.append((k, v))
        k_self = torch.stack([kv[0] for kv in kvs])
        v_self = torch.stack([kv[1] for kv in kvs])
        return (k_self, v_self)
