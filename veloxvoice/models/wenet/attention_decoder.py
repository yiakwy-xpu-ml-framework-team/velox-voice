"""Native BiTransformerDecoder implementation of WeNet style Conformer encoder + attention rescoring."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_pad_mask(lengths: torch.Tensor, maxlen: int | None = None) -> torch.Tensor:
    """True at PAD positions (>= length)."""
    if maxlen is None:
        maxlen = int(lengths.max().item())
    seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=lengths.device)
    return seq_range.unsqueeze(0) >= lengths.unsqueeze(-1)


def subsequent_mask(size: int, device="cpu") -> torch.Tensor:
    """True at allowed (causal) positions."""
    ret = torch.ones(size, size, device=device, dtype=torch.bool).tril()
    return ret.unsqueeze(0)


class MultiheadAttention(nn.Module):
    """Rescore attention module, no SPDA for audio task"""

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.h_kv = num_heads
        self.dropout_rate = dropout_rate
        self.use_sdpa = False
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, _, _ = x.size()
        return x.view(B, -1, self.h, self.d_k).transpose(-3, -2)

    def forward_qkv(self, query, key, value):
        q = self._split(self.linear_q(query))
        k = self._split(self.linear_k(key))
        v = self._split(self.linear_v(value))
        return q, k, v

    def forward_attention(self, value, scores, mask):
        if mask.size(-1) > 0:
            mask4 = mask.unsqueeze(-3) == 0
            mask4 = mask4[..., : scores.size(-1)]
            scores = scores.masked_fill(mask4, float("-inf"))
            attn = torch.softmax(scores.float(), dim=-1).type_as(value)
            attn = attn.masked_fill(mask4, 0.0)
        else:
            attn = torch.softmax(scores.float(), dim=-1).type_as(value)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(-3, -2).contiguous()
        B = x.size(0)
        x = x.view(B, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, query, key, value, mask=None, pos_emb=None, cache=None):
        q, k, v = self.forward_qkv(query, key, value)
        # eval, cache=None -> k/v pass through
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), (k, v)


class PositionalEncoding(nn.Module):
    """WeNet PE with the checkpointed `pe` buffer (1, max_len, d_model)."""

    def __init__(self, d_model: int, dropout_rate: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.max_len = max_len
        self.xscale = math.sqrt(d_model)
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def position_encoding(self, offset: int, size: int, apply_dropout: bool):
        pos_emb = self.pe[:, int(offset) : int(offset) + size]
        return self.dropout(pos_emb) if apply_dropout else pos_emb

    def forward(self, x: torch.Tensor, offset: int = 0):
        pos_emb = self.position_encoding(offset, x.size(1), False)
        return self.dropout(x * self.xscale + pos_emb), self.dropout(pos_emb)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, units: int, dropout_rate: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, units)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(units, d_model)

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class DecoderLayer(nn.Module):
    """Pre-norm layer: self-attn -> src-attn -> ReLU FFN (residuals)."""

    def __init__(self, d_model, num_heads, units, dropout_rate,
                 self_attn_dropout=0.0, src_attn_dropout=0.0):
        super().__init__()
        self.normalize_before = True
        
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.self_attn = MultiheadAttention(d_model, num_heads, self_attn_dropout)

        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.src_attn = MultiheadAttention(d_model, num_heads, src_attn_dropout)

        self.norm3 = nn.LayerNorm(d_model, eps=1e-5)
        self.feed_forward = FeedForward(d_model, units, dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        residual = tgt
        x = self.norm1(tgt)
        x, _ = self.self_attn(x, x, x, tgt_mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x, _ = self.src_attn(x, memory, memory, memory_mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        return x, tgt_mask, memory, memory_mask


class TransformerDecoder(nn.Module):
    """BaseTransformerDecoder: embed -> N layers -> after_norm -> output_layer."""

    def __init__(self, vocab, d_model, num_heads, units, num_blocks,
                 dropout_rate=0.1, positional_dropout_rate=0.1,
                 self_attn_dropout=0.0, src_attn_dropout=0.0, max_len=5000):
        super().__init__()
        self.normalize_before = True
        self.use_output_layer = True
        self.gradient_checkpointing = False
        self.use_sdpa = False
        self.embed = nn.Sequential(
            nn.Embedding(vocab, d_model),
            PositionalEncoding(d_model, positional_dropout_rate, max_len),
        )
        self.decoders = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, units, dropout_rate,
                             self_attn_dropout, src_attn_dropout)
                for _ in range(num_blocks)
            ]
        )
        self.after_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.output_layer = nn.Linear(d_model, vocab)

    def forward(self, memory, memory_mask, ys_in_pad, ys_in_lens,
                offset: int = 0, max_len: int = 0):
        maxlen = int(ys_in_pad.size(1))
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen)
        tgt_mask = tgt_mask.to(ys_in_pad.device)
        tgt_mask = tgt_mask.unsqueeze(1) & subsequent_mask(maxlen, ys_in_pad.device)

        x, _ = self.embed(ys_in_pad)
        for layer in self.decoders:
            x, _, _, _ = layer(x, tgt_mask, memory, memory_mask)
        x = self.after_norm(x)
        x = self.output_layer(x)
        olens = tgt_mask.sum(1)
        return x, torch.tensor(0.0, device=x.device), olens


class BiTransformerDecoder(nn.Module):
    """Left (forward) + optional right (reverse) decoders. With
    reverse_weight <= 0 the right decoder is not executed (bundle graph)."""

    def __init__(self, vocab, d_model, num_heads, units, num_blocks,
                 r_num_blocks, dropout_rate=0.1, positional_dropout_rate=0.1,
                 self_attn_dropout=0.0, src_attn_dropout=0.0, max_len=5000):
        super().__init__()
        self.left_decoder = TransformerDecoder(
            vocab, d_model, num_heads, units, num_blocks, dropout_rate,
            positional_dropout_rate, self_attn_dropout, src_attn_dropout,
            max_len)
        self.right_decoder = TransformerDecoder(
            vocab, d_model, num_heads, units, r_num_blocks, dropout_rate,
            positional_dropout_rate, self_attn_dropout, src_attn_dropout,
            max_len)

    def forward(self, memory, memory_mask, ys_in_pad, ys_in_lens,
                r_ys_in_pad, reverse_weight: float = 0.0):
        l_x, _, olens = self.left_decoder(memory, memory_mask, ys_in_pad,
                                          ys_in_lens)
        if reverse_weight > 0.0:
            r_x, _, _ = self.right_decoder(memory, memory_mask, r_ys_in_pad,
                                           ys_in_lens)
        else:
            r_x = torch.tensor(0.0, device=memory.device)
        return l_x, r_x, olens


def build_bi_transformer_decoder(dec_state: dict, num_heads: int,
                                 max_len: int = 5000,
                                 dropout_rate: float = 0.1,
                                 positional_dropout_rate: float = 0.1,
                                 self_attn_dropout: float = 0.0,
                                 src_attn_dropout: float = 0.0):
    """Build the native decoder sized entirely from the checkpoint's own
    tensors and load the `decoder.*` weights (keys are prefix-stripped)."""
    vocab, d_model = dec_state["left_decoder.embed.0.weight"].shape
    num_blocks = len({k.split(".")[2] for k in dec_state
                      if k.startswith("left_decoder.decoders.")})
    r_num_blocks = len({k.split(".")[2] for k in dec_state
                        if k.startswith("right_decoder.decoders.")})
    units = dec_state["left_decoder.decoders.0.feed_forward.w_1.weight"].shape[0]
    dec = BiTransformerDecoder(
        vocab, d_model, num_heads, units, num_blocks, r_num_blocks,
        dropout_rate, positional_dropout_rate, self_attn_dropout,
        src_attn_dropout, max_len)
    dec.load_state_dict(dec_state, strict=True)
    return dec
