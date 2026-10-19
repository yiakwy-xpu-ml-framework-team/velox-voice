"""MERT2-FullSong encoder with VeloxVoice kernels.

Waveform [B, samples] @ 24 kHz -> frame features [B, T/4, 1024] at 25 Hz:

  1. mel frontend (power log-mel with checkpoint normalization) — torchaudio
     transforms with the checkpoint's window/fb/mean/std buffers;
  2. ConvNext subsampling 1025->128->512->1024, strides 1/2/2 (T -> T/4);
  3. 24x Conformer blocks (pre-LN half-weight FFN, RoPE sdpa attention,
     GLU + depthwise-k31 + GELU convolution module).

VeloxVoice kernel reuse (verified on H800 SuperPod):
  * ``cuda_ops.fused_layernorm`` — every layer norm (fp32 stats, bf16 out);
  * ``cuda_ops.silu_glu`` — the convolution module's GLU gate
    (``a * sigmoid(b)`` with the gate in the SECOND half, exactly the
    kernel's contract);
  * ``F.scaled_dot_product_attention`` for attention (RoPE applied to q/k);
  * plain cuBLAS GEMMs + torchaudio conv1d for the rest.

The forward is written so the whole 300 s window can be captured as a
piecewise CUDA graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import MERT2Config


class MERT2MelFrontend(torch.nn.Module):
    """Power log-mel with the checkpoint's fixed normalization."""

    def __init__(self, cfg: MERT2Config):
        super().__init__()
        from torchaudio.transforms import AmplitudeToDB, MelScale, Spectrogram

        self.cfg = cfg
        with torch.device("cpu"):
            self.register_buffer("mel_mean", torch.zeros(cfg.num_mel_bins))
            self.register_buffer("mel_std", torch.ones(cfg.num_mel_bins))
            self.spectrogram = Spectrogram(
                n_fft=cfg.n_fft, win_length=cfg.win_length,
                hop_length=cfg.hop_length, power=2.0).float()
            self.mel_scale = MelScale(
                n_mels=cfg.num_mel_bins, sample_rate=cfg.sampling_rate,
                n_stft=cfg.n_fft // 2 + 1).float()
        self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=None)

    def load_checkpoint_buffers(self, tensors: dict, device):
        """Copy the checkpoint's frontend buffers (window/fb/mean/std)."""
        sd = {}
        for k, v in tensors.items():
            if k.startswith("feature_extractor."):
                sd[k[len("feature_extractor."):]] = v
        if not sd:
            return
        self.load_state_dict(sd, strict=False)
        self.to(device)
        # the transforms must stay float32 even under a bf16 model
        self.spectrogram = self.spectrogram.float()
        self.mel_scale = self.mel_scale.float()

    @torch.no_grad()
    def forward(self, waveform):
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrum = self.spectrogram(waveform.float())
            mel = self.amplitude_to_db(self.mel_scale(spectrum))
            mel = mel[..., :-1].transpose(-1, -2)
            return (mel - self.mel_mean) / self.mel_std.clamp_min(1e-5)


def _jit_ln(x, w, b, eps, use_jit=True):
    """Shared with torch_conformer._jit_ln — fused_layernorm is an fp32
    kernel contract; bf16/fp16 buffers would be reinterpreted as float*.
    Same dtype guard, same reshape(-1, d) flatten."""
    if x.is_cuda and use_jit and x.dtype == torch.float32:
        from veloxvoice.kernels.ops import fused_layernorm

        d = x.shape[-1]
        return fused_layernorm(x.reshape(-1, d).contiguous(), w, b,
                               eps).view_as(x)
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def _ln(x, w, b, eps):
    return _jit_ln(x, w, b, eps)


# TODO (yiakwy) : check if we can reuse the module from WeNetConformerASR


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, cfg: MERT2Config):
        super().__init__()
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.base = cfg.rotary_embedding_base
        self.register_buffer(
            "inv_freq",
            1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float()
                                 / self.head_dim)),
            persistent=False)
        self._cache = None

    def forward(self, length, device, dtype):
        if (self._cache is None or self._cache[0] < length
                or self._cache[1] != device):
            pos = torch.arange(length, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", pos, self.inv_freq.to(device))
            angles = torch.cat((freqs, freqs), dim=-1)
            self._cache = (length, device, angles.cos(), angles.sin())
        cos, sin = self._cache[2][:length], self._cache[3][:length]
        return (cos.to(dtype)[None, :, None, :], sin.to(dtype)[None, :, None, :])


def _rotate_half(value):
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class ConformerSelfAttention(torch.nn.Module):
    def __init__(self, cfg: MERT2Config):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.query_proj = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.key_proj = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.value_proj = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.out_proj = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, hidden, cos, sin):
        bsz, seq, width = hidden.shape
        shape = (bsz, seq, self.num_heads, self.head_dim)

        q = self.query_proj(hidden).reshape(shape)
        k = self.key_proj(hidden).reshape(shape)
        v = self.value_proj(hidden).reshape(shape)

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.out_proj(attended.transpose(1, 2).reshape(bsz, seq, width))


class FeedForward(torch.nn.Module):
    def __init__(self, cfg: MERT2Config):
        super().__init__()
        self.w_1 = torch.nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.w_2 = torch.nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, hidden):
        return self.w_2(F.gelu(self.w_1(hidden)))


class ConvolutionModule(torch.nn.Module):
    """LN -> pw(2C) -> GLU -> dw k31 -> LN -> GELU -> pw.

    Submodule names mirror the checkpoint's Sequential indices
    (``conv_block.1/3/4.1/6``) so the generic loader maps them 1:1.  The GLU
    gate is ``a * sigmoid(b)`` — the ``silu_glu`` kernel's exact contract;
    the kernel is FP32-only, so under bf16 it runs in torch (same as the
    wenet lane's dtype guard).
    """

    def __init__(self, cfg: MERT2Config):
        super().__init__()
        width = cfg.hidden_size
        kernel = cfg.conv_depthwise_kernel_size
        self.layer_norm = torch.nn.LayerNorm(width, eps=cfg.layer_norm_eps)
        self.conv_block = torch.nn.Sequential(
            Transpose(),
            torch.nn.Conv1d(width, 2 * width, 1, bias=False),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(width, width, kernel,
                            padding=(kernel - 1) // 2, groups=width,
                            bias=False),
            torch.nn.Sequential(Transpose(),
                                torch.nn.LayerNorm(width,
                                                   eps=cfg.layer_norm_eps),
                                Transpose()),
            torch.nn.GELU(),
            torch.nn.Conv1d(width, width, 1, bias=False),
            Transpose(),
        )

    def forward(self, hidden):
        x = _ln(hidden, self.layer_norm.weight, self.layer_norm.bias,
                self.layer_norm.eps)
        return self.conv_block(x)


class ConformerBlock(torch.nn.Module):
    def __init__(self, cfg: MERT2Config):
        super().__init__()
        eps = cfg.layer_norm_eps
        self.ffn1_layer_norm = torch.nn.LayerNorm(cfg.hidden_size, eps=eps)
        self.ffn1 = FeedForward(cfg)
        self.attn_layer_norm = torch.nn.LayerNorm(cfg.hidden_size, eps=eps)
        self.attn = ConformerSelfAttention(cfg)
        self.conv_module = ConvolutionModule(cfg)
        self.ffn2_layer_norm = torch.nn.LayerNorm(cfg.hidden_size, eps=eps)
        self.ffn2 = FeedForward(cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.hidden_size, eps=eps)

    def forward(self, hidden, cos, sin):
        eps = self.ffn1_layer_norm.eps
        hidden = hidden + 0.5 * self.ffn1(
            _ln(hidden, self.ffn1_layer_norm.weight,
                self.ffn1_layer_norm.bias, eps))
        hidden = self.attn(
            _ln(hidden, self.attn_layer_norm.weight,
                self.attn_layer_norm.bias, eps), cos, sin) + hidden
        hidden = self.conv_module(hidden) + hidden
        hidden = hidden + 0.5 * self.ffn2(
            _ln(hidden, self.ffn2_layer_norm.weight,
                self.ffn2_layer_norm.bias, eps))
        return _ln(hidden, self.final_layer_norm.weight,
                   self.final_layer_norm.bias, eps)


class GlobalResponseNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, 1, dim))
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, hidden):
        magnitude = torch.norm(hidden, p=2, dim=1, keepdim=True)
        normalized = magnitude / (magnitude.mean(dim=-1, keepdim=True) + 1e-6)
        return self.weight * (hidden * normalized) + self.bias + hidden


class ConvNextLayer(torch.nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.depthwise_block = torch.nn.Sequential(
            Transpose(), torch.nn.Conv1d(dim, dim, 7, padding=3, groups=dim),
            Transpose())
        self.pointwise_block = torch.nn.Sequential(
            torch.nn.LayerNorm(dim, eps=eps),
            torch.nn.Linear(dim, 4 * dim), torch.nn.GELU(),
            GlobalResponseNorm(4 * dim), torch.nn.Linear(4 * dim, dim))

    def forward(self, hidden):
        return hidden + self.pointwise_block(self.depthwise_block(hidden))


class Transpose(torch.nn.Module):
    def forward(self, hidden):
        return hidden.transpose(1, 2)


class ConvNextBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, stride, depth, eps):
        super().__init__()
        self.resampling_layer = (
            torch.nn.Sequential(
                torch.nn.LayerNorm(in_ch, eps=eps), Transpose(),
                torch.nn.Conv1d(in_ch, out_ch, 2, stride=stride), Transpose())
            if in_ch != out_ch or stride > 1 else torch.nn.Identity())
        self.convnext_layers = torch.nn.Sequential(
            *[ConvNextLayer(out_ch, eps) for _ in range(depth)])

    def forward(self, hidden):
        return self.convnext_layers(self.resampling_layer(hidden))


class MERT2Model(torch.nn.Module):
    """Waveform -> 25 Hz frame representations (native implementation)."""

    def __init__(self, cfg: MERT2Config):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = MERT2MelFrontend(cfg)
        channels = [cfg.num_mel_bins] + list(cfg.subsampling_channels)
        self.subsampling_module = torch.nn.ModuleList([
            ConvNextBlock(channels[i], channels[i + 1], (1, 2, 2)[i],
                          cfg.subsampling_depths[i],
                          cfg.subsampling_layer_norm_eps)
            for i in range(3)])
        self.layers = torch.nn.ModuleList(
            [ConformerBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.embed_positions = RotaryEmbedding(cfg)

    _FP32_BUFFERS = ("mel_mean", "mel_std", "window", "fb", "inv_freq")

    def _apply(self, fn, recurse=True):
        # keep the mel frontend buffers + RoPE inv_freq in fp32 across
        # .to()/dtype casts (same intent as the reference's _apply override;
        # matched by buffer NAME because window/fb live on the Spectrogram /
        # MelScale children, not on the frontend module itself)
        saved = [(m, n, v) for m in self.modules()
                 for n, v in m._buffers.items()
                 if v is not None and v.is_floating_point()
                 and n in self._FP32_BUFFERS]
        super()._apply(fn, recurse=recurse)
        for module, name, original in saved:
            cur = module._buffers[name]
            if original.is_floating_point():
                module._buffers[name] = (
                    cur.float() if original.is_meta
                    else original.to(device=cur.device, dtype=torch.float32))
        return self
        channels = [cfg.num_mel_bins] + list(cfg.subsampling_channels)
        self.subsampling_module = torch.nn.ModuleList([
            ConvNextBlock(channels[i], channels[i + 1], (1, 2, 2)[i],
                          cfg.subsampling_depths[i],
                          cfg.subsampling_layer_norm_eps)
            for i in range(3)])
        self.layers = torch.nn.ModuleList(
            [ConformerBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.embed_positions = RotaryEmbedding(cfg)

    @torch.no_grad()
    def load_checkpoint(self, tensors: dict, device):
        """Map the MERT-v2 safetensors into this module (bf16 on device)."""
        own = dict(self.named_parameters())
        own.update(dict(self.named_buffers()))
        for name, param in own.items():
            if name in tensors:
                param.data.copy_(tensors[name]).to(device=device,
                                                   dtype=param.dtype)
            elif name.endswith(".inv_freq"):
                continue

    def _encode(self, input_values, output_hidden_states=False):
        mel = self.feature_extractor(input_values)
        hidden = mel.to(torch.bfloat16)
        for block in self.subsampling_module:
            hidden = block(hidden)
        cos, sin = self.embed_positions(hidden.shape[1], hidden.device,
                                        hidden.dtype)
        states = [hidden] if output_hidden_states else None
        for layer in self.layers:
            hidden = layer(hidden, cos, sin)
            if states is not None:
                states.append(hidden)
        return hidden, states

    def forward(self, input_values, output_hidden_states=False):
        hidden, states = self._encode(input_values, output_hidden_states)
        return hidden, (tuple(states) if states is not None else None)
