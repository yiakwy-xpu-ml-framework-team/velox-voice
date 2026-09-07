"""Native torch.nn.Module implementation of WeNet's Conformer encoder + CTC head.

Default torch backend : identical math to the
MLX implementation (models/wenet/mlx_model.py) and to upstream WeNet numerics,
with hot ops dispatched to the csrc JIT kernels
(`veloxvoice.kernels.ops`: fused_layernorm / silu_glu / dw_causal_conv1d).

Module structures :

DenseLinear
WenetConformerASR
  - WenetConformerEncoder
    - WenetConformerEmbed
    - WenetConformerLayer
      - LayerNorm
      - WenetConformerFF : ff, ff_macaron
      - WenetConformerAttention
      - WenetConformerConvModule
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WeNetConfig, load_config

# enabled by setting VELOXVOICE_DISABLE_OFFLINE=1


class DenseLinear(nn.Linear):

    # TODO (yiakwy) : moving states to loading functions, replace weights with nn.Parameters
    def __init__(self, w, b=None):
        super().__init__(w.shape[1], w.shape[0], bias=b is not None)
        with torch.no_grad():
            self.weight.copy_(w)
            if b is not None:
                self.bias.copy_(b)
        self.precision = os.environ.get("VELOXVOICE_ASR_PRECISION", "fp32")

    def forward(self, x):
        if getattr(self, "precision", "fp32") == "nvfp4":
            orig_shape = x.shape
            if x.dim() == 3:
                x = x.reshape(-1, orig_shape[-1])
            from veloxvoice.kernels.ops import dgx_mxfp4_gemm
            from veloxvoice.models.wenet.mxfp4_linear import (
                CODE_16,
                _encode_codes,
                _pad16,
                pack_codes,
            )

            absmax = self.weight.detach().abs().max().clamp(1e-6)
            w_scale = absmax / CODE_16
            codes_w = pack_codes(
                _encode_codes(_pad16(self.weight.detach(), 128) / w_scale)
            )
            x_scale = x.detach().abs().max().clamp(1e-6) / CODE_16
            codes_x = pack_codes(_encode_codes(_pad16(x, 128) / x_scale))
            out = dgx_mxfp4_gemm(
                codes_x,
                codes_w,
                torch.full((codes_w.shape[1],), 127, device="cuda", dtype=torch.uint8),
                torch.full((codes_w.shape[1],), 127, device="cuda", dtype=torch.uint8),
            )
            out = out[: x.shape[0], : self.out_features] * x_scale * w_scale
            if self.bias is not None:
                out = out + self.bias
            if len(orig_shape) == 3:
                out = out.view(orig_shape[0], orig_shape[1], -1)
            return out
        return torch.nn.functional.linear(x, self.weight, self.bias)


class WenetConformerEmbed(nn.Module):
    """input_layer=conv2d subsampling4 (convmask: 80->39->19, C×F fold) + out_lin."""

    # TODO (yiakwy) : moving states to loading functions, replace weights with nn.Parameters
    def __init__(self, cfg, states):
        super().__init__()
        d = cfg.output_size

        self.conv1 = nn.Conv2d(1, d, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(d, d, kernel_size=3, stride=2)

        pe = states["encoder.embed.pos_enc.pe"]

        self.register_buffer("pos_pe", pe)

        self.d_model, self.xscale = d, math.sqrt(d)

        self.drain = DenseLinear(
            states["encoder.embed.out.0.weight"], states["encoder.embed.out.0.bias"]
        )
        with torch.no_grad():
            self.conv1.weight.copy_(states["encoder.embed.conv.0.weight"])
            self.conv1.bias.copy_(states["encoder.embed.conv.0.bias"])
            self.conv2.weight.copy_(states["encoder.embed.conv.2.weight"])
            self.conv2.bias.copy_(states["encoder.embed.conv.2.bias"])

    def forward(self, x, offset=0):
        x = x.unsqueeze(1)  # [1, 1, T, 80]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # [1, d, T', F']
        b, c, t, f = x.shape
        x = self.drain(x.permute(0, 2, 1, 3).reshape(b, t, c * f))  # [C×F]
        pos = self.pos_pe[:, offset : offset + t, :].to(x.dtype)
        x = x * self.xscale
        return x, pos


class WenetConformerAttention(nn.Module):

    # TODO (yiakwy) : moving states to loading functions, replace weights with nn.Parameters
    def __init__(self, cfg, states, i):
        super().__init__()
        p = f"encoder.encoders.{i}.self_attn."

        self.h, self.dk = cfg.attention_heads, cfg.output_size // cfg.attention_heads
        self.d = cfg.output_size

        self.pos_u = nn.Parameter(states[p + "pos_bias_u"].clone(), requires_grad=False)
        self.pos_v = nn.Parameter(states[p + "pos_bias_v"].clone(), requires_grad=False)

        self.lq = DenseLinear(
            states[p + "linear_q.weight"], states[p + "linear_q.bias"]
        )
        self.lk = DenseLinear(
            states[p + "linear_k.weight"], states[p + "linear_k.bias"]
        )
        self.lv = DenseLinear(
            states[p + "linear_v.weight"], states[p + "linear_v.bias"]
        )
        self.lo = DenseLinear(
            states[p + "linear_out.weight"], states[p + "linear_out.bias"]
        )
        self.lp = DenseLinear(states[p + "linear_pos.weight"])

    def forward_qkv(self, x):
        n = x.shape[0]
        q = self.lq(x).view(n, -1, self.h, self.dk)
        k = self.lk(x).view(n, -1, self.h, self.dk)
        v = self.lv(x).view(n, -1, self.h, self.dk)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(self, x5, mask, pos_emb, att_cache=None):
        n = x5.shape[0]
        q, k, v = self.forward_qkv(x5)
        n_pos = pos_emb.shape[0]
        p = self.lp(pos_emb).view(n_pos, -1, self.h, self.dk)
        p0 = p.transpose(1, 2)
        qu = (q.transpose(1, 2) + self.pos_u).transpose(1, 2)
        qv = (q.transpose(1, 2) + self.pos_v).transpose(1, 2)
        mac = qu @ k.transpose(-2, -1)
        mbd = qv @ p0.transpose(-2, -1)
        scores = (mac + mbd) / math.sqrt(self.dk)
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(n, -1, self.d)
        return self.lo(x), None


# NOTE (yiakwy) : velox JIT-kernel switch (WenetConformerASR.set_jit)
# - fused_layernorm + silu_glu (dw-conv),
# - both verified against torch refs by tools/kernel_harness.py (flash float jit kernel).
_ASR_USE_JIT = False


def asr_set_jit(on: bool):
    global _ASR_USE_JIT
    _ASR_USE_JIT = bool(on)


def _jit_ln(x, w, b, eps):
    if x.is_cuda and _ASR_USE_JIT:
        from veloxvoice.kernels.ops import fused_layernorm

        d = x.shape[-1]
        return fused_layernorm(x.reshape(-1, d).contiguous(), w, b, eps).view_as(x)
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b, eps)


class WenetConformerConvModule(nn.Module):

    # TODO (yiakwy) : moving states to loading functions, replace weights with nn.Parameters
    def __init__(self, cfg, states, i):
        super().__init__()
        p = f"encoder.encoders.{i}.conv_module."
        d = cfg.output_size
        self.causal = cfg.causal_conv
        pad = 0 if self.causal else cfg.cnn_module_kernel // 2
        self.cv1 = nn.Conv1d(d, 2 * d, kernel_size=1)
        self.dw = nn.Conv1d(
            d, d, kernel_size=cfg.cnn_module_kernel, padding=pad, groups=d
        )
        self.pw2 = nn.Conv1d(d, d, kernel_size=1)
        self.norm = nn.LayerNorm(d)
        self.norm.weight.data.copy_(states[p + "norm.weight"])
        self.norm.bias.data.copy_(states[p + "norm.bias"])

        with torch.no_grad():
            self.cv1.weight.copy_(states[p + "pointwise_conv1.weight"])
            self.cv1.bias.copy_(states[p + "pointwise_conv1.bias"])
            self.dw.weight.copy_(states[p + "depthwise_conv.weight"])
            self.dw.bias.copy_(states[p + "depthwise_conv.bias"])
            self.pw2.weight.copy_(states[p + "pointwise_conv2.weight"])
            self.pw2.bias.copy_(states[p + "pointwise_conv2.bias"])

    def forward(self, x, mask_pad, cnn_cache=None):
        # fp16 开时回退到 torch native (autocast 自动半位化); velox silu_glu 只有 fp32 路径
        if _ASR_USE_JIT and x.is_cuda and x.dtype == torch.float32:
            from veloxvoice.kernels.ops import silu_glu

            if getattr(self, "_pwlin", None) is None:
                self._pwlin = nn.Linear(self.cv1.in_channels, self.cv1.out_channels).to(
                    x.device
                )
                with torch.no_grad():
                    self._pwlin.weight.copy_(self.cv1.weight.squeeze(-1))
                    self._pwlin.bias.copy_(self.cv1.bias)
            g = self._pwlin(x)  # [B, T, C]

            # NOTE (yiakwy) : optimize
            sg = silu_glu(g.squeeze(0).contiguous()).unsqueeze(0)  # [B, T, C]

            x = sg.transpose(1, 2)  # [B, C, T]
        else:
            x = x.transpose(1, 2)
            x = self.cv1(x)
            x = F.glu(x, dim=1)
        x = self.dw(x)
        x = self.norm(x.transpose(1, 2))
        x = F.silu(x).transpose(1, 2)
        x = self.pw2(x)
        return x.transpose(1, 2), None


class WenetConformerFF(nn.Module):

    # TODO (yiakwy) : moving states to loading functions, replace weights with nn.Parameters
    def __init__(self, cfg, states, prefix):
        super().__init__()
        self.w1 = DenseLinear(
            states[prefix + ".w_1.weight"], states[prefix + ".w_1.bias"]
        )
        self.w2 = DenseLinear(
            states[prefix + ".w_2.weight"], states[prefix + ".w_2.bias"]
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class WenetConformerLayer(nn.Module):

    # TODO (yiakwy) : moving states to loading functions, replace weights with nn.Parameters
    def __init__(self, cfg, states, i):
        super().__init__()

        p = f"encoder.encoders.{i}."

        for name in (
            "norm_ff",
            "norm_mha",
            "norm_conv",
            "norm_ff_macaron",
            "norm_final",
        ):
            m = nn.LayerNorm(cfg.output_size)
            m.weight.data.copy_(states[p + name + ".weight"])
            m.bias.data.copy_(states[p + name + ".bias"])
            setattr(self, name, m)

        self.ff_scale = 0.5
        self.ff = WenetConformerFF(cfg, states, p + "feed_forward")
        self.ff_macaron = WenetConformerFF(cfg, states, p + "feed_forward_macaron")

        self.attn = WenetConformerAttention(cfg, states, i)

        self.conv = WenetConformerConvModule(cfg, states, i)

    def _N(self, m, x):
        return _jit_ln(x, m.weight, m.bias, m.eps) if _ASR_USE_JIT else m(x)

    def forward(self, x, mask, pos_emb, _):
        x = x + self.ff_scale * self.ff_macaron(self._N(self.norm_ff_macaron, x))
        x = (
            x
            + self.attn(
                self._N(self.norm_mha, x),
                torch.zeros(1, x.shape[1], x.shape[1], dtype=x.dtype, device=x.device),
                pos_emb,
            )[0]
        )
        x = x + self.conv(self._N(self.norm_conv, x), None)[0]
        x = x + self.ff_scale * self.ff(self._N(self.norm_ff, x))
        return self._N(self.norm_final, x)


class WenetConformerEncoder(nn.Module):

    # TODO (yiakwy) : moving states to loading functions, replace weights with nn.Parameters
    def __init__(self, cfg, states):
        super().__init__()
        self.embed = WenetConformerEmbed(cfg, states)
        self.layers = nn.ModuleList(
            [WenetConformerLayer(cfg, states, i) for i in range(cfg.num_blocks)]
        )
        self.after_norm = nn.LayerNorm(cfg.output_size)
        self.after_norm.weight.data.copy_(states["encoder.after_norm.weight"])
        self.after_norm.bias.data.copy_(states["encoder.after_norm.bias"])

    def forward(self, x):  # x [1, T, 80]
        x3, pos = self.embed(x)
        for layer in self.layers:
            x3 = layer(x3, None, pos, None)
        return self.after_norm(x3)


class WenetConformerASR:
    """TorchWenEtModel-style Conformer for ASR"""

    def __init__(
        self, model_dir: str, device: str = "cuda:0", required_cache_size: int = -1
    ):
        # NOTE (yiakwy) : torch, mlx backends
        self.backend = torch

        # NOTE (yiakwy) : targeted devices include cpu, nvgpu (amdgpu) and mlx (mps)
        self.device = device

        self.model_dir = model_dir

        self.cfg = load_config(model_dir)

        # NOTE (yiakwy) : main pt file of ASR model
        # TODO (yiakwy) : enable standard huggingface module format
        bundle = (
            os.path.join(model_dir, "offline")
            if os.path.exists(os.path.join(model_dir, "offline", "final.zip"))
            else model_dir
        )

        # TODO (yiakwy) : directly load to GPU
        state = (
            torch.load(
                os.path.join(self.model_dir, "final.pt"),
                map_location="cpu",
                weights_only=False,
            )
            if os.path.exists(os.path.join(self.model_dir, "final.pt"))
            else torch.load(
                os.path.join(bundle, "final.pt"), map_location="cpu", weights_only=False
            )
        )

        self.encoder = WenetConformerEncoder(self.cfg, state).to(device).eval()

        self.global_cmvn_mean = state["encoder.global_cmvn.mean"].to(device)
        self.global_cmvn_istd = state["encoder.global_cmvn.istd"].to(device)
        self.ctc_lo = (
            DenseLinear(state["ctc.ctc_lo.weight"], state["ctc.ctc_lo.bias"])
            .to(device)
            .eval()
        )

    def set_jit(self, on: bool):
        asr_set_jit(on)
        for L in self.encoder.layers:
            L.conv._pwlin = None

    def set_fp16(self, on: bool):
        """A2 ablation: autocast-cuda fp16 for the encoder (weights unchanged).
        autocast 处理: Linear/Conv/matmul → fp16.accum; LN/softmax/LayerNorm → fp32 同路."""
        self.fp16_mode = bool(on)

    @property
    def embeds_cmvn(self) -> bool:
        return True

    @property
    def pos_max_len(self) -> int:
        return int(self.encoder.embed.pos_pe.shape[1])

    def encode_utterance(self, feats):
        autocast = getattr(self.backend, "autocast", None)
        if autocast and getattr(self, "fp16_mode", False):
            with (
                self.backend.inference_mode(),
                autocast(device_type="cuda", dtype=torch.float16),
            ):
                x = (feats - self.global_cmvn_mean) * self.global_cmvn_istd
                return self.encoder(x)
        with self.backend.inference_mode():
            x = (feats - self.global_cmvn_mean) * self.global_cmvn_istd
            return self.encoder(x)

    def ctc_logp(self, enc_out):
        with self.backend.inference_mode():
            z = self.ctc_lo(enc_out)
            return z - torch.logsumexp(z, dim=-1, keepdim=True)
