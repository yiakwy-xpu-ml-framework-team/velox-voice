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

import glob
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WeNetConfig, load_config


def _checkpoint_path(model_dir: str) -> str | None:
    """Return the preferred native WeNet checkpoint, if one exists."""
    for pattern in ("avg_*.pt", "final.pt", "*.pt"):
        candidates = sorted(glob.glob(os.path.join(model_dir, pattern)))
        if candidates:
            return candidates[0]
    return None


def _normalize_legacy_state(raw: dict) -> dict:
    """Normalize WeNet/TorchScript keys for WenetConformerASR."""
    state: dict = {}
    for raw_key, value in raw.items():
        key = raw_key[len("model.") :] if raw_key.startswith("model.") else raw_key
        if (
            "num_batches_tracked" in key
            or key.startswith("decoder.")
            or ".pos_encoding.pe" in key
        ):
            continue
        if key.startswith("encoder.embed.out_lin."):
            key = key.replace("encoder.embed.out_lin.", "encoder.embed.out.0.", 1)
        elif key.startswith("encoder.embed.linear."):
            key = key.replace("encoder.embed.linear.", "encoder.embed.out.0.", 1)
        if key.endswith("conv_module.norm_scale"):
            key = key[: -len("norm_scale")] + "norm.weight"
        elif key.endswith("conv_module.norm_shift"):
            key = key[: -len("norm_shift")] + "norm.bias"
        state[key] = value
    return state


def _state_from_ts_archive(model_dir: str) -> dict:
    """Normalize a TorchScript bundle into WenetConformerASR checkpoint keys."""
    from .ts_archive import read_ts_archive_tensor_map

    return _normalize_legacy_state(
        read_ts_archive_tensor_map(os.path.join(model_dir, "final.zip"))
    )


def load_asr_state(model_dir: str) -> dict:
    """Load the checkpoint once for both vocabulary discovery and model init.

    Native ``*.pt`` checkpoints are preferred.  TorchScript ``final.zip``
    bundles are supported through the robust archive reader for hosts where
    ``torch.jit.load`` cannot deserialize quantized modules.
    """
    pt_path = _checkpoint_path(model_dir)
    if pt_path is not None:
        state = torch.load(pt_path, map_location="cpu", weights_only=False)
        if not all(
            key in state
            for key in (
                "encoder.embed.pos_enc.pe",
                "encoder.global_cmvn.mean",
                "encoder.global_cmvn.istd",
                "ctc.ctc_lo.weight",
            )
        ):
            state = _normalize_legacy_state(state)
        return state

    zip_path = os.path.join(model_dir, "final.zip")
    if os.path.exists(zip_path):
        return _state_from_ts_archive(model_dir)

    raise FileNotFoundError(
        f"no ASR checkpoint found in {model_dir!r}; expected avg_*.pt, final.pt, "
        "*.pt, or final.zip"
    )


def load_avg_pt(model_dir: str) -> dict:
    """Backward-compatible alias for the unified ASR checkpoint loader."""
    return load_asr_state(model_dir)


def torch_state_dict(states: dict) -> dict:
    import re

    out = {}
    blocks = 0
    for k0, v in states.items():
        k = k0[len("model.") :] if k0.startswith("model.") else k0
        if (
            "num_batches_tracked" in k
            or k.startswith("decoder.")
            or k.startswith("global_cmvn.")
            or ".pos_encoding.pe" in k
        ):
            continue

        m = re.fullmatch(r"encoder\.embed\.conv\.([024])\.(weight|bias)", k)
        if m:
            idx = {"0": "conv1", "2": "conv2", "4": "conv3"}[m.group(1)]
            out[f"encoder.embed.{idx}.{m.group(2)}"] = v
            continue
        m = re.fullmatch(r"encoder\.embed\.(?:linear|out\.0)\.(weight|bias)", k)
        if m:
            out[f"encoder.embed.out_lin.{m.group(1)}"] = v
            continue
        m = re.fullmatch(r"encoder\.after_norm\.(weight|bias)", k)
        if m:
            out[f"encoder.after_norm_{'w' if m.group(1) == 'weight' else 'b'}"] = v
            continue
        m = re.fullmatch(r"ctc\.ctc_lo\.(weight|bias)", k)
        if m:
            out[f"ctc_lo.{m.group(1)}"] = v
            continue

        m = re.fullmatch(r"encoder\.encoders\.(\d+)\.(.*)", k)
        if not m or ("conv_module.norm." in k):
            continue
        blocks = max(blocks, int(m.group(1)) + 1)
        rest = m.group(2)
        base = f"encoder.encoders.{m.group(1)}"
        for lname in (
            "norm_ff",
            "norm_ff_macaron",
            "norm_mha",
            "norm_conv",
            "norm_final",
        ):
            if rest == f"{lname}.weight":
                out[f"{base}.{lname}_w"] = v
                break
            if rest == f"{lname}.bias":
                out[f"{base}.{lname}_b"] = v
                break
        else:
            if rest == "conv_module.depthwise_conv.bias":
                out[f"{base}.{rest}"] = v
            elif rest == "conv_module.depthwise_conv.weight":
                out[f"{base}.{rest}"] = v
            elif rest in (
                "conv_module.pointwise_conv1.weight",
                "conv_module.pointwise_conv2.weight",
            ):
                out[f"{base}.{rest}"] = v.squeeze(-1)
            elif rest == "self_attn.linear_pos.weight":
                out[f"{base}.self_attn.linear_pos_w"] = v
            elif rest == "self_attn.linear_pos.bias":
                pass
            elif rest.startswith("concat_linear.") or rest.startswith("dropout."):
                continue
            else:
                out[f"{base}.{rest}"] = v


class DenseLinear(nn.Linear):

    # TODO (yiakwy) : moving states to loading functions, replace weights with nn.Parameters
    def __init__(self, w, b=None):
        super().__init__(w.shape[1], w.shape[0], bias=b is not None)
        with torch.no_grad():
            self.weight.copy_(w)
            if b is not None:
                self.bias.copy_(b)
        self.precision = os.environ.get("VELOXVOICE_ASR_PRECISION", "fp32")
        # Pre-quantize weight for mxfp4 (user mandate: weight pre-quantize)
        self._wq = None
        self._ws = None
        self._w_bf16 = None

    def _prequantize_weight(self, precision):
        """Pre-quantize weight once for nvfp4/mxfp4. Call after moving to device."""
        if (
            precision in ("nvfp4", "mxfp4")
            and self._wq is None
            and self.weight.device.type == "cuda"
        ):
            from veloxvoice.kernels.ops.triton_ops import per_row_col_quantize
            from veloxvoice.models.wenet.mxfp4_linear import _pad16

            # Pad N (weight rows) to K_BN=128 for GEMM alignment
            w_padded = _pad16(self.weight.detach().float(), 128)
            self._wq, self._ws = per_row_col_quantize(w_padded)
            self._w_bf16 = w_padded.to(torch.bfloat16)
            self._w_N_actual = self.weight.shape[0]

    def forward(self, x):
        precision = getattr(self, "precision", "fp32")
        if precision in ("nvfp4", "mxfp4"):
            orig_shape = x.shape
            if x.dim() == 3:
                x = x.reshape(-1, orig_shape[-1])

            # Lazy pre-quantize weight
            if self._wq is None:
                self._prequantize_weight(precision)

            if precision == "nvfp4":
                from veloxvoice.kernels.ops import nvfp4_linear

                out = nvfp4_linear(x, self._wq, self._ws, w_bf16=self._w_bf16)
                out = out[:, : self._w_N_actual]
            else:  # mxfp4 — same per-matrix scalar scale path, different naming
                from veloxvoice.kernels.ops import dgx_mxfp4_gemm
                from veloxvoice.kernels.ops.triton_ops import triton_quantize_w

                M_actual = x.shape[0]
                K_BM = 128
                M_padded = ((M_actual + K_BM - 1) // K_BM) * K_BM
                if M_padded != M_actual:
                    x_pad = torch.zeros(
                        M_padded, x.shape[1], device=x.device, dtype=x.dtype
                    )
                    x_pad[:M_actual] = x
                else:
                    x_pad = x
                xq, xs = triton_quantize_w(x_pad.float())
                out = dgx_mxfp4_gemm(xq, self._wq, xs, self._ws)
                out = out[:M_actual, : self._w_N_actual]

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

    def forward_chunk(self, x, cache):
        if x.dim() == 3:
            x = x[0]
        xc = torch.cat([cache, x], dim=0) if cache.numel() > 0 else x
        n = xc.shape[0]
        if n < 7:
            return xc.new_zeros((0, self.drain.out_features)), xc
        k = (n - 7) // 4 + 1
        proc = xc[: 4 * (k - 1) + 7]
        xx = proc.unsqueeze(0).unsqueeze(0)
        xx = F.relu(self.conv1(xx))
        xx = F.relu(self.conv2(xx))
        t, c, f = xx.shape[2], xx.shape[3], xx.shape[1]
        y = xx.permute(0, 2, 1, 3).reshape(t, c * f)
        y = self.drain(y) * self.xscale
        pos = self.pos_pe[:, offset if False else 0 : 0, :]
        # Relative attention consumes its own sinusoidal table; embed output
        # must not be scaled twice.
        return y, torch.empty(0, device=y.device)


class WenetConformerAttention(nn.Module):
    """flash attention with attention score[t, s] = (q[t] + pos_u) * k[s] + (q[t] + pos_v) * lp(pe[s])"""

    # TODO (yiakwy) : moving states to loading functions, replace weights with nn.Parameters
    def __init__(self, cfg, states, i):
        super().__init__()
        p = f"encoder.encoders.{i}.self_attn."

        self.h = cfg.attention_heads

        # hidden size
        self.d = cfg.output_size

        # headdim size
        self.head_dim = cfg.output_size // cfg.attention_heads

        self.scale = 1.0 / math.sqrt(self.head_dim)

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

        # NOTE (yiakwy) : enable nvfp4/mxfp4 in dgx spark
        self.fp4_quantized = getattr(self.lq, "precision", "fp32") in ("nvfp4", "mxfp4")

        with torch.no_grad():
            wqkv = torch.cat([self.lq.weight, self.lk.weight, self.lv.weight], dim=0)
            bqkv = (
                torch.cat([self.lq.bias, self.lk.bias, self.lv.bias], dim=0)
                if self.lq.bias is not None
                else None
            )

        self.register_buffer("_wqkv", wqkv, persistent=False)
        self.register_buffer("_bqkv", bqkv, persistent=False)

        self.qkv_unpack = True
        # NOTE (yiakwy) : disable in H800 since torch native linear is faster enough
        self.use_fused_qkv = False

    def forward_chunk(self, x_q, offset_t, k_cache, v_cache):
        """Streaming relative-position attention.

        x_q [Tq, d], offset_t is a device scalar, k_cache [H, DK, L], and
        v_cache [H, L, DK].  Keeps the bounded u2++ cache shape-static.
        """
        if x_q.dim() == 3:
            x_q = x_q[0]
        Tq, d = x_q.shape
        H, DK, L = k_cache.shape

        qkv = self._qkv_proj(x_q.unsqueeze(0))[0]
        q = qkv[:, :d].view(Tq, H, DK)
        k_new = qkv[:, d : 2 * d].view(Tq, H, DK).permute(1, 2, 0)
        v_new = qkv[:, 2 * d :].view(Tq, H, DK).permute(1, 0, 2)

        k = torch.cat([k_cache, k_new], dim=2)[:, :, -L:]
        v = torch.cat([v_cache, v_new], dim=1)[:, -L:, :]

        total = offset_t + Tq
        valid = torch.clamp(total, max=L)

        half = L - 1
        pos = torch.arange(2 * half + 1, device=x_q.device, dtype=x_q.dtype)
        div = torch.exp(
            torch.arange(0, d, 2, device=x_q.device, dtype=x_q.dtype)
            * -(math.log(10000.0) / d)
        )
        pe = torch.zeros(2 * half + 1, d, device=x_q.device, dtype=x_q.dtype)
        pe[:, 0::2] = torch.sin(pos[:, None] * div)
        pe[:, 1::2] = torch.cos(pos[:, None] * div)
        pe = self.lp(pe).to(x_q.dtype)

        k_first_abs = total - valid
        abs_i = offset_t + torch.arange(Tq, device=x_q.device)
        abs_j = k_first_abs + torch.clamp(
            torch.arange(L, device=x_q.device) - (L - valid), min=0
        )
        idx = (abs_i[:, None] - abs_j[None, :]) + half

        q_u = (q + self.pos_u).permute(1, 0, 2)
        q_v = q + self.pos_v
        m_ac = q_u @ k
        P = pe[idx].view(Tq, L, H, DK)
        m_bd = (q_v[:, None, :, :] * P).sum(-1).permute(2, 0, 1)

        scores = (m_ac + m_bd) / math.sqrt(DK)
        mask = torch.where(
            torch.arange(L, device=x_q.device) < (L - valid),
            torch.full((L,), -1.0e9, device=x_q.device, dtype=scores.dtype),
            torch.zeros(L, device=x_q.device, dtype=scores.dtype),
        )
        attn = torch.softmax(scores + mask[None, None, :], dim=-1)
        out = (attn @ v).permute(1, 0, 2).reshape(Tq, H * DK)
        return self.lo(out), k, v

    def _qkv_proj(self, x):
        n = x.shape[0]

        if self.fp4_quantized:
            return torch.cat(
                [
                    self.lq(x).view(n, -1, self.d),
                    self.lk(x).view(n, -1, self.d),
                    self.lv(x).view(n, -1, self.d),
                ],
                dim=-1,
            )

        if self.use_fused_qkv:
            from veloxvoice.kernels.ops import fused_qkv

            return fused_qkv(x.reshape(-1, x.shape[-1]), self._wqkv, self._bqkv).view(
                n, -1, 3 * self.d
            )
        else:
            return torch.nn.functional.linear(x, self._wqkv, self._bqkv)

    def forward(self, x, mask, pos_emb, att_cache=None):
        n = x.shape[0]
        n_pos = pos_emb.shape[0]

        # NOTE (yiakwy) : enable nvfp4/mxfp4 in dgx spark
        fp4_quantized = getattr(self.lq, "precision", "fp32") in ("nvfp4", "mxfp4")

        # [n, T, 3d], q|k|v per token
        qkv = self._qkv_proj(x)

        # [n_pos, T, d]
        p_raw = self.lp(pos_emb)

        if (
            self.qkv_unpack
            and n_pos == 1
            and n == 1
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and not fp4_quantized
        ):
            from veloxvoice.kernels.ops import qkv_pack

            q, k, v = qkv_pack(qkv[0], p_raw[0], self.pos_u, self.pos_v, self.scale)
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        else:
            # slice q, k, v from qkv

            # TODO (yiakwy) : enable fast fused path for B > 1
            q, k, v = (
                qkv[:, :, : self.d].view(n, -1, self.h, self.head_dim),
                qkv[:, :, self.d : 2 * self.d].view(n, -1, self.h, self.head_dim),
                qkv[:, :, 2 * self.d :].view(n, -1, self.h, self.head_dim),
            )

            # NOTE (yiakwy) : [n, T, h, head_dim] -> [n, h, T, head_dim] for scaled_dot_product_attention
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

            # NOTE (yiakwy) : [n_pos, T, d] -> [n_pos, h, T, head_dim] for scaled_dot_product_attention
            p_raw = p_raw.view(n_pos, -1, self.h, self.head_dim).transpose(1, 2)

            if n_pos == 1 and n > 1:
                p_raw = p_raw.expand(n, -1, -1, -1)

            u = self.pos_u.view(1, self.h, 1, self.head_dim)
            vb = self.pos_v.view(1, self.h, 1, self.head_dim)

            q = torch.cat([(q + u), (q + vb)], dim=-1) * self.scale

            k = torch.cat([k, p_raw], dim=-1)
            v = torch.cat([v, torch.zeros_like(v)], dim=-1)

        # TODO (yiakwy) : optimize attention on DGX Spark
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=1.0
        )
        x = x[..., : self.head_dim]
        x = x.transpose(1, 2).reshape(n, -1, self.d)
        return self.lo(x), None


# NOTE (yiakwy) : velox JIT-kernel switch (WenetConformerASR.set_jit)
# - fused_layernorm + silu_glu (dw-conv),
# - both verified against torch refs by tools/kernel_harness.py (flash float jit kernel).
_ASR_USE_JIT = False


def asr_set_jit(on: bool):
    global _ASR_USE_JIT
    _ASR_USE_JIT = bool(on)


def _jit_ln(x, w, b, eps):
    # fused_layernorm is an fp32 kernel: it would reinterpret a bf16/fp16
    # buffer as float* under autocast — guard on dtype like the conv path.
    if x.is_cuda and _ASR_USE_JIT and x.dtype == torch.float32:
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

    def forward_chunk(self, x, cache):
        if x.dim() == 3:
            x = x[0]
        from veloxvoice.kernels.ops import pwlin_glu

        wq, bq = (self.cv1.weight.squeeze(-1).contiguous(), self.cv1.bias.contiguous())
        sg = pwlin_glu(x, wq, bq)
        if sg is None:
            g = F.linear(x, self.cv1.weight.squeeze(-1), self.cv1.bias)
            a, b = g.chunk(2, dim=-1)
            sg = a * torch.sigmoid(b)
        pad = torch.cat([cache, sg], dim=0)
        y = (
            F.conv1d(
                pad.T.unsqueeze(0), self.dw.weight, self.dw.bias, groups=self.dw.groups
            )
            .squeeze(0)
            .T
        )
        new_cache = pad[-(self.dw.kernel_size[0] - 1) :]
        y = self.norm(y)
        y = F.silu(y)
        return F.linear(y, self.pw2.weight.squeeze(-1), self.pw2.bias), new_cache

    def forward(self, x, mask_pad, cnn_cache=None):
        # TODO (yiakwy) : add fp16 support for velox silu_glu
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
            if g.dtype != torch.float32:
                # autocast may return bf16/fp16 even when x is fp32; silu_glu
                # is an fp32 kernel and would reinterpret the buffer.
                g = g.float()

            # NOTE (yiakwy) : optimize
            sg = silu_glu(g.squeeze(0).contiguous()).unsqueeze(0)  # [B, T, C]

            x = sg.transpose(1, 2)  # [B, C, T]
        elif _ASR_USE_JIT and x.is_cuda and x.dtype == torch.bfloat16:
            # Fused pointwise1 (1x1 conv) + silu-GLU via the sm_90a wgmma kernel.
            # Replaces Conv1d(1x1) + F.glu (2 kernels) with one fused kernel.
            from veloxvoice.kernels.ops import pwlin_glu

            if getattr(self, "_pwlin_bf", None) is None:
                self._pwlin_bf = (
                    self.cv1.weight.squeeze(-1).contiguous(),
                    self.cv1.bias.contiguous(),
                )
            wq, bq = self._pwlin_bf
            sg = pwlin_glu(x.reshape(-1, x.shape[-1]), wq, bq)
            if sg is not None:
                x = sg.reshape(x.shape).transpose(1, 2)  # [B, C, T]
            else:
                x = x.transpose(1, 2)
                x = self.cv1(x)
                x = F.glu(x, dim=1)
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

    def forward_chunk(self, x, offset, att_cache, conv_cache):
        x = x + self.ff_scale * self.ff_macaron(self._N(self.norm_ff_macaron, x))
        att_out, k, v = self.attn.forward_chunk(
            self._N(self.norm_mha, x), offset, att_cache[0], att_cache[1]
        )
        x = x + att_out
        conv_out, conv_cache = self.conv.forward_chunk(
            self._N(self.norm_conv, x), conv_cache
        )
        x = x + conv_out
        x = x + self.ff_scale * self.ff(self._N(self.norm_ff, x))
        return self._N(self.norm_final, x), (k, v), conv_cache

    def forward(self, x, mask, pos_emb, _):
        x = x + self.ff_scale * self.ff_macaron(self._N(self.norm_ff_macaron, x))
        # mask=None (full-context path): pass through so the attention
        # skips the additive mask entirely. Allocating a zeros [1,T,T] here
        # cost a memset + a full [1,h,T,T] broadcast add per layer per chunk.
        x = x + self.attn(self._N(self.norm_mha, x), mask, pos_emb)[0]
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

    def forward_chunk(self, x, offset, sub_cache, att_caches, conv_caches):
        y, sub_cache = self.embed.forward_chunk(x, sub_cache)
        if y.shape[0] == 0:
            return y, sub_cache, att_caches, conv_caches
        new_att, new_conv = [], []
        for layer, att, conv in zip(self.layers, att_caches, conv_caches):
            y, att, conv = layer.forward_chunk(y, offset, att, conv)
            new_att.append(att)
            new_conv.append(conv)
        y = self.after_norm(y)
        return y, sub_cache, new_att, new_conv

    def forward(self, x):  # x [1, T, 80]
        x3, pos = self.embed(x)
        for layer in self.layers:
            x3 = layer(x3, None, pos, None)
        return self.after_norm(x3)


class WenetConformerASR:
    """TorchWenEtModel-style Conformer for ASR"""

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda:0",
        required_cache_size: int = -1,
        *,
        cfg=None,
        vocab: int | None = None,
        state: dict | None = None,
    ):
        # NOTE (yiakwy) : torch, mlx backends
        self.backend = torch

        # Tensor-core fp32 (tf32) matmul: cuDNN convs already run tf32 by
        # default; leaving matmul at ieee pins every Linear/bmm to SIMT fp32
        # (~8x slower than tf32 tensor cores on GB10).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # NOTE (yiakwy) : targeted devices include cpu, nvgpu (amdgpu) and mlx (mps)
        self.device = device

        self.model_dir = model_dir

        self.cfg = cfg or load_config(model_dir)
        self.vocab = vocab
        self.cache_frames = required_cache_size if required_cache_size >= 0 else 256

        # NOTE (yiakwy) : main pt file of ASR model.  The API lane may pass a
        # preloaded state dict to avoid a second large-checkpoint read.
        if state is None:
            state = load_asr_state(self.model_dir)

        self.encoder = WenetConformerEncoder(self.cfg, state).to(device).eval()

        self.global_cmvn_mean = state["encoder.global_cmvn.mean"].to(device)
        self.global_cmvn_istd = state["encoder.global_cmvn.istd"].to(device)
        self.ctc_lo = (
            DenseLinear(state["ctc.ctc_lo.weight"], state["ctc.ctc_lo.bias"])
            .to(device)
            .eval()
        )

    def load_state_dict(self, sd, strict: bool = False):
        _normalize = torch_state_dict

        normalized = _normalize(sd)
        self.encoder.load_state_dict(normalized, strict=strict)
        ctc_sd = {
            k[len("ctc_lo.") :]: v
            for k, v in normalized.items()
            if k.startswith("ctc_lo.")
        }
        self.ctc_lo.load_state_dict(ctc_sd, strict=strict)

    def to(self, device):
        self.device = str(device)
        self.encoder.to(device)
        self.ctc_lo.to(device)
        self.global_cmvn_mean = self.global_cmvn_mean.to(device)
        self.global_cmvn_istd = self.global_cmvn_istd.to(device)
        return self

    def eval(self):
        self.encoder.eval()
        self.ctc_lo.eval()
        return self

    def set_jit(self, on: bool):
        asr_set_jit(on)
        for L in self.encoder.layers:
            # NOTE (yiakwy) : erase cached
            L.conv._pwlin = None
            L.conv._pwlin_bf = None

    def set_fp16(self, on: bool):
        """A2 ablation: autocast-cuda fp16 for the encoder (weights unchanged).
        autocast handles: Linear/Conv/matmul -> fp16 accum; LN/softmax/LayerNorm -> fp32 same path.
        """
        self.fp16_mode = bool(on)

    def set_precision(self, precision: str):
        """Set precision for all DenseLinear layers (fp32/nvfp4/mxfp4).

        mxfp4: weight pre-quantized once, activation online quantized per call.
        nvfp4: per-matrix absolute-max quantization, smart dispatch.
        bf16: cast encoder+ctc weights to bfloat16 (H800: ~2x GEMM/SDPA vs TF32;
              full-context lane measured 99.2% token parity, gate on CER not equality).
        """
        for module in self.encoder.modules():
            if isinstance(module, DenseLinear):
                module.precision = precision
                if precision in ("nvfp4", "mxfp4"):
                    module._prequantize_weight(precision)

        self.ctc_lo.precision = precision

        if precision in ("nvfp4", "mxfp4"):
            self.ctc_lo._prequantize_weight(precision)
        if precision == "bf16":
            self.encoder.bfloat16()
            self.ctc_lo.bfloat16()
            self.global_cmvn_mean = self.global_cmvn_mean.bfloat16()
            self.global_cmvn_istd = self.global_cmvn_istd.bfloat16()
        elif precision == "fp32":
            self.encoder.float()
            self.ctc_lo.float()
            self.global_cmvn_mean = self.global_cmvn_mean.float()
            self.global_cmvn_istd = self.global_cmvn_istd.float()

        for L in self.encoder.layers:
            L.conv._pwlin_bf = None

        if getattr(self, "_graphs", None):
            self._graphs.clear()

    @property
    def embeds_cmvn(self) -> bool:
        return True

    @property
    def pos_max_len(self) -> int:
        return int(self.encoder.embed.pos_pe.shape[1])

    def encode_utterance(self, feats):
        autocast = getattr(self.backend, "autocast", None)

        p = next(self.encoder.parameters())
        if feats.dtype != p.dtype:
            feats = feats.to(p.dtype)

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
            logp = z - torch.logsumexp(z, dim=-1, keepdim=True)
            return logp.float()

    def enable_graphs(self, max_graphs: int = 8):
        self._graphs_enabled = True
        self._max_graphs = int(max_graphs)
        self._graphs = {}
        self._graph_pool = self.backend.cuda.graph_pool_handle()

    def encode_ctc_utterance(self, feats):
        """feats [1, T, 80] -> CTC log-probs [1, T', V]."""

        p = next(self.encoder.parameters())
        if feats.dtype != p.dtype:
            feats = feats.to(p.dtype)

        if (
            getattr(self, "_graphs_enabled", False)
            and feats.shape[0] == 1
            and self.device.startswith("cuda")
            and feats.dtype == torch.bfloat16
        ):
            return self._encode_ctc_graphed(feats)
        with self.backend.inference_mode():
            return self.ctc_logp(self.encode_utterance(feats))

    def init_stream_state(self, device):
        d, h, k, cache_len = (
            self.cfg.output_size,
            self.cfg.attention_heads,
            self.cfg.cnn_module_kernel,
            self.cache_frames,
        )
        dtype = next(self.encoder.parameters()).dtype
        dk = d // h
        att = [
            (
                torch.zeros(h, dk, cache_len, device=device, dtype=dtype),
                torch.zeros(h, cache_len, dk, device=device, dtype=dtype),
            )
            for _ in range(self.cfg.num_blocks)
        ]
        conv = [
            torch.zeros(k - 1, d, device=device, dtype=dtype)
            for _ in range(self.cfg.num_blocks)
        ]
        sub = torch.zeros(0, self.cfg.input_dim, device=device, dtype=dtype)
        return dict(
            offset=torch.zeros((), dtype=torch.int32, device=device),
            sub=sub,
            att=att,
            conv=conv,
        )

    def encode_chunk(self, feats, state):
        p = next(self.encoder.parameters())
        if feats.dtype != p.dtype:
            feats = feats.to(p.dtype)
        out, sub, att, conv = self.encoder.forward_chunk(
            feats, state["offset"], state["sub"], state["att"], state["conv"]
        )
        state.update(offset=state["offset"] + out.shape[0], sub=sub, att=att, conv=conv)
        return out

    def encode_ctc_chunk(self, feats, state):
        return self.ctc_logp(self.encode_chunk(feats, state))

    def _encode_ctc_graphed(self, feats):
        backend = self.backend

        if not getattr(self, "_graphs_enabled", False):
            with backend.inference_mode():
                return self.ctc_logp(self.encode_utterance(feats))

        p = next(self.encoder.parameters())

        if feats.dtype != p.dtype:
            with backend.inference_mode():
                return self.ctc_logp(self.encode_utterance(feats))

        key = tuple(feats.shape)
        entry = self._graphs.get(key)

        if entry is not None:
            static_in, graph, static_out = entry
            static_in.copy_(feats, non_blocking=True)
            graph.replay()
            return static_out.clone()

        if len(self._graphs) >= self._max_graphs:
            self._graphs.clear()

        with backend.inference_mode(False):
            static_in = feats.detach().clone()
        with backend.inference_mode():
            self.ctc_logp(self.encode_utterance(static_in))  # 1 次预跑
        backend.cuda.synchronize()

        graph = backend.cuda.CUDAGraph()
        try:
            with backend.cuda.graph(graph, pool=self._graph_pool):
                static_out = self.ctc_logp(self.encode_utterance(static_in))

            graph.replay()

            backend.cuda.synchronize()
        except Exception as exc:
            import os as _os

            if _os.environ.get("VELOXVOICE_DEBUG_GRAPHS"):
                import traceback

                print(f"[graphs] capture failed for {key}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
            self._graphs_enabled = False

            with backend.inference_mode():
                return self.ctc_logp(self.encode_utterance(feats))
        self._graphs[key] = (static_in, graph, static_out)
        return static_out.clone()
