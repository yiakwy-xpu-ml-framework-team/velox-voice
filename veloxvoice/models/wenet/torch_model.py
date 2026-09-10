"""Torch-backend **torchscript loader**: wraps WeNet's exported torchscript
checkpoint bundle (`final.zip`) presenting its own black-box execution graph.

The exported torchscript provides:
  forward_encoder_chunk(xs[1,T,F], offset:int, required_cache_size:int,
                        att_cache, cnn_cache) -> (y, r_att_cache, r_cnn_cache)
  ctc_activation(xs) -> log_softmax over vocab
State (att_cache/cnn_cache/offset) lives on-device inside the session, so a
streaming chunk is pure GPU work — graph-capturable between the split ops.

Role in models/wenet/ layout (sanctioned):
  torch_model.py        = THIS FILE: torchscript bundle loader (TorchWenEtModel)
  torch_conformer.py    = torch Module implementations (WenetConformerASR) + cuda jit kernels (sm90a;sm121a)
  mlx_model.py          = same computation backbone over mlx.core + metal jit
                          kernels (Apple-Metal validated target)
"""

from __future__ import annotations

import os

import torch


# NOTE (yiakwy) : wenet e2e torchscript baseline
class TorchWenEtModel:
    def __init__(
        self, model_dir: str, device: str = "cuda:0", required_cache_size: int = -1
    ):
        self.backend = torch
        self.device = device

        self.model_dir = model_dir
        self.required_cache_size = required_cache_size

        # NOTE (yiakwy) : use wenet e2e torchscript runtime as baseline
        ckpt = os.path.join(model_dir, "final.zip")
        self.script = torch.jit.load(ckpt, map_location="cpu").to(device).eval()

    def new_session(self) -> "TorchWenEtSession":
        return TorchWenEtSession(self)

    @property
    def embeds_cmvn(self) -> bool:
        """True when the exported encoder applies global_cmvn internally
        (then the audio frontend must NOT apply CMVN again)."""
        try:
            self.script.encoder.global_cmvn
            return True
        except Exception:
            return False

    @property
    def pos_max_len(self) -> int:
        """Bundle's sinusoidal positional-encoding table length (encoder frames).
        forward_encoder_chunk asserts `offset + size <= max_len` — the absolute
        limit of any single full-context pass (np. wenet default 5000 frames).
        """
        try:
            return int(self.script.encoder.embed.pos_enc.pe.shape[1])
        except Exception:
            pass
        try:
            return int(self.script.encoder.embed.pos_enc.max_len)
        except Exception:
            return 5000

    def encode_utterance(self, feats):
        """Whole-utterance, full-context encode (offline lane).

        feats: [1, T, F] device tensor of kaldi fbank with NO externally-applied
        CMVN (the bundle's encoder.global_cmvn handles it). One-shot full
        attention = the bundle's offline training regime.
        """
        backend = self.backend
        with backend.inference_mode():
            att0 = backend.zeros(0, 0, 0, 0, device=feats.device)
            cnn0 = backend.zeros(0, 0, 0, 0, 0, device=feats.device)
            out, _, _ = self.script.forward_encoder_chunk(feats, 0, -1, att0, cnn0)
        return out

    def ctc_logp(self, enc_out):
        with self.backend.inference_mode():
            return self.script.ctc_activation(enc_out)


class TorchWenEtSession:
    def __init__(self, model: TorchWenEtModel):
        backend = model.backend
        self.m = model
        self.offset = 0
        self.att_cache = backend.zeros(0, 0, 0, 0, device=model.device)
        self.cnn_cache = backend.zeros(0, 0, 0, 0, 0, device=model.device)

    def encode_chunk(self, feats):
        """feats: [1, T, 80] on device (already CMVN'd) -> encoder out [1, T', D]."""
        backend = self.m.backend
        with backend.inference_mode():
            out, r_att, r_cnn = self.m.script.forward_encoder_chunk(
                feats,
                self.offset,
                self.m.required_cache_size,
                self.att_cache,
                self.cnn_cache,
            )
        self.att_cache, self.cnn_cache = r_att.to(self.m.device), r_cnn.to(
            self.m.device
        )
        self.offset += int(out.shape[1])
        return out

    def ctc_logp(self, enc_out):
        return self.m.ctc_logp(enc_out)
