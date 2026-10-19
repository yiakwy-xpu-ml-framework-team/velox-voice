"""SheetSage2 v2 configuration"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MERT2Config:
    """MERT2-FullSong encoder: mel -> ConvNext subsample -> 24x Conformer."""

    sampling_rate: int = 24000
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 240
    num_mel_bins: int = 128
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    conv_depthwise_kernel_size: int = 31
    rotary_embedding_base: int = 10000
    layer_norm_eps: float = 1e-5
    subsampling_channels: tuple = (128, 512, 1024)
    subsampling_depths: tuple = (3, 4, 5)
    subsampling_layer_norm_eps: float = 1e-6
    frame_rate: float = 25.0
    inputs_to_logits_ratio: int = 960  # samples per output frame
    minimum_input_samples: int = 1025

    @classmethod
    def from_dict(cls, d: dict) -> "MERT2Config":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class SheetSage2Config:
    """Music-to-lead-sheet head: layer-weighted encoder mix + Bart decoder."""

    vocab_size: int = 31678
    hidden_size: int = 512
    decoder_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    max_output_seq_len: int = 5120
    decoder_dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    lora_rank: int = 64
    lora_alpha: float = 128.0
    input_audio_length: float = 300.0
    time_hz: int = 100
    tokenizer_schema_version: str = "v1"
    tokenizer_fingerprint: str = "5ba3325af0344c7f"
    backbone_config: dict = field(default_factory=dict)

    model_dir: str = ""

    @classmethod
    def from_json(cls, path) -> "SheetSage2Config":
        d = json.load(open(path, encoding="utf-8"))
        backbone = d.pop("backbone_config", {})
        known = {f.name for f in cls.__dataclass_fields__.values()}
        cfg = cls(**{k: v for k, v in d.items() if k in known},
                  model_dir=str(path))
        object.__setattr__(cfg, "backbone_config", backbone)
        return cfg

    @property
    def mert(self) -> MERT2Config:
        return MERT2Config.from_dict(self.backbone_config)
