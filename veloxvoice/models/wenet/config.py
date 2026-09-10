"""WeNet model config: parse train.yaml (pyyaml if available, else a small
flat-section parser) and the global_cmvn file."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np


@dataclass
class WeNetConfig:
    output_size: int = 256
    attention_heads: int = 4
    linear_units: int = 2048
    num_blocks: int = 12
    cnn_module_kernel: int = 15
    input_layer: str = "conv2d"  # "conv2d" (subsampling4) or "conv2d6" (subsampling6)
    input_dim: int = 80  # num_mel_bins
    right_context_frames: int = 6  # conv2d-4 subsampling lookahead (frames)
    decoding_chunk_size: int = 16  # mel frames per streaming chunk
    causal_conv: bool = True  # conv_module: rolling left-cache causal (u2++ streaming)
    # vs symmetric padding=(K-1)//2 window (offline "unstreaming" U2)

    @property
    def subsampling(self) -> int:
        return {"conv2d": 4, "conv2d6": 6}.get(self.input_layer, 4)

    @property
    def head_dim(self) -> int:
        return self.output_size // self.attention_heads

    @property
    def chunk_stride(self) -> int:
        return self.decoding_chunk_size * 160  # samples per chunk @16kHz, 10ms shift


def _mini_yaml(text: str) -> dict:
    """Parse a tiny YAML subset: 'key: value' pairs grouped under un-indented sections."""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        val = val.strip()
        if val == "":
            child: dict = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            if val.lower() in ("true", "false"):
                v: object = val.lower() == "true"
            else:
                try:
                    v = int(val)
                except ValueError:
                    try:
                        v = float(val)
                    except ValueError:
                        v = val
            parent[key.strip()] = v
    return root


def load_config(model_dir: str) -> WeNetConfig:
    yaml_path = os.path.join(model_dir, "train.yaml")
    with open(yaml_path, encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml

        conf = yaml.safe_load(text)
    except ImportError:
        conf = _mini_yaml(text)
    encoder_conf = conf.get("encoder_conf", {})
    model_dir_name = str(conf.get("model_dir", ""))
    causal = encoder_conf.get("causal", "unstreaming" not in model_dir_name)
    return WeNetConfig(
        output_size=int(encoder_conf.get("output_size", 256)),
        attention_heads=int(encoder_conf.get("attention_heads", 4)),
        linear_units=int(encoder_conf.get("linear_units", 2048)),
        num_blocks=int(encoder_conf.get("num_blocks", 12)),
        cnn_module_kernel=int(encoder_conf.get("cnn_module_kernel", 15)),
        input_layer=str(encoder_conf.get("input_layer", "conv2d")),
        input_dim=int(conf.get("input_dim", 80)),
        decoding_chunk_size=int(encoder_conf.get("decoding_chunk_size", 16)),
        causal_conv=bool(causal),
    )


def _stats_to_mean_istd(mean_stat, var_stat, frame_num):
    mean = np.asarray(mean_stat, dtype=np.float64) / float(frame_num)
    var = np.asarray(var_stat, dtype=np.float64) / float(frame_num) - mean**2
    return mean.astype(np.float32), (1.0 / np.sqrt(np.maximum(var, 1e-20))).astype(
        np.float32
    )


def load_cmvn(model_dir: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse CMVN stats -> (mean, istd) float32 [n_mel].

    Supports both wenet formats: the json dump (mean_stat/var_stat/frame_num) and
    the kaldi-text global_cmvn matrix file (AddShift/Rescale <Values> [...]).
    """
    text = None
    for name in ("cmvn.json", "global_cmvn.json", "global_cmvn"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            text = open(path, encoding="utf-8").read()
            break
    if text is None:
        return None
    if text.lstrip().startswith("{"):
        import json

        d = json.loads(text)
        return _stats_to_mean_istd(d["mean_stat"], d["var_stat"], d["frame_num"])
    vals = re.findall(r"<Values>\s*\[([^\]]*)\]", text)
    arrays = [np.fromstring(v, sep=" ", dtype=np.float64) for v in vals[:2]]
    if len(arrays) < 2:
        return None
    sums, sumsq = arrays
    n = (len(sums) - 1) // 2
    return _stats_to_mean_istd(sums[:n], sumsq[:n], sums[-1])
