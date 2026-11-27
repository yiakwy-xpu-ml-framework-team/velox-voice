"""Audio loading for the music lane: path / bytes / stream / waveform ->
mono float32 torch tensor @ 24 kHz (torchaudio, ffmpeg CLI fallback)."""

from __future__ import annotations

import io
from pathlib import Path

SAMPLE_RATE = 24000


def _load_with_ffmpeg(source):
    """ffmpeg CLI: decode + mono + 24 kHz resample in one pass (f32le)."""
    import subprocess

    import numpy as np
    import torch

    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(source), "-vn",
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    import numpy as np

    pcm = np.frombuffer(raw, dtype=np.float32)
    return torch.from_numpy(pcm.copy())[None], SAMPLE_RATE


def load_audio(source, *, sampling_rate: int = SAMPLE_RATE,
               max_seconds: float | None = None, preset: str = "default"):
    """Decode any supported source to a float32 mono waveform @ 24 kHz.

    Waveform inputs pass through with a resample; paths/bytes/streams go
    through torchaudio's ffmpeg backend (the ffmpeg CLI is the fallback).
    """
    import numpy as np
    import torch
    import torchaudio
    import torchaudio.functional as AF

    if isinstance(source, torch.Tensor):
        wav = source.detach().cpu().float()
        sr = int(sampling_rate)
    elif isinstance(source, np.ndarray):
        wav = torch.from_numpy(np.asarray(source)).float()
        sr = int(sampling_rate)
    else:
        # file/bytes sources go through the ffmpeg CLI exactly like the
        # reference loader (decode + mono + 24 kHz resample in one pass) —
        # a different resampler shifts the mel by ~0.7 dB and the
        # transcription drifts.
        wav, sr = _load_with_ffmpeg(source)

    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    if sr != SAMPLE_RATE:
        wav = AF.resample(wav, sr, SAMPLE_RATE)
    max_seconds = 3600.0 if preset == "default" else max_seconds
    if max_seconds is not None and wav.numel() > int(max_seconds * SAMPLE_RATE):
        wav = wav[: int(max_seconds * SAMPLE_RATE)]
    return wav.contiguous()
