"""Mel filterbank math shared by all device frontends (numpy, host-side, computed once)."""

from __future__ import annotations

import numpy as np


def hz_to_mel_kaldi(f: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz_kaldi(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.exp(np.asarray(mel, dtype=np.float64) / 1127.0) - 1.0)


def mel_filterbank(
    num_mel_bins: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Kaldi-style triangular mel filters (no area normalization — kaldi/WeNet convention).

    Returns float32 [num_mel_bins, n_fft // 2 + 1].
    """
    fmax = float(fmax if fmax is not None else sample_rate / 2)
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2, n_bins)

    mel_pts = np.linspace(
        hz_to_mel_kaldi(fmin), hz_to_mel_kaldi(fmax), num_mel_bins + 2
    )
    hz_pts = mel_to_hz_kaldi(mel_pts)  # [num_mel+2] lower/center/upper

    fb = np.zeros((num_mel_bins, n_bins), dtype=np.float64)
    for i in range(num_mel_bins):
        lo, ctr, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        down = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.clip(np.minimum(up, down), 0.0, None)
    return fb.astype(np.float32)


def povey_window(frame_length: int) -> np.ndarray:
    """Kaldi povey window w = 0.5 - 0.5*cos(2*pi*n/(M-1)) (periodic hann)."""
    n = np.arange(frame_length, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / frame_length)).astype(np.float32)


def log_floor() -> float:
    # kaldi uses log(max(x, epsilon)) with epsilon ~ 1.19e-7 (default --use-log-fbank)
    return float(np.log(1.1920929e-7))
