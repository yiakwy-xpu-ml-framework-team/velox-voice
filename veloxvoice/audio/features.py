"""Mel filterbank math shared by all device frontends (numpy, host-side, computed once)."""

from __future__ import annotations

import numpy as np


def hz_to_mel_kaldi(f: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz_kaldi(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.exp(np.asarray(mel, dtype=np.float64) / 1127.0) - 1.0)


def legacy_triangular_mel_filterbank(
    num_mel_bins: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 20.0, # NOTE (yiakwy) : low frequencies to exclude
    fmax: float | None = None,
) -> np.ndarray:
    """Kaldi-style triangular mel filters.

    ``fmin=20`` default, ``fmax`` = 0.5 * sample_rate.

    Returns float32 [num_mel_bins, n_fft // 2 + 1].
    """
    fmax = float(fmax if fmax is not None else 0.5 * sample_rate)

    n_fft_bins = n_fft // 2 + 1

    fft_freqs = np.linspace(0.0, sample_rate / 2, n_fft_bins)

    # NOTE (yiakwy) : Kaldi-style triangular mel
    mel_pts = np.linspace(
        hz_to_mel_kaldi(fmin), hz_to_mel_kaldi(fmax), num_mel_bins + 2
    )
    hz_pts = mel_to_hz_kaldi(mel_pts)  # [num_mel+2] lower/center/upper

    fb = np.zeros((num_mel_bins, n_fft_bins), dtype=np.float64)
    for i in range(num_mel_bins):
        lo, ctr, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        down = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.clip(np.minimum(up, down), 0.0, None)

    return fb.astype(np.float32)

def mel_filterbank(
    num_mel_bins: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 20.0, # NOTE (yiakwy) : low frequencies to exclude
    fmax: float | None = None,
) -> np.ndarray:
    """Kaldi mel filters (``torchaudio.compliance.kaldi.get_mel_banks``).

    ``fmin=20`` default, ``fmax`` = Nyquist; ``num_fft_bins = n_fft // 2`` and the banks 
    are right-padded with one zero column to ``n_fft // 2 + 1``.

    Returns float32 [num_mel_bins, n_fft // 2 + 1].
    """
    if fmax is None or fmax <= 0.0:
        fmax = 0.5 * sample_rate
    else:
        pass

    n_fft_bins = n_fft // 2

    fft_bin_width = sample_rate / n_fft

    mel_low = hz_to_mel_kaldi(np.array(fmin, dtype=np.float64))
    mel_high = hz_to_mel_kaldi(np.array(fmax, dtype=np.float64))

    delta = (mel_high - mel_low) / (num_mel_bins + 1)

    bins = np.arange(num_mel_bins, dtype=np.float64)[:, None]
    left = mel_low + bins * delta
    center = mel_low + (bins + 1.0) * delta
    right = mel_low + (bins + 2.0) * delta

    mel = hz_to_mel_kaldi(fft_bin_width * np.arange(n_fft_bins, dtype=np.float64))[None, :]

    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)

    fb = np.clip(np.minimum(up, down), 0.0, None)  # [num_mel, n_fft//2]
    fb = np.pad(fb, ((0, 0), (0, 1)), mode="constant")  # +1 zero column
    return fb.astype(np.float32)


def povey_window(frame_length: int) -> np.ndarray:
    """Kaldi povey window w = hann(periodic=False) ** 0.85."""
    n = np.arange(frame_length, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (frame_length - 1))
    return (hann**0.85).astype(np.float32)


def log_floor() -> float:
    # kaldi uses log(max(x, epsilon)) with epsilon ~ 1.19e-7 (default --use-log-fbank)
    return float(np.log(1.1920929e-7))
