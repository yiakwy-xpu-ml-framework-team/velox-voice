"""Piecewise CUDA Graph capturable utterance segmentation

Inspired by SGLang Piecewise CUDA Graph, this energy-VAD Semantics expressed as
device tensors so it can be captured inside `veloxvoice.graphs` segments.

Output is a (speech, start_sample_offset, end_sample_offset) vectorized per flank;
downstream utterance streaming gets chunk boundaries same way.
"""

from __future__ import annotations

import torch

# TODO (yiakwy) : optimize piecewise CUDA Graph capture for audio


def frame_energy_db(
    samples: torch.Tensor, frame_ms: int = 10, sample_rate: int = 16000
) -> torch.Tensor:
    """1-D float samples -> per-frame RMS energy in dB (torch tensor, stays on device)."""
    if samples.dim() > 1:
        samples = samples.reshape(-1)
    fin = sample_rate * frame_ms // 1000
    n = samples.shape[0] // fin * fin
    frames = samples[:n].view(-1, fin)
    return 20.0 * torch.log10(torch.clamp(frames.pow(2).mean(-1).sqrt(), min=1e-9))


class EnergyVADSegmenter:
    """Streaming-slice segmentation as GPU node: accepts one slice at a time,
    emits finished utterance boundaries (absolute sample offsets from stream start).
    Adaptive energy threshold from the 10th percentile + 2.5 dB headroom.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        min_sil_ms: int = 300,
        min_speech_ms: int = 250,
        merge_gap_ms: int = 300,
        device: str | None = None,
    ):
        self.sample_rate = sample_rate
        self.min_sil_ms = min_sil_ms
        self.min_speech_ms = min_speech_ms
        self.merge_gap_ms = merge_gap_ms
        self.device = torch.device(device) if device else None
        self._buf: torch.Tensor | None = None
        self._consumed = 0

    def _segs_from_vector(self, pcm: torch.Tensor, absolute: bool = True):
        db = frame_energy_db(
            pcm.to(self.device or pcm.device), sample_rate=self.sample_rate
        )
        floor = torch.quantile(db, 0.10) + 2.5
        spk = (db > floor).cpu().numpy()
        sr = self.sample_rate
        segs, start, last_active = [], None, 0
        for i, on in enumerate(spk):
            t = i * 10
            if on and start is None:
                start = t
            if on:
                last_active = t + 10
            if start is not None and (not on) and (t - last_active) >= self.min_sil_ms:
                if last_active - start >= self.min_speech_ms:
                    segs.append((start, last_active))
                start = None
        if start is not None:
            segs.append((start, last_active))
        return segs

    def accept(self, chunk: torch.Tensor) -> list[tuple[int, int]]:
        if self.device is None:
            self.device = chunk.device
        chunk = chunk.reshape(-1).to(self.device)
        self._buf = chunk if self._buf is None else torch.cat([self._buf, chunk])
        out = []
        if self._buf.shape[0] >= int(0.6 * self.sample_rate):
            segs = self._segs_from_vector(self._buf)
            if segs:
                last_end_samp = int(segs[-1][1] / 1000 * self.sample_rate)
                out = (
                    segs[:-1]
                    if len(segs) > 1
                    else ([] if self._buf.shape[0] - last_end_samp > 0 else [])
                )
                self._consumed += last_end_samp
                self._buf = self._buf[last_end_samp:]
        return out

    def finish(self) -> list[tuple[int, int]]:
        if self._buf is None or self._buf.shape[0] == 0:
            return []
        out = self._segs_from_vector(self._buf)
        self._buf = None
        return out
