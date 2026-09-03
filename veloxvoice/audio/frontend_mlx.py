"""GPU fbank frontend for the MLX/Metal backend (Mac Studio M3).

Identical math to the torch frontend; all tensors are `mx.array` on the Metal GPU
and stay lazy (only the recognizer forces evaluation).
"""

from __future__ import annotations

from .features import mel_filterbank, povey_window
from .frontend_torch import FrontendConfig, _FrameBookkeeping


class MlxGpuFrontend:
    def __init__(self, cfg: FrontendConfig):
        import mlx.core as mx

        self.mx = mx
        self.module = mx

        self.cfg = cfg

        self.fl = round(cfg.sample_rate * cfg.frame_length_ms / 1000)
        self.fs = round(cfg.sample_rate * cfg.frame_shift_ms / 1000)

        self.n_fft = 1
        while self.n_fft < self.fl:
            self.n_fft *= 2

        self.window = mx.array(povey_window(self.fl))
        self.mel = mx.array(
            mel_filterbank(cfg.num_mel_bins, self.n_fft, cfg.sample_rate)
        )
        self._book = _FrameBookkeeping(self.fl, self.fs)
        self._buf = mx.zeros((0,))
        self._tail = mx.zeros((1,))
        # gather indices for framing are precomputed lazily per request size
        self._idx_cache: dict[int, object] = {}

    def _frame_idx(self, n: int):
        mx = self.mx
        idx = self._idx_cache.get(n)
        if idx is None:
            idx = (
                mx.arange(n)[:, None] * self.fs + mx.arange(self.fl)[None, :]
            ).astype(mx.int32)
            self._idx_cache[n] = idx
        return idx

    def _preemph(self, pcm):
        x = self.mx.concatenate([self._tail, pcm])
        y = x[1:] - self.cfg.preemph * x[:-1]
        self._tail = pcm[-1:]
        return y

    def _frames_to_fbank(self, frames):
        mx = self.mx
        win = frames * self.window
        spec = mx.fft.rfft(win, n=self.n_fft)
        power = spec.real**2 + spec.imag**2
        feat = power @ self.mel.T
        feat = mx.log(mx.maximum(feat, 1.1920929e-7))
        if self.cfg.cmvn_mean is not None:
            feat = (feat - self.cfg.cmvn_mean) * self.cfg.cmvn_istd
        return feat

    def accept(self, pcm) -> object:
        mx = self.mx
        if not isinstance(pcm, mx.array):
            pcm = mx.array(pcm, dtype=mx.float32)
        if pcm.ndim > 1:
            pcm = pcm.reshape(-1)
        y = self._preemph(pcm * 32768.0)  # kaldi/WeNet int16 scaling
        self._buf = mx.concatenate([self._buf, y])

        n = self._book.n_complete(int(self._buf.shape[0]))
        if n == 0:
            return mx.zeros((0, self.cfg.num_mel_bins))
        idx = self._frame_idx(n)
        frames = self._buf[idx]  # [n, fl]
        self._buf = self._buf[n * self.fs :]
        return self._frames_to_fbank(frames)

    def flush(self) -> object:
        mx = self.mx
        r = int(self._buf.shape[0])
        if r < max(1, self.fs):
            return mx.zeros((0, self.cfg.num_mel_bins))
        padded = mx.concatenate([self._buf, mx.zeros((self.fl + self.fs - r,))])
        frames = padded[None, : self.fl]
        self._buf = mx.zeros((0,))
        return self._frames_to_fbank(frames)
