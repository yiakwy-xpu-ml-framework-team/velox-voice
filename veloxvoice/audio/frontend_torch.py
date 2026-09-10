"""GPU fbank frontend for torch/CUDA backends (H800 sm90, DGX-Spark sm121).

Everything after the host->device PCM upload stays on the GPU:
pre-emphasis, framing (unfold), povey window, rFFT power, mel matmul, log, CMVN.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .features import log_floor, mel_filterbank, povey_window


@dataclass
class FrontendConfig:
    sample_rate: int = 16000
    num_mel_bins: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    preemph: float = 0.97
    cmvn_mean: "object | None" = None  # [n_mel] device array
    cmvn_istd: "object | None" = None
    precision: str = "bf16"  # "fp32", "bf16", "mxfp4"


class _FrameBookkeeping:
    """CPU-scalar bookkeeping: which frames are complete given buffered samples."""

    def __init__(self, frame_length: int, frame_shift: int):
        self.fl = frame_length
        self.fs = frame_shift

    def n_complete(self, buf_len: int) -> int:
        return max(0, (buf_len - self.fl) // self.fs + 1)


class TorchGpuFrontend:
    def __init__(self, cfg: FrontendConfig, device: str = "cuda:0"):

        self.backend = torch
        self.cfg = cfg
        self.device = device
        c = cfg
        self.fl = int(round(c.sample_rate * c.frame_length_ms / 1000))
        self.fs = int(round(c.sample_rate * c.frame_shift_ms / 1000))
        self.n_fft = 1
        while self.n_fft < self.fl:
            self.n_fft *= 2

        self.window = torch.from_numpy(povey_window(self.fl)).to(device)
        mel = mel_filterbank(c.num_mel_bins, self.n_fft, c.sample_rate)
        self.mel = torch.from_numpy(mel).to(device)  # [M, F]
        self._book = _FrameBookkeeping(self.fl, self.fs)
        self._buf = torch.zeros(0, device=device)  # pre-emphasized samples
        self._tail = torch.zeros(1, device=device)

        # Pre-quantize mel for mxfp4 path
        self._mel_codes = None
        self._mel_scales = None
        self._F_padded = None
        self._M_padded = None
        if c.precision == "mxfp4" and device.startswith("cuda"):
            from veloxvoice.kernels.ops.triton_ops import mxfp4_block_quantize

            F = self.mel.shape[1]
            M = self.mel.shape[0]
            BK, BN = 64, 128
            F_padded = ((F + BK - 1) // BK) * BK
            M_padded = ((M + BN - 1) // BN) * BN
            self._F_padded = F_padded
            self._M_padded = M_padded
            mel_padded = torch.zeros(
                M_padded, F_padded, device=device, dtype=torch.float32
            )
            mel_padded[:M, :F] = self.mel
            self._mel_codes, self._mel_scales = mxfp4_block_quantize(
                mel_padded, mode="B"
            )
            # Free the padded mel tensor
            del mel_padded

    def _preemph(self, pcm):
        """pcm: 1-D device tensor. Prepends tail sample so the filter is seamless."""
        x = self.backend.cat([self._tail, pcm])
        y = x[1:] - self.cfg.preemph * x[:-1]
        self._tail = pcm[-1:]
        return y

    def _frames_to_fbank(self, frames):
        """frames: [T, fl] device."""
        backend = self.backend
        win = frames * self.window
        spec = backend.fft.rfft(win, n=self.n_fft)  # [T, F] complex
        if self.device.startswith("cuda"):
            precision = self.cfg.precision
            if precision == "mxfp4" and self._mel_codes is not None:
                from veloxvoice.kernels.ops import power_mel_log_mxfp4

                # GEMM works on M_padded=128 columns; slice back to the model's
                # num_mel_bins (padded columns are garbage from zero-power rows).
                return power_mel_log_mxfp4(
                    spec.contiguous(),
                    self._mel_codes,
                    self._mel_scales,
                    self.cfg.cmvn_mean,
                    self.cfg.cmvn_istd,
                    self._F_padded,
                    self._M_padded,
                )[:, : self.cfg.num_mel_bins]
            else:
                from veloxvoice.kernels.ops import power_mel_log

                return power_mel_log(
                    spec.contiguous(), self.mel, self.cfg.cmvn_mean, self.cfg.cmvn_istd
                )
        power = spec.real**2 + spec.imag**2
        feat = power @ self.mel.T  # [T, M]
        feat = backend.log(backend.clamp_min(feat, float(1.1920929e-7)))
        if self.cfg.cmvn_mean is not None:
            feat = (feat - self.cfg.cmvn_mean) * self.cfg.cmvn_istd
        return feat

    def accept(self, pcm) -> "object":
        """Accept new 16-bit-normalized PCM (numpy or torch), return new fbank
        frames [T_new, n_mel] on device (may be empty)."""
        backend = self.backend
        if not isinstance(pcm, backend.Tensor):
            pcm = backend.as_tensor(pcm, dtype=backend.float32)
        pcm = pcm.to(self.device) * 32768.0  # kaldi/WeNet int16 scaling
        y = self._preemph(pcm)
        self._buf = backend.cat([self._buf, y])

        n = self._book.n_complete(int(self._buf.shape[0]))
        if n == 0:
            return self._buf.new_zeros((0, self.cfg.num_mel_bins))
        end = (n - 1) * self.fs + self.fl
        frames = self._buf[:end].unfold(0, self.fl, self.fs)  # [n, fl]
        self._buf = self._buf[n * self.fs :]
        return self._frames_to_fbank(frames)

    def flush(self) -> "object":
        """Flush trailing partial frame (zero-padded to frame_length)."""
        backend = self.backend
        r = int(self._buf.shape[0])
        if r < max(1, self.fs):
            return self._buf.new_zeros((0, self.cfg.num_mel_bins))
        pad = self.fl + self.fs - r  # one extra frame
        padded = backend.cat([self._buf, backend.zeros(pad, device=self.device)])
        frames = padded[: (0) * self.fs + self.fl][None, :]  # single frame
        self._buf = self._buf.new_zeros(0)
        return self._frames_to_fbank(frames)
