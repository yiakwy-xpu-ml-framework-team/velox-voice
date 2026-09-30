"""Top-level user API: identical frontends on both platforms."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import numpy as np

from .runtime.device import detect_platform

_SAMPLE_RATE = 16000


@dataclass
class TranscriptionResult:
    """Normalized result for the API-level full-context transcriber."""

    text: str
    ids: list[int]
    spans: list[list[int]]
    audio_seconds: float
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "ids": [int(t) for t in self.ids],
            "spans": [[int(a), int(b)] for a, b in self.spans],
            "audio_seconds": self.audio_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "rtf": self.rtf,
        }

    @property
    def rtf(self) -> float:
        if self.audio_seconds <= 0:
            return 0.0
        return self.elapsed_seconds / self.audio_seconds


class Velox:
    """Public API: native WeNet Conformer + optimized Hopper/DGX Spark/Metal kernels.

    Example:
        vx = Velox.load(model_dir, native_kernels=True)
        result = vx.transcribe_pcm(pcm16k)
        print(result.text, result.rtf)
    """

    def __init__(
        self,
        platform,
        model,
        session_factory,
        frontend_cfg,
        text_tok,
        cfg,
        use_graphs="eager",
        precision: str = "bf16",
        enable_graphs: bool = False,
    ):
        self.platform = platform
        self._model = model
        self._session_factory = session_factory
        self._frontend_cfg = frontend_cfg
        self._text = text_tok
        self.cfg = cfg
        self.use_graphs = use_graphs
        self.precision = precision
        self.enable_graphs = enable_graphs
        self.model_dir = getattr(model, "model_dir", "")
        self._lock = threading.Lock()
        self._warmed = False

    @classmethod
    def load(
        cls,
        model_dir: str,
        backend: str = "auto",
        device: str | None = None,
        cache_frames: int = 256,
        use_graphs: str = "eager",
        *,
        native_kernels: bool = True,
        kernel_precision: str = "bf16",
        enable_graphs: bool = False,
    ) -> "Velox":
        """Load a WeNet bundle with the public API with optimized kernels"""
        from .models.wenet.config import load_cmvn, load_config
        from .runtime.device import detect_platform

        # NOTE (yiakwy) : mlx modules for Apple Metal GPU; pytorch modules for NVIDIA GPU
        platform = (
            detect_platform(refresh=True) if backend == "auto" else _force(backend)
        )
        cfg = load_config(model_dir)

        from .audio.text_tokenizer import TextTokenizer

        # NOTE (yiakwy) : units.txt is from wenet base ASR model directory
        text_tok = TextTokenizer(os.path.join(model_dir, "units.txt"))

        cmvn = load_cmvn(model_dir)

        from .audio import FrontendConfig

        # TODO (yiakwy) : user can adjust sample rate via restful API and frontend config
        fcfg = FrontendConfig(sample_rate=_SAMPLE_RATE, num_mel_bins=cfg.input_dim)
        device = device or platform.device

        if platform.module_backend == "torch":
            from .models.wenet.torch_conformer import WenetConformerASR, load_asr_state

            # Load once: the same state supplies CTC vocab discovery and model init.
            state = load_asr_state(model_dir)
            vocab = _vocab_from_state(state, default=text_tok.vocab_size)
            if vocab != text_tok.vocab_size:
                text_tok.units = text_tok.units[:vocab]

            model = WenetConformerASR(
                model_dir,
                device=device,
                required_cache_size=cache_frames,
                cfg=cfg,
                vocab=vocab,
                state=state,
            )
            del state
            factory = lambda: None  # state lives inside the recognizer session

            if native_kernels and hasattr(model, "set_precision"):
                model.set_precision(kernel_precision)
                model.set_jit(True)
            if enable_graphs and hasattr(model, "enable_graphs"):
                model.enable_graphs()
        else:
            # TODO (yiakwy) : double check in M3/M5 Ultra and M4 Max
            import mlx.core as mx

            from .models.wenet.convert import load_model_from_checkpoint
            from .models.wenet.torch_conformer import load_asr_state

            vocab = _vocab_from_state(
                load_asr_state(model_dir), default=text_tok.vocab_size
            )
            if vocab != text_tok.vocab_size:
                text_tok.units = text_tok.units[:vocab]

            if cmvn is not None:
                fcfg.cmvn_mean = mx.array(cmvn[0])
                fcfg.cmvn_istd = mx.array(cmvn[1])
            model = load_model_from_checkpoint(
                model_dir, cfg, vocab, cache_frames=cache_frames
            )
            factory = lambda: None  # mlx path state lives in the recognizer

        if cmvn is not None and not getattr(model, "embeds_cmvn", False):
            if platform.module_backend == "torch":
                import torch

                fcfg.cmvn_mean = torch.tensor(cmvn[0], device=device)
                fcfg.cmvn_istd = torch.tensor(cmvn[1], device=device)

        # The native encoder owns global CMVN
        if getattr(model, "embeds_cmvn", False):
            fcfg.cmvn_mean = None
            fcfg.cmvn_istd = None

        return cls(
            platform,
            model,
            factory,
            fcfg,
            text_tok,
            cfg,
            use_graphs=use_graphs,
            precision=kernel_precision,
            enable_graphs=enable_graphs,
        )

    def new_session(self, chunk_frames: int | None = None) -> "StreamingSession":
        return StreamingSession(self, chunk_frames or self.cfg.decoding_chunk_size)

    def warmup(self, seconds: Iterable[float] = (0.5, 1.0, 5.0, 20.0)) -> "Velox":
        """Warm the server for common audio durations."""
        if self._warmed or self.platform.module_backend != "torch":
            return self

        import torch

        with self._lock:
            if self._warmed:
                return self
            with torch.inference_mode():
                from .audio import make_frontend

                fe = make_frontend(self._frontend_cfg, device=self.platform.device)
                for sec in dict.fromkeys(seconds):

                    samples = int(round(_SAMPLE_RATE * float(sec)))

                    feats = torch.cat(
                        [
                            fe.accept(
                                torch.zeros(samples, device=self.platform.device)
                            ),
                            fe.flush(),
                        ],
                        dim=0,
                    )

                    self._model.encode_ctc_utterance(feats[None])

            if self.platform.device.startswith("cuda"):
                torch.cuda.synchronize()

            self._warmed = True
        return self

    # our main API refactored from old codes
    def transcribe_pcm(
        self, pcm, sample_rate: int = _SAMPLE_RATE
    ) -> TranscriptionResult:
        """Full-context transcribe from mono float PCM in [-1, 1].

        Other sample rates are resampled with ffmpeg.  This is the benchmark-
        equivalent API lane: frontend -> low-energy spans -> native encoder +
        CTC collapse, all in-process.
        """
        if sample_rate != _SAMPLE_RATE:
            pcm = _resample_pcm16k(pcm, sample_rate)

        import torch

        self.warmup()
        pcm_t = torch.from_numpy(
            np.array(pcm, dtype=np.float32, copy=True, order="C").reshape(-1)
        )
        audio_seconds = float(pcm_t.numel() / _SAMPLE_RATE)
        if audio_seconds <= 0:
            return TranscriptionResult("", [], [], 0.0, 0.0)

        from time import perf_counter

        started = perf_counter()
        with self._lock:
            from .audio import make_frontend

            fe = make_frontend(self._frontend_cfg, device=self.platform.device)
            with torch.inference_mode():
                feats = torch.cat([fe.accept(pcm_t), fe.flush()], dim=0)
            spans = self._spans_for(feats, pcm_t)

            # On-device CTC collapse: only the compact token list is copied back.
            ids: list[int] = []
            last = 0
            with torch.inference_mode():
                for start, end in spans:
                    logp = self._model.encode_ctc_utterance(feats[start:end][None])
                    frame_ids = logp[0].argmax(-1).reshape(-1)
                    prev = torch.cat((frame_ids.new_full((1,), last), frame_ids[:-1]))
                    keep = (frame_ids != 0) & (frame_ids != prev)
                    ids.extend(frame_ids[keep].cpu().tolist())
                    if frame_ids.numel():
                        last = int(frame_ids[-1])
            if self.platform.device.startswith("cuda"):
                torch.cuda.synchronize()
        elapsed = perf_counter() - started
        text = self._text.ids_to_text(ids)
        return TranscriptionResult(
            text=text,
            ids=ids,
            spans=[[int(a), int(b)] for a, b in spans],
            audio_seconds=audio_seconds,
            elapsed_seconds=elapsed,
        )

    def _spans_for(self, feats, pcm) -> list[tuple[int, int]]:
        import torch

        max_len = max(getattr(self._model, "pos_max_len", 5000), 1000)
        hard_cap = (max_len - 200) * self.cfg.subsampling
        max_frames = min(15000, hard_cap)
        n = int(feats.shape[0])
        if n <= max_frames:
            return [(0, n)]

        from .audio.vad_energy import frame_energy_db

        db = (
            frame_energy_db(torch.as_tensor(pcm, dtype=torch.float32).reshape(-1).cpu())
            .cpu()
            .numpy()
        )
        return _low_energy_spans(n, max_frames, db)

    def ctc_logp_pcm(self, pcm, sample_rate: int = _SAMPLE_RATE):
        """Return full-context CTC log-probs and the frame subsampling rate.

        This is useful for token/diarization alignment without exposing the
        underlying model object.
        """
        if sample_rate != _SAMPLE_RATE:
            pcm = _resample_pcm16k(pcm, sample_rate)

        import torch

        self.warmup()
        pcm_t = torch.from_numpy(
            np.array(pcm, dtype=np.float32, copy=True, order="C").reshape(-1)
        )
        if pcm_t.numel() == 0:
            return torch.empty(0, 0), self.cfg.subsampling

        from .audio import make_frontend

        with self._lock, torch.inference_mode():
            fe = make_frontend(self._frontend_cfg, device=self.platform.device)
            feats = torch.cat([fe.accept(pcm_t), fe.flush()], dim=0)
            spans = self._spans_for(feats, pcm_t)
            pieces = [
                self._model.encode_ctc_utterance(feats[start:end][None])[0]
                for start, end in spans
            ]
            if self.platform.device.startswith("cuda"):
                torch.cuda.synchronize()
        logp = torch.cat(pieces, dim=0) if pieces else torch.empty(0, 0)
        return logp, self.cfg.subsampling

    def transcribe_pcm_iter(
        self,
        pcm_iter: Iterable[Any],
        sample_rate: int = _SAMPLE_RATE,
        pool_seconds: float = 20.0,
    ) -> Iterator[TranscriptionResult]:
        """Stream pooled full-context transcriptions from 16 kHz mono PCM.

        For incremental input, pool enough audio before invoking the encoder.
        Other sample rates should first be decoded/resampled as one stream;
        chunk-wise resampling would create filter-boundary artifacts.
        """
        if sample_rate != _SAMPLE_RATE:
            raise ValueError("transcribe_pcm_iter requires 16 kHz mono PCM")

        buffer: list[np.ndarray] = []
        buffered = 0
        target = max(1, int(pool_seconds * _SAMPLE_RATE))

        for chunk in pcm_iter:
            arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if arr.size:
                buffer.append(arr)
                buffered += arr.size

            while buffered >= target:
                pcm = np.concatenate(buffer)
                piece = pcm[:target]
                rest = pcm[target:]
                yield self.transcribe_pcm(piece)
                buffer = [rest] if rest.size else []
                buffered = int(rest.size)

        if buffered:
            yield self.transcribe_pcm(np.concatenate(buffer))

    def transcribe(self, pcm, chunk_frames: int | None = None) -> str:
        """Compatibility API returning plain text."""
        del chunk_frames  # full-context lane does not use streaming chunk size
        return self.transcribe_pcm(pcm).text


class StreamingSession:
    def __init__(self, velox: Velox, chunk_frames: int):
        from .audio import make_frontend
        from .stream.recognizer import StreamingRecognizer
        from .vocoding import ContinuousVoiceTokenizer

        frontend = make_frontend(velox._frontend_cfg, device=velox.platform.device)
        vt = ContinuousVoiceTokenizer(frontend)
        model = velox._model
        session = velox._session_factory()
        if session is None and velox.use_graphs != "eager":
            if velox.platform.module_backend == "torch":
                from .graphs.chunk_pipeline import CudaGraphChunkRunner

                mode = (
                    velox.use_graphs
                    if velox.use_graphs in ("cuda-graph", "cuda-graph-fused")
                    else "cuda-graph"
                )
                model = CudaGraphChunkRunner(model, mode=mode)
            else:
                from .graphs.chunk_pipeline import MlxCompiledChunkRunner

                model = MlxCompiledChunkRunner(model, mode="metal-graph")
        self._v = velox
        self._r = StreamingRecognizer(
            model,
            session,
            frontend,
            vt,
            velox._text,
            chunk_frames,
            device=velox.platform.device,
        )

    # pcm: 1-D float32 numpy in [-1, 1] at 16 kHz
    def accept(self, pcm) -> str:
        return self._r.text_of(self._r.accept(pcm))

    def finish(self) -> str:
        return self._r.text_of(self._r.finish())

    def tokens(self):
        return self._r.ctc.tokens

    def text(self) -> str:
        return self._v._text.ids_to_text(self.tokens())

    def stream(self, pcm_iter):
        for chunk in pcm_iter:
            new = self.accept(chunk)
            if new:
                yield self.text()  # partial transcription after each chunk
        self.finish()
        yield self.text()


def _low_energy_spans(n_frames: int, max_frames: int, db) -> list[tuple[int, int]]:
    """Split [0, n_frames) into spans of <= max_frames, cutting each span at the
    lowest-energy point found in a backward search window (<= 30 s) from the
    nominal cut, so segment edges fall inside pauses rather than mid-word."""
    spans = []
    a = 0
    while n_frames - a > max_frames:
        nominal = a + max_frames
        lo = max(a + max_frames // 4, nominal - 3000)  # search back <= 30 s
        hi = min(nominal, len(db) - 1)
        cut = lo + int(np.argmin(db[lo:hi])) if hi > lo else nominal
        cut = max(cut, a + 1)
        spans.append((a, cut))
        a = cut
    spans.append((a, n_frames))
    return spans


def _force(backend: str):
    from .runtime import device as dev

    if backend == "mlx":
        return dev._detect_metal()
    if backend == "torch":
        return dev._detect_torch()
    raise ValueError(backend)


def _vocab_from_state(state: dict, default: int) -> int:
    """Read CTC output size from an already-loaded WeNet checkpoint."""
    weight = state.get("ctc.ctc_lo.weight")
    if weight is None:
        return default
    return int(weight.shape[0])


def _resample_pcm16k(pcm, sample_rate: int):
    """Small dependency-free HTTP/WebRTC helper."""
    import io
    import subprocess
    import wave

    pcm_np = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if sample_rate == _SAMPLE_RATE:
        return pcm_np

    pcm_i16 = np.clip(pcm_np * 32768.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_i16.tobytes())

    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-ac",
            "1",
            "-ar",
            f"{_SAMPLE_RATE}",
            "-f",
            "f32le",
            "-",
        ],
        input=buf.getvalue(),
        stdout=subprocess.PIPE,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32)
