"""StreamingRecognizer: pcm chunks -> partial text.

Loop invariant per accept(pcm): all arithmetic runs on the accelerator; the only
host sync is ids->TextTokenizer once per emitted chunk (the user-visible result).
"""

from __future__ import annotations


class StreamingRecognizer:
    def __init__(
        self,
        model,
        session,
        frontend,
        voice_tokenizer,
        text_tokenizer,
        chunk_frames: int = 16,
        subsampling: int = 4,
        device: str | None = None,
    ):
        self.model = model
        self.session = session  # TorchWenEtSession | None (mlx uses state)
        self.frontend = frontend
        self.voice_tokenizer = voice_tokenizer
        self.text = text_tokenizer
        self.chunk_frames = chunk_frames
        self.subsampling = subsampling
        self._pending = None  # device array [N, n_mel] leftovers
        self._pending_codes = None  # parallel FSQ voice-token stream
        self._device = device
        self._mlx_state = model.init_stream_state(device) if session is None else None
        from .ctc import CtcPrefixBeamDecoder

        self.ctc = CtcPrefixBeamDecoder(beam_size=8, nbest=1)

    def _cat(self, a, b):
        if a is None:
            return b
        mod = type(b).__module__.split(".")[0]
        if mod == "torch":
            import torch

            return torch.cat([a, b])
        import mlx.core as mx

        return mx.concatenate([a, b])

    def _pad_to(self, x, n):
        mod = type(x).__module__.split(".")[0]
        pad = n - x.shape[0]
        if pad <= 0:
            return x
        if mod == "torch":
            import torch

            return torch.cat([x, x[-1:].repeat(pad, 1)])
        import mlx.core as mx

        return mx.concatenate([x, mx.repeat(x[-1:], pad, axis=0)])

    def _run_encoder_ctc(self, feats):
        """feats [T, n_mel] -> newly decoded token ids."""
        if self.session is not None:  # torch backend
            enc = self.session.encode_chunk(feats[None])
            logp = self.session.ctc_logp(enc)[0]
        else:  # mlx backend (stateful model + state dict)
            logp = self.model.encode_ctc_chunk(feats, self._mlx_state)
        return self.ctc.push_logp(logp)

    def accept(self, pcm_chunk) -> list[int]:
        tok = self.voice_tokenizer.encode_chunk(pcm_chunk)
        self._pending = self._cat(self._pending, tok.data)
        if tok.codes is not None:
            self._pending_codes = self._cat(self._pending_codes, tok.codes)
        emitted: list[int] = []
        while self._pending is not None and self._pending.shape[0] >= self.chunk_frames:
            head, self._pending = (
                self._pending[: self.chunk_frames],
                self._pending[self.chunk_frames :],
            )
            if self._pending_codes is not None:
                self._pending_codes = self._pending_codes[self.chunk_frames :]
            emitted += self._run_encoder_ctc(head)
        return emitted

    def voice_tokens(self):
        """The parallel FSQ voice-token stream (discrete channel) for this session —
        the uniform structure later consumed by the AR TTS model."""
        return self._pending_codes

    def finish(self, min_frames: int | None = None) -> list[int]:
        emitted: list[int] = []
        flush = self.voice_tokenizer.flush()
        self._pending = self._cat(self._pending, flush.data)
        if self._pending is not None and self._pending.shape[0] > 0:
            floor_pad = max(
                min_frames or 0, self.subsampling * 6
            )  # heads-up both direction ahead
            target = max(floor_pad, self._pending.shape[0])
            target = (
                (target + self.subsampling - 1) // self.subsampling * self.subsampling
            )
            emitted = self._run_encoder_ctc(self._pad_to(self._pending, target))
            self._pending = None
        return emitted

    def text_of(self, ids) -> str:
        return self.text.ids_to_text(list(ids))
