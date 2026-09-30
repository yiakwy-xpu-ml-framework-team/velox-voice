"""Streaming CTC decoding.

Greedy (`CtcGreedyDecoder`) collapses repeats/blanks across chunks. The quality path
(`CtcPrefixBeamDecoder`) runs prefix beam search over the same stream for the
accuracy-critical production lane.
"""

from __future__ import annotations

import numpy as np

from .ctc_beam import prefix_beam_search


class CtcGreedyDecoder:
    def __init__(self, blank: int = 0, dedup: bool = False):
        self.blank = blank

        # NOTE (yiakwy) : follow benchmark to disable decrease duplicated remover
        self.dedup = bool(dedup)
        self.tokens: list[int] = []
        self._last: int = blank

    # TODO (yiakwy) : move to gpu
    def push_ids(self, ids) -> list[int]:
        """Feed the argmax ids of one chunk; returns newly emitted token ids."""
        new: list[int] = []
        last = self._last
        for tok in ids:
            tok = int(tok)
            if tok != last and tok != self.blank:
                new.append(tok)
            last = tok
        self._last = last
        self.tokens.extend(new)
        return new

    # TODO (yiakwy) : move to gpu
    def push_logp(self, logp):
        import torch as backend

        ids = logp.argmax(-1)
        if backend.is_tensor(ids) and getattr(ids, "is_cuda", False):
            # collapse adjacent duplicate frames on-device (runs -> first frame);
            flat = ids.reshape(-1)
            keep = backend.ones_like(flat, dtype=backend.bool)
            keep[1:] = flat[1:] != flat[:-1]
            seq = flat[keep].cpu().tolist()
        elif backend.is_tensor(ids):
            seq = ids.reshape(-1).cpu().tolist()
        else:
            seq = list(ids)

        if self.dedup:
            kept: list[int] = []
            prev = self.blank
            for tok in seq:
                tok = int(tok)
                if tok != self.blank and tok != prev:
                    kept.append(tok)
                prev = tok
            return self.push_ids(kept)
        new: list[int] = []
        last = self._last
        for tok in seq:
            if tok != last and tok != self.blank:
                new.append(tok)
            last = tok
        self._last = last
        self.tokens.extend(new)
        return new


# TODO (yiakwy) : add beamSearch kernel support on GPU side
class CtcPrefixBeamDecoder:
    """Decodes chunk-by-chunk like CtcGreedyDecoder with prefix beam search."""

    def __init__(
        self, blank: int = 0, beam_size: int = 8, nbest: int = 1, dedup: bool = False
    ):
        self.blank = int(blank)
        self.beam_size = int(beam_size)
        self.nbest = int(nbest)

        self.dedup = bool(dedup)
        self.tokens: list[int] = []
        self._last: int = int(blank)
        self.nbest_tokens: list[list[int]] = []

    # TODO (yiakwy) : move to gpu
    def push_logp(self, logp) -> list[int]:
        lp = logp.detach().cpu().numpy() if hasattr(logp, "cpu") else np.asarray(logp)
        nbest = prefix_beam_search(
            lp, beam_size=self.beam_size, allow_doubled=not self.dedup
        )
        ids = nbest[0][0] if nbest else []
        new: list[int] = []

        if self.dedup:
            last = self._last
            for t in ids:
                t = int(t)
                if t != last and t != self.blank:
                    new.append(t)
                last = t
            self._last = last
        else:
            for i, t in enumerate(ids):
                t = int(t)
                if t == self.blank:
                    continue
                if i == 0 and t == self._last:
                    continue
                new.append(t)
            if new:
                self._last = new[-1]
        self.tokens.extend(new)
        self.nbest_tokens.append([int(u) for u in ids])
        return new
