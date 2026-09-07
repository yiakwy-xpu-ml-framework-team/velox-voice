"""Streaming CTC decoding.

Greedy (`CtcGreedyDecoder`) collapses repeats/blanks across chunks. The quality path
(`CtcPrefixBeamDecoder`) runs prefix beam search over the same stream for the
accuracy-critical production lane.
"""

from __future__ import annotations

import numpy as np

from .ctc_beam import prefix_beam_search


class CtcGreedyDecoder:
    def __init__(self, blank: int = 0):
        self.blank = blank
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
        if backend.is_tensor(ids):
            if ids.ndim == 1:
                prev = backend.cat([ids.new_zeros(1), ids[:-1]])
                keep = ((ids != self.blank) & (ids != prev)).nonzero().squeeze(-1)
                kept = ids[keep]
                ids = kept.cpu().tolist()
            else:
                ids = ids.cpu().tolist()
        return self.push_ids(ids)


# TODO (yiakwy) : add beamSearch kernel support on GPU side
class CtcPrefixBeamDecoder:
    """Decodes chunk-by-chunk like CtcGreedyDecoder, but with prefix beam search
    inside each chunk (n-best path by default; same push/state semantics).
    """

    def __init__(self, blank: int = 0, beam_size: int = 8, nbest: int = 1):
        self.blank = int(blank)
        self.beam_size = int(beam_size)
        self.nbest = int(nbest)
        self.tokens: list[int] = []
        self._last: int = int(blank)
        self.nbest_tokens: list[list[int]] = []

    # TODO (yiakwy) : move to gpu
    def push_logp(self, logp) -> list[int]:
        lp = logp.detach().cpu().numpy() if hasattr(logp, "cpu") else np.asarray(logp)
        nbest = prefix_beam_search(lp, beam_size=self.beam_size)
        ids = nbest[0][0] if nbest else []
        new: list[int] = []
        last = self._last
        for t in ids:
            t = int(t)
            if t != last and t != self.blank:
                new.append(t)
            last = t
        self._last = last
        self.tokens.extend(new)
        self.nbest_tokens.append([int(u) for u in ids])
        return new
