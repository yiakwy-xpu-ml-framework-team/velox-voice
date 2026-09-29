"""CTC prefix beam search

Semantics (classic prefix beam):

  seq-argmax takes the argmax greedly at each frame to get the most likely frame-level label
  sequence, then remove duplicates and blanks.

  While CTC prefix beam search method performs beam search upon partially decoded sequences, and
  it keeps multiple candidate prefixes.

  * states to track
    - pb: log P ending in blank,
    - pnb: log P ending in non-blank

  * per frame:
    - emit blank onto each prefix to get pb
    - emit each non-blank `u` (symbol) :
      - append to prefix
      - repeat folding
  * prune beams by total `log(e^pb + e^pnb)`.
"""

from __future__ import annotations

import numpy as np

LOG_ZERO = float("-inf")


def _lg(a: float, b: float) -> float:
    if a == LOG_ZERO:
        return b
    if b == LOG_ZERO:
        return a
    hi = max(a, b)
    return hi + np.log1p(np.exp(-abs(a - b)))


def prefix_beam_search(
    logp: np.ndarray,
    beam_size: int = 8,
    blank: int = 0,
    allow_doubled: bool = True,
) -> list[tuple[list[int], float]]:
    T = logp.shape[0]
    cur = {tuple(): (0.0, LOG_ZERO)}
    for t in range(T):
        scores = logp[t]
        nxt: dict[tuple[int, ...], list[float]] = {}

        def _slot(key: tuple[int, ...]) -> list[float]:
            e = nxt.get(key)
            if e is None:
                e = [LOG_ZERO, LOG_ZERO]
                nxt[key] = e
            return e

        for prefix, (pb, pnb) in cur.items():
            # blank extension keeps the prefix
            e = _slot(prefix)
            e[0] = _lg(e[0], _lg(pb, pnb) + float(scores[blank]))
            idx = np.argsort(-scores)[:8]
            for u in idx:
                u = int(u)
                if u == blank:
                    continue

                su = float(scores[u])

                if prefix and prefix[-1] == u:
                    # non-blank stream continuation merges onto `prefix`
                    _slot(prefix)[1] = _lg(_slot(prefix)[1], pnb + su)
                    if allow_doubled:
                        # blank-ended stream re-emits u: NEW doubled symbol
                        _slot(prefix + (u,))[1] = _lg(_slot(prefix + (u,))[1], pb + su)
                    else:
                        # legacy: fold the re-emission into the same prefix
                        _slot(prefix)[1] = _lg(_slot(prefix)[1], pb + su)
                else:
                    _slot(prefix + (u,))[1] = _lg(
                        _slot(prefix + (u,))[1], _lg(pb, pnb) + su
                    )
        # prune
        if not nxt:
            break
        ordered = sorted(
            nxt.items(), key=lambda kv: _lg(kv[1][0], kv[1][1]), reverse=True
        )
        cur = {k: (v[0], v[1]) for k, v in ordered[:beam_size]}
    return [
        (list(p), _lg(pb, pnb))
        for p, (pb, pnb) in sorted(
            cur.items(), key=lambda kv: _lg(kv[1][0], kv[1][1]), reverse=True
        )
    ]


# ----- Integration to Harness -----


def test_prefix_beam():
    lp2 = np.full((4, 5), -10.0, dtype=np.float64)
    for t, i in enumerate([1, 0, 1, 2]):
        lp2[t, i] = 10.0
    ids, _ = prefix_beam_search(lp2, beam_size=3)[0]

    assert ids == [1, 1, 2], f"doubled symbol lost: {ids}"

    lp3 = np.full((3, 5), -10.0, dtype=np.float64)
    for t, i in enumerate([1, 1, 2]):
        lp3[t, i] = 10.0
    ids, _ = prefix_beam_search(lp3, beam_size=3)[0]

    assert ids == [1, 2], f"frame merge broken: {ids}"

    rng = np.random.default_rng(0)
    T, V = 20, 5
    logits = rng.standard_normal((T, V)).astype(np.float32)
    logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))

    out = prefix_beam_search(logp, beam_size=3)

    assert len(out) >= 1

    ids, p = out[0]
    # should be better than greedy on random sym topology
    g = logp.argmax(-1)
    gt = [i for i in g if i != 0]
    print("greedy :", gt)
    print("best   :", ids, p)


if __name__ == "__main__":
    test_prefix_beam()
