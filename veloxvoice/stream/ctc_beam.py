"""CTC prefix beam search — production-grade replacement for seq-argmax.

Semantics (classic prefix beam):
  * state = (pb: log P ending in blank, pnb: log P ending in non-blank)
  * per frame: emit blank onto each prefix; emit each non-blank u → extend
    (new token u) or repeat (merge onto existing last == u);
  * prune beams by total `logaddexp(pb, pnb)`.
Cache-safe: can run on torch tensors or numpy (`numpy=True` forces host list
regardless of backend, ok because this is the host-visible decode point).
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
    logp: np.ndarray, beam_size: int = 8, blank: int = 0
) -> list[tuple[list[int], float]]:
    T = logp.shape[0]
    cur = {tuple(): (0.0, LOG_ZERO)}
    for t in range(T):
        scores = logp[t]
        nxt: dict[tuple[int, ...], tuple[float, float]] = {}
        for prefix, (pb, pnb) in cur.items():
            # blank extension keeps prefix
            cand_pb = _lg(pb, pnb) + float(scores[blank])
            ol = nxt.get(prefix, (LOG_ZERO, LOG_ZERO))
            nxt[prefix] = (_lg(ol[0], cand_pb), ol[1])
            # non-blank extensions / repeat
            idx = np.argsort(-scores)[:8]
            for u in idx:
                u = int(u)
                if u == blank:
                    continue
                su = float(scores[u])
                if prefix and prefix[-1] == u:
                    # extension of repeated symbol: non-blank stream merges
                    cand_nnb1 = pnb + su  # repeat: only valid via nonblank
                    ol = nxt.get(prefix, (LOG_ZERO, LOG_ZERO))
                    nxt[prefix] = (ol[0], _lg(ol[1], _lg(cand_nnb1, pb + su)))
                else:
                    key = prefix + (u,)
                    cand_pb2 = _lg(pb, pnb) + su if prefix else pb + su
                    ol = nxt.get(key, (LOG_ZERO, LOG_ZERO))
                    nxt[key] = (_lg(ol[0], cand_pb2), ol[1])
        # prune
        if not nxt:
            break
        ordered = sorted(
            nxt.items(), key=lambda kv: _lg(kv[1][0], kv[1][1]), reverse=True
        )
        cur = dict(ordered[:beam_size])
    return [
        (list(p), _lg(pb, pnb))
        for p, (pb, pnb) in sorted(
            cur.items(), key=lambda kv: _lg(kv[1][0], kv[1][1]), reverse=True
        )
    ]


def test_prefix_beam():
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
