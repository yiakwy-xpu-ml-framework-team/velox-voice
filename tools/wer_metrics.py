"""ASR accuracy metrics: WER / CER / cpWER.

Normalization follows the common Whisper/VibeVoice-style contract:
  - NFKC, lowercase, strip punctuation (ASCII + CJK), collapse whitespace
  - Chinese refs are scored per-character (CER); WER on Chinese datasets is
    computed over per-character "words" so WER == CER there (standard practice)
cpWER (concatenated permutation-PER) scores speaker-attributed transcripts:
the optimal speaker->speaker assignment minimizes total edit distance. With
ASR-only output (no diarization) each scored segment is single-speaker, so
cpWER reduces to the utterance WER and is labeled cpWER(oracle).
"""

from __future__ import annotations

import itertools
import re
import unicodedata

import jiwer

_PUNCT = (
    "!#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "。，、！？；：”“‘’（）《》〈〉【】〔〕・…—～·«»“”‘’"
)


def normalize(text: str) -> str:
    """ASR-eval text normalization."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    # drop bracketed noise/filler markers used by AISHELL-4 (<%>, <_>, [laughter])
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    for ch in _PUNCT:
        text = text.replace(ch, " ")
    return " ".join(text.split())


def _chars(text: str) -> list[str]:
    """Token list for Chinese-style scoring: one char = one word."""
    return [c for c in text if not c.isspace()]


def _words(text: str) -> list[str]:
    return text.split()


def cer(ref: str, hyp: str) -> float:
    r, h = _chars(normalize(ref)), _chars(normalize(hyp))
    if not r:
        return 0.0 if not h else 1.0
    return jiwer.wer(" ".join(r), " ".join(h))


def wer(ref: str, hyp: str) -> float:
    """Word-level WER; for Chinese refs (no spaces) this equals CER because
    normalization leaves a single token."""
    r, h = normalize(ref), normalize(hyp)
    if not _chars(r):
        return 0.0 if not _chars(h) else 1.0
    if not re.search(r"[\u4e00-\u9fff]", r):
        return jiwer.wer(r, h)
    # Chinese: score per char
    return cer(ref, hyp)


def _tok_wer(text: str) -> list[str]:
    """Token list for WER: words for English, per-char for CJK (WER == CER)."""
    t = normalize(text)
    if re.search(r"[\u4e00-\u9fff]", t):
        return _chars(t)
    return _words(t)


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def cpwer(
    refs_by_spk: dict[str, str], hyps_by_spk: dict[str, str]
) -> tuple[float, float]:
    """Concatenated permutation WER/CER.

    refs_by_spk / hyps_by_spk: speaker -> concatenated transcript.
    Returns (cp_wer, cp_cer) minimized over speaker assignments. Speaker sets
    are padded to equal size with empty transcripts. Token granularity is
    decided per speaker-pair by the REF language and applied to BOTH sides
    (mixing word-level English with char-level CJK in one edit distance
    inflates the score nonsensically).
    """
    r_spk = list(refs_by_spk)
    h_spk = list(hyps_by_spk)
    n = max(len(r_spk), len(h_spk), 1)
    r_spk = r_spk + [""] * (n - len(r_spk))
    h_spk = h_spk + [""] * (n - len(h_spk))

    best_w, best_c = float("inf"), float("inf")
    # cap permutations; fall back to identity for large sets
    perms = itertools.permutations(range(n)) if n <= 6 else [tuple(range(n))]
    for perm in perms:
        w = c = 0
        w_len = c_len = 0
        for ri, hi in enumerate(perm):
            r, h = refs_by_spk.get(r_spk[ri], ""), hyps_by_spk.get(h_spk[hi], "")
            rt, ht = normalize(r), normalize(h)
            if re.search(r"[\u4e00-\u9fff]", rt):
                rw, hw = _chars(rt), _chars(ht)
            else:
                rw, hw = _words(rt), _words(ht)
            w += _edit_distance(rw, hw)
            w_len += len(rw)
            rc, hc = _chars(normalize(r)), _chars(normalize(h))
            c += _edit_distance(rc, hc)
            c_len += len(rc)
        best_w = min(best_w, w / max(w_len, 1))
        best_c = min(best_c, c / max(c_len, 1))
    return best_w, best_c
