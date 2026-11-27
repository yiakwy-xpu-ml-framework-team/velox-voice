"""Correctness helpers shared by the SheetSage2 MLX benchmarks."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from veloxvoice.models.sheetsage.v2.audio_loader import SAMPLE_RATE
from veloxvoice.models.sheetsage.v2.mlx_model import MLXDecoder
from veloxvoice.stream.grammar_ar_decode import (FULL_TASK_PROMPTS,
                                                 PromptGrammarState,
                                                 decode_generated_tokens)


def fresh_prefix_state(tokenizer):
    prompts = tokenizer.normalize_prompts(FULL_TASK_PROMPTS)
    prefix = tokenizer.prompt_prefix(prompts)
    if prefix[-1] == tokenizer.eos_token:
        prefix = prefix[:-1]
    state = PromptGrammarState(tokenizer)
    oi = prefix.index(tokenizer.out_token)
    for token in prefix[oi + 1:]:
        state.update(token)
    return prefix, state


def decode_one(model, mem, prefix, state, tokenizer, seconds,
               max_tokens=512):
    """Sequential greedy decode, mirroring the official stopping rule."""
    cfg = model.cfg
    dec = MLXDecoder(model, cfg.max_output_seq_len, mem.shape[1])
    logits = dec.prefill(prefix)
    mx.eval(logits)
    dec.set_memory(mem)

    allowed_np = state.allowed("cpu").numpy()
    lg = np.array(logits[0].astype(mx.float32))
    lg[~allowed_np] = -np.inf
    token = int(lg.argmax())
    out = list(prefix) + [token]
    pos = len(prefix)
    for _ in range(max_tokens):
        if state.update(token):
            break
        if (tokenizer.time_token_start <= token < tokenizer.time_token_end
                and tokenizer.token_to_time_id(token) / tokenizer.time_hz
                >= seconds):
            out.append(tokenizer.eos_token)
            break
        allowed = mx.array(state.allowed("cpu").numpy())
        token = dec.step(token, pos, allowed)
        pos += 1
        out.append(token)
    if out[-1] != tokenizer.eos_token:
        out.append(tokenizer.eos_token)
    mx.eval(dec.k_self, dec.v_self, dec.k_cross, dec.v_cross)
    return out


def _decoded(tokens, tokenizer):
    decoded, warning = decode_generated_tokens(tokenizer, tokens, "bench")
    if warning is not None:
        raise RuntimeError(f"fixture decode warning: {warning}")
    return decoded


def _slice(wav, seconds: float, sample_rate: int):
    n = int(seconds * sample_rate)
    seg = np.asarray(wav).reshape(-1)[:n]
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    return seg


def _event_issues(decoded, seconds: float):
    issues = []
    if not decoded["events"]:
        issues.append("no events")
    if not decoded.get("has_eos", False):
        issues.append("missing eos")
    for event in decoded["events"]:
        t = event["values"].get("timestamp")
        if t is not None and (t < -1e-3 or t > seconds + 1.0):
            issues.append(f"timestamp {t} outside [0,{seconds}]")
            break
    return issues


def verify_correctness(model, tokenizer, wav, *, seconds: float = 20.0,
                       max_tokens: int = 512, candidate_fn=None) -> dict:
    """Sanity-check the decode path used by the benchmark.

    The 20 s clip is decoded twice with the reference sequential path and,
    when a batched backend is being benchmarked, once more with that backend.
    This is not an HF parity gate: SheetSage2 has near-ties that flip on tiny
    numeric differences.  It does catch an empty/garbage stream, a broken stop
    rule, and a batched decoder that disagrees with the sequential decoder.
    """
    seg = _slice(wav, seconds, SAMPLE_RATE)
    prefix, state = fresh_prefix_state(tokenizer)
    mem = model.encode(mx.array(seg.reshape(1, -1)))
    mx.eval(mem)
    if not np.isfinite(np.asarray(mem.astype(mx.float32))).all():
        return {"ok": False, "why": "non-finite encoder memory"}

    ref_tokens = decode_one(model, mem, prefix, state, tokenizer, seconds,
                            max_tokens)
    repeat_prefix, repeat_state = fresh_prefix_state(tokenizer)
    repeat_tokens = decode_one(model, mem, repeat_prefix, repeat_state,
                               tokenizer, seconds, max_tokens)
    ref_decoded = _decoded(ref_tokens, tokenizer)
    repeat_decoded = _decoded(repeat_tokens, tokenizer)
    issues = _event_issues(ref_decoded, seconds)
    if ref_tokens != repeat_tokens:
        issues.append("sequential decode is not deterministic")

    cand_tokens = ref_tokens
    if candidate_fn is not None:
        cand_tokens = candidate_fn(model, mem, prefix, state, tokenizer,
                                   seconds, max_tokens)
        cand_decoded = _decoded(cand_tokens, tokenizer)
        if cand_decoded["events"] != ref_decoded["events"]:
            issues.append("candidate backend events differ from sequential")
    else:
        cand_decoded = ref_decoded

    return {
        "ok": not issues,
        "why": issues,
        "seconds": seconds,
        "events": len(cand_decoded["events"]),
        "ref_tokens": len(ref_tokens),
        "repeat_tokens": len(repeat_tokens),
        "cand_tokens": len(cand_tokens),
    }
