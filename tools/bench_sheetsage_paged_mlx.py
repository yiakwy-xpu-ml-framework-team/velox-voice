"""Paged-window MLX benchmark (batched encode + batched decode).

This is the Apple-silicon analogue of the CUDA paged voice path:
non-overlapping (or sliding) windows are encoded as one batch and decoded as
one batch, so the 6-layer decoder and LM head amortize weight reads across
windows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np

from veloxvoice.kernels.ops import metal_ops
from veloxvoice.models.sheetsage.v2.mlx_model import Sheetsage2MLX
from veloxvoice.models.sheetsage.v2.mlx_batch_decoder import MLXBatchedDecoder
from veloxvoice.models.sheetsage.v2.audio_loader import load_audio, SAMPLE_RATE
from veloxvoice.audio.music_tokenizer import SheetSage2Tokenizer
from veloxvoice.stream.grammar_ar_decode import FULL_TASK_PROMPTS, PromptGrammarState
from benchmark.kernels.sheetsage2_common import verify_correctness


def build_prefix(tok):
    prompts = tok.normalize_prompts(FULL_TASK_PROMPTS)
    prefix = tok.prompt_prefix(prompts)
    if prefix[-1] == tok.eos_token:
        prefix = prefix[:-1]
    return prefix


def init_states(tok, prefix, n):
    states = [PromptGrammarState(tok) for _ in range(n)]
    oi = prefix.index(tok.out_token)
    for s in states:
        for t in prefix[oi + 1:]:
            s.update(t)
    return states


def allowed_matrix(states, tok, active):
    rows = []
    for s, a in zip(states, active):
        rows.append(s.allowed("cpu").numpy() if a
                    else np.ones(tok.n_tokens, dtype=bool))
    return mx.array(np.stack(rows))


def decode_batch(model, mem, prefix, tok, window_seconds, max_tokens):
    B = mem.shape[0]
    dec = MLXBatchedDecoder(model, max_seq=max_tokens + len(prefix) + 8,
                            t_mem=mem.shape[1], batch=B)
    logits = dec.prefill(prefix)
    mx.eval(logits)
    dec.set_memory(mem)

    states = init_states(tok, prefix, B)
    active = [True] * B
    outs = [list(prefix) for _ in range(B)]
    allowed = allowed_matrix(states, tok, active)
    first = mx.argmax(mx.where(allowed, logits,
                               mx.array(float("-inf"), dtype=logits.dtype)),
                      axis=-1).tolist()
    for b in range(B):
        outs[b].append(first[b])
        if states[b].update(first[b]):
            active[b] = False

    tokens_in = [first[b] if active[b] else tok.eos_token for b in range(B)]
    steps = 0
    for _ in range(max_tokens):
        if not any(active):
            break
        allowed = allowed_matrix(states, tok, active)
        nxt = dec.step(tokens_in, allowed)
        steps += 1
        for b in range(B):
            if not active[b]:
                tokens_in[b] = tok.eos_token
                continue
            t = int(nxt[b])
            outs[b].append(t)
            done = states[b].update(t)
            if (tok.time_token_start <= t < tok.time_token_end
                    and tok.token_to_time_id(t) / tok.time_hz >= window_seconds):
                outs[b].append(tok.eos_token)
                done = True
            if done:
                active[b] = False
                tokens_in[b] = tok.eos_token
            else:
                tokens_in[b] = t
    mx.eval(dec.k_self, dec.v_self, dec.k_cross, dec.v_cross)
    return outs, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="test_data/music/bohemian_rhapsody_queen.mp3")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--window-seconds", type=float, default=20.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--no-jit", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    metal_ops.set_jit(not args.no_jit)
    model = Sheetsage2MLX.load("data/music/SheetSage2",
                               "data/music/MERT-v2-FullSong")
    mx.eval(model.enc_proj_w)
    tok = SheetSage2Tokenizer(model.cfg.input_audio_length, model.cfg.time_hz,
                              model.cfg.tokenizer_schema_version,
                              expected_fingerprint=model.cfg.tokenizer_fingerprint)
    prefix = build_prefix(tok)
    wav = load_audio(args.audio)[: int(args.seconds * SAMPLE_RATE)]
    n_win = max(1, int(np.ceil(len(wav) / (args.window_seconds * SAMPLE_RATE))))
    win = int(args.window_seconds * SAMPLE_RATE)
    segs = []
    for i in range(n_win):
        s = wav[i * win:(i + 1) * win]
        if len(s) < win:
            s = np.pad(s, (0, win - len(s)))
        segs.append(s)
    batch = min(args.batch, n_win)

    if not args.no_verify:
        def candidate_fn(model, mem, prefix, state, tok, seconds, max_tokens):
            outs, _ = decode_batch(model, mem, prefix, tok, seconds, max_tokens)
            return outs[0]
        check = verify_correctness(model, tok, wav, seconds=20.0,
                                   candidate_fn=candidate_fn)
        if not check["ok"]:
            print("correctness FAILED", check)
            raise SystemExit(1)
        print("correctness ok")

    enc_t, dec_t, steps, outs = [], [], 0, []
    for _ in range(args.iters):
        outs, steps = [], 0
        enc_total = dec_total = 0.0
        for start in range(0, n_win, batch):
            chunk = segs[start:start + batch]
            x = mx.array(np.stack(chunk))
            t0 = time.perf_counter()
            mem = model.encode(x)
            mx.eval(mem)
            mx.clear_cache()  # MLX 0.29 reuses encoder scratch; drain it
            enc_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            o, st = decode_batch(model, mem, prefix, tok, args.window_seconds,
                                 args.max_tokens)
            dec_total += time.perf_counter() - t0
            outs.extend(o)
            steps += st
        enc_t.append(enc_total)
        dec_t.append(dec_total)

    enc = sorted(enc_t)[len(enc_t) // 2]
    dec = sorted(dec_t)[len(dec_t) // 2]
    total = enc + dec
    n_tok = sum(len(o) - len(prefix) for o in outs)
    duration = n_win * args.window_seconds
    rtf = total / duration
    print(f"windows {n_win}  batch {batch}  steps {steps}  tokens {n_tok}  "
          f"encode {enc*1e3:.1f}ms  decode {dec*1e3:.1f}ms  total {total:.3f}s")
    print(f"  RTF {rtf:.5f}  tok/s {n_tok/max(dec,1e-9):.1f}  "
          f"step {dec/max(steps,1)*1e3:.3f}ms  "
          f"encode-RTF {enc/duration:.5f}  decode-RTF {dec/duration:.5f}")
    report = {"windows": n_win, "batch": batch, "steps": steps,
              "tokens": n_tok, "encode_s": enc, "decode_s": dec,
              "total_s": total, "rtf": rtf,
              "tok_per_s": n_tok / max(dec, 1e-9),
              "step_ms": dec / max(steps, 1) * 1e3}
    if args.json:
        json.dump(report, open(args.json, "w"), indent=2)
        print(f"  report {args.json}")


if __name__ == "__main__":
    main()
