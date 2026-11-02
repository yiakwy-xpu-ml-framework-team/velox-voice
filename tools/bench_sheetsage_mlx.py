"""End-to-end SheetSage2 benchmark on the MLX/Metal"""

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
from veloxvoice.models.sheetsage.v2.mlx_model import Sheetsage2MLX, MLXDecoder
from veloxvoice.models.sheetsage.v2.audio_loader import load_audio, SAMPLE_RATE
from veloxvoice.audio.music_tokenizer import SheetSage2Tokenizer
from veloxvoice.stream.grammar_ar_decode import FULL_TASK_PROMPTS, PromptGrammarState
from benchmark.kernels.sheetsage2_common import verify_correctness


def build_prefix(tokenizer):
    prompts = tokenizer.normalize_prompts(FULL_TASK_PROMPTS)
    prefix = tokenizer.prompt_prefix(prompts)
    if prefix[-1] == tokenizer.eos_token:
        prefix = prefix[:-1]
    state = PromptGrammarState(tokenizer)
    oi = prefix.index(tokenizer.out_token)
    for t in prefix[oi + 1:]:
        state.update(t)
    return prefix, state


def decode(model, mem, prefix, state, tokenizer, seconds, max_tokens):
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
                and tokenizer.token_to_time_id(token) / tokenizer.time_hz >= seconds):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="test_data/music/bohemian_rhapsody_queen.mp3")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--model-dir", default="data/music/SheetSage2")
    ap.add_argument("--base-model-dir", default="data/music/MERT-v2-FullSong")
    ap.add_argument("--no-jit", action="store_true")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--ref-json", default=None)
    ap.add_argument("--ref-total-s", type=float, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    metal_ops.set_jit(not args.no_jit)
    t0 = time.perf_counter()
    model = Sheetsage2MLX.load(args.model_dir, args.base_model_dir)
    mx.eval(model.enc_proj_w)
    load_s = time.perf_counter() - t0

    cfg = model.cfg
    tokenizer = SheetSage2Tokenizer(
        cfg.input_audio_length, cfg.time_hz, cfg.tokenizer_schema_version,
        expected_fingerprint=cfg.tokenizer_fingerprint)
    prefix, state = build_prefix(tokenizer)
    wav = load_audio(args.audio)[: int(args.seconds * SAMPLE_RATE)].reshape(1, -1)
    x = mx.array(wav.numpy())

    if not args.no_verify:
        check = verify_correctness(model, tokenizer, wav, seconds=20.0)
        if not check["ok"]:
            print("correctness FAILED", check)
            raise SystemExit(1)
        print("correctness ok")

    enc_times, dec_times, n_tokens, tokens = [], [], 0, None
    for _ in range(args.iters):
        t0 = time.perf_counter()
        mem = model.encode(x)
        mx.eval(mem)
        enc_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        tokens = decode(model, mem, prefix, state, tokenizer, args.seconds,
                        args.max_tokens)
        dec_times.append(time.perf_counter() - t0)
        n_tokens = len(tokens) - len(prefix)

    enc = sorted(enc_times)[len(enc_times) // 2]
    dec = sorted(dec_times)[len(dec_times) // 2]
    total = enc + dec
    rtf = total / args.seconds
    print(f"load {load_s:.3f}s  encode {enc*1e3:.1f}ms  decode {dec*1e3:.1f}ms  "
          f"tokens {n_tokens}  total {total:.3f}s  RTF {rtf:.4f}")
    print(f"  encoder RTF {enc/args.seconds:.4f}  decoder RTF {dec/args.seconds:.4f}  "
          f"{n_tokens/max(dec,1e-9):.1f} tok/s")

    h, t_mem = 1024, int(args.seconds * 25)
    enc_flops = 24 * (12 * h * h * t_mem + 2 * t_mem * t_mem * h + 8 * h * h * t_mem)
    enc_sol = enc_flops / 9.0e12
    dec_flops = n_tokens * (6 * (4 * 512 * 512 + 2 * 512 * 2048) + 2 * 31678 * 512)
    dec_sol = dec_flops / 9.0e12
    print(f"  SOL encode {enc_sol*1e3:.1f}ms (gap {enc/max(enc_sol,1e-9):.2f}x)  "
          f"decode {dec_sol*1e3:.1f}ms (gap {dec/max(dec_sol,1e-9):.2f}x)")

    report = {"seconds": args.seconds, "jit": not args.no_jit,
              "load_s": load_s, "encode_s": enc, "decode_s": dec,
              "tokens": n_tokens, "rtf": rtf,
              "encode_rtf": enc / args.seconds,
              "decode_rtf": dec / args.seconds,
              "tok_per_s": n_tokens / max(dec, 1e-9),
              "encode_sol_s": enc_sol, "decode_sol_s": dec_sol,
              "encode_gap": enc / max(enc_sol, 1e-9),
              "decode_gap": dec / max(dec_sol, 1e-9)}
    if args.ref_json:
        ref = json.load(open(args.ref_json))
        if isinstance(ref, list):
            n = min(len(ref), len(tokens))
            same = sum(a == b for a, b in zip(ref[:n], tokens[:n]))
            report["ref_tokens"] = len(ref)
            report["token_match"] = same / max(n, 1)
            print(f"  ref tokens {len(ref)}  match {same}/{n} "
                  f"({same/max(n,1):.2%})")
            rt = args.ref_total_s
        else:
            rt = args.ref_total_s or ref.get("total_s") or (
                ref.get("encode_s", 0) + ref.get("decode_s", 0))
        if rt:
            report["ref_total_s"] = rt
            report["speedup_vs_ref"] = rt / total
            print(f"  ref {rt:.3f}s  speedup {rt/total:.2f}x")
    if args.json:
        json.dump(report, open(args.json, "w"), indent=2)
        print(f"  report {args.json}")
    print("tokens", tokens[:20], "..." if tokens and len(tokens) > 20 else "")


if __name__ == "__main__":
    main()
