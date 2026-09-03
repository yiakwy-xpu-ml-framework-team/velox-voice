"""Single Model Test: wenet torch module transcribing test (bypass veloxvoice.api).
Enable velox JIT kernel (fused_layernorm + silu_glu) with --use-jit / --no-jit.

Usage:
  python tools/bench_mirror.py \
    --model-dir <含 final.pt 的 bundle 目录> (--use-jit | --no-jit) [--iters N]
"""

from __future__ import annotations

import argparse
import os
import time
import wave

import numpy as np
import torch

from veloxvoice.audio import FrontendConfig
from veloxvoice.audio.frontend_torch import TorchGpuFrontend
from veloxvoice.audio.text_tokenizer import TextTokenizer
from veloxvoice.models.wenet.config import load_config
from veloxvoice.models.wenet.torch_conformer import WenetConformerASR
from veloxvoice.stream.ctc import CtcGreedyDecoder

_AUDIO_EXTS = (
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".mp4",
    ".mkv",
    ".wma",
    ".aac",
    ".opus",
)


def read_audio(p):
    """wav/mp3/flac/m4a/... audio -> 16kHz mono audio resampled by ffmpeg."""
    import subprocess

    try:
        with wave.open(p, "rb") as w:
            sr, ch = w.getframerate(), w.getnchannels()
            if sr == 16000 and ch == 1:
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                return pcm.astype(np.float32) / 32768.0
    except Exception:
        pass
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            p,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "-",
        ],
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def read_trans(dirpath):
    refs = {}
    p = os.path.join(dirpath, "trans.txt")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            q = line.strip().split(" ", 1)
            if len(q) == 2:
                refs[q[0]] = q[1].lower()
    return refs


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    n = len(r)
    d = [[0] * (len(h) + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (r[i - 1] != h[j - 1]),
            )
    return d[n][len(h)], n


def verify_correctness(fe, args, suffix=".wav"):
    wav0 = args.audio or (
        os.path.join(args.audio_dir, sorted(read_trans(args.audio_dir))[0] + suffix)
        if args.audio_dir and read_trans(args.audio_dir)
        else None
    )

    if wav0 is not None:

        # read audio
        pcm0 = read_audio(wav0)
        if args.seconds:
            pcm0 = pcm0[: int(args.seconds * 16000)]
        feats0 = torch.cat([fe.accept(pcm0), fe.flush()], dim=0)

        # NOTE (yiakwy) w/ JIT kernel
        m_on = WenetConformerASR(args.model_dir, device=args.device)
        m_on.set_jit(True)
        d_on = CtcGreedyDecoder()
        d_on.push_logp(m_on.ctc_logp(m_on.encode_utterance(feats0[None]))[0])

        # NOTE (yiakwy) w/o JIT kernel
        m_off = WenetConformerASR(args.model_dir, device=args.device)
        m_off.set_jit(False)
        d_off = CtcGreedyDecoder()
        d_off.push_logp(m_off.ctc_logp(m_off.encode_utterance(feats0[None]))[0])

        same_tokens = d_on.tokens == d_off.tokens
        mae = float(
            (m_on.encode_utterance(feats0[None]) - m_off.encode_utterance(feats0[None]))
            .abs()
            .max()
        )
        print(
            f"[verify_correctness] JIT-ON ≡ JIT-OFF tokens_same={same_tokens}  max|∆enc|={mae:.5f} "
            f"n={len(d_on.tokens)}"
        )
        assert same_tokens, "velox JIT kernel deviated."
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir", required=True, help="model (final.pt + train.yaml + units.txt)"
    )
    ap.add_argument("--audio-dir", default=None)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=5, help="defaults to 5")
    ap.add_argument(
        "--use-jit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="velox JIT kernel",
    )
    args = ap.parse_args()

    cfg = load_config(args.model_dir)
    text_tok = TextTokenizer(os.path.join(args.model_dir, "units.txt"))
    fe = TorchGpuFrontend(
        FrontendConfig(sample_rate=16000, num_mel_bins=cfg.input_dim), args.device
    )

    verify_correctness(fe, args)

    m = WenetConformerASR(args.model_dir, device=args.device)
    m.set_jit(args.use_jit)

    # NOTE (yiakwy) : prepare audio and transcripts
    jobs = []
    refs = read_trans(args.audio_dir) if args.audio_dir else {}
    if args.audio:
        base = os.path.splitext(os.path.basename(args.audio))[0]
        jobs.append((base, args.audio, refs.get(base)))
    else:
        if not args.audio_dir:
            jobs = []
        else:
            files = []
            for f in sorted(os.listdir(args.audio_dir)):
                if os.path.splitext(f)[1].lower() in _AUDIO_EXTS:
                    files.append(f)
            for f in files:
                base = os.path.splitext(f)[0]
                jobs.append((base, os.path.join(args.audio_dir, f), refs.get(base)))

    print(f"benchmark [jit={'ON' if args.use_jit else 'OFF'}] iters={args.iters}...")
    tot_e = tot_w = 0.0
    elapse_enc_list, enc_rtf_list, tot_wall = [], [], 0.0
    for name, path, ref in jobs:
        start = time.perf_counter()

        # read audio
        pcm = read_audio(path)
        if args.seconds:
            pcm = pcm[: int(args.seconds * 16000)]
        t_audio = len(pcm) / 16000.0

        # extract features
        feats = torch.cat([fe.accept(pcm), fe.flush()], dim=0)

        start_enc = time.perf_counter()

        enc = m.encode_utterance(feats[None])

        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

        enc_elapsed = time.perf_counter() - start_enc

        start_dec = time.perf_counter()

        logp = m.ctc_logp(enc)[0]

        dec = CtcGreedyDecoder()
        dec.push_logp(logp)
        elapsed_dec_1 = time.perf_counter() - start_dec

        txt = text_tok.ids_to_text(dec.tokens).strip().lower()
        elapsed_dec_2 = time.perf_counter() - start_dec

        late_wall = time.perf_counter() - start

        wer_txt = ""
        if ref:
            e, n = wer(ref, txt)
            tot_e += e
            tot_w += n
            wer_txt = f" err={e}" + (" (EXACT)" if e == 0 else "")

        enc_rtf = enc_elapsed / t_audio
        total_rtf = late_wall / t_audio

        elapse_enc_list.append(enc_elapsed)
        enc_rtf_list.append(enc_rtf)

        tot_wall += late_wall

        print(
            f"[{name}] {t_audio:6.2f}s enc={enc_elapsed *1e3:8.2f}ms, late_wall={late_wall *1e3:8.2f} "
            f"total rtf={total_rtf:.4f}, encoder rtf={enc_rtf:.4f} (+ctc={(elapsed_dec_1 + elapsed_dec_2)*1e3:4.1f}ms){wer_txt}"
        )
        print("  hyp:", txt)
        if ref:
            print("  ref:", ref)

    if tot_w:
        print(f"\nTOTAL WER = {int(tot_e)}/{int(tot_w)} = {tot_e / tot_w * 100:.2f}%")
    print(f"avg enc-rtf = {sum(enc_rtf_list)/max(len(enc_rtf_list),1):.4f}")


if __name__ == "__main__":
    main()
