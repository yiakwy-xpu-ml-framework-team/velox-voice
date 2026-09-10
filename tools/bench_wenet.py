"""Single Model Test : wenet torch module transcribing test (bypass veloxvoice.api).
Enable velox JIT kernel (fused_layernorm + silu_glu) with --use-jit / --no-jit.

Usage:
  python tools/bench_wenet.py \
    --model-dir <bundle with final.pt> --audio <audio with supported suffix> [--seconds N] [--use-jit | --no-jit] [--iters N]
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


# TODO (yiakwy) : replaced with cer
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


def _max_mel_frames(model, cfg):
    """Long-audio cap, see veloxvoice/api.py"""
    max_len = max(getattr(model, "pos_max_len", 5000), 1000)
    return min(15000, (max_len - 200) * cfg.subsampling)


def _spans_for(feats, max_mel, pcm):
    """[(a, b)] slices of feats; long audio split at low-energy points (VAD)."""
    n = feats.shape[0]
    if n <= max_mel:
        return [(0, n)]

    # from veloxvoice.api import _low_energy_spans
    def _low_energy_spans(n_frames: int, max_frames: int, db) -> list[tuple[int, int]]:
        """Split [0, n_frames) into spans of <= max_frames, cutting each span at the
        lowest-energy point found in a backward search window (<= 30 s) from the
        nominal cut, so segment edges fall inside pauses rather than mid-word."""
        import numpy as np

        spans = []
        a = 0
        while n_frames - a > max_frames:
            nominal = a + max_frames
            lo = max(a + max_frames // 4, nominal - 3000)  # search back <= 30 s
            hi = min(nominal, len(db) - 1)
            cut = lo + int(np.argmin(db[lo:hi])) if hi > lo else nominal
            cut = max(cut, a + 1)
            spans.append((a, cut))
            a = cut
        spans.append((a, n_frames))
        return spans

    from veloxvoice.audio.vad_energy import frame_energy_db

    db = frame_energy_db(
        torch.as_tensor(pcm, dtype=torch.float32).reshape(-1).cpu()
    ).numpy()
    return _low_energy_spans(n, max_mel, db)


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

        # NOTE (yiakwy) : full-context encode
        max_mel = _max_mel_frames(m_on, load_config(args.model_dir))
        if feats0.shape[0] > max_mel:
            print(
                f"[verify_correctness] feats {feats0.shape[0]} -> {max_mel} "
                f"(positional window; full audio is segmented in the benchmark)"
            )
            feats0 = feats0[:max_mel]

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
    else:
        raise Exception("No valid audio file")


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
    ap.add_argument(
        "--precision",
        choices=["fp32", "bf16", "mxfp4"],
        default="bf16",
        help="frontend precision mode (default: bf16)",
    )
    args = ap.parse_args()

    cfg = load_config(args.model_dir)
    text_tok = TextTokenizer(os.path.join(args.model_dir, "units.txt"))
    fe = TorchGpuFrontend(
        FrontendConfig(
            sample_rate=16000, num_mel_bins=cfg.input_dim, precision=args.precision
        ),
        args.device,
    )

    verify_correctness(fe, args)

    m = WenetConformerASR(args.model_dir, device=args.device)
    m.set_jit(args.use_jit)
    m.set_precision(args.precision)

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

    print(
        f"benchmark [jit={'ON' if args.use_jit else 'OFF'}] precision={args.precision} iters={args.iters}..."
    )
    tot_e = tot_w = 0.0
    elapse_enc_list, enc_rtf_list, tot_wall = [], [], 0.0
    for name, path, ref in jobs:
        start = time.perf_counter()

        # read audio
        pcm = read_audio(path)
        if args.seconds:
            pcm = pcm[: int(args.seconds * 16000)]
        audio_dur = len(pcm) / 16000.0

        # extract features
        feats = torch.cat([fe.accept(pcm), fe.flush()], dim=0)

        # long audio: segment at low-energy points (pos_pe caps at pos_max_len)
        max_mel = _max_mel_frames(m, cfg)
        spans = _spans_for(feats, max_mel, pcm)

        start_enc = time.perf_counter()

        # enc = m.encode_utterance(feats[None])
        enc = torch.cat([m.encode_utterance(feats[a:b][None]) for a, b in spans], dim=1)

        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

        enc_elapsed = time.perf_counter() - start_enc

        start_dec = time.perf_counter()

        logp = m.ctc_logp(enc)[0]

        dec = CtcGreedyDecoder()
        dec.push_logp(logp)
        txt = text_tok.ids_to_text(dec.tokens).strip().lower()
        elapsed_dec_ctc = time.perf_counter() - start_dec

        late_wall = time.perf_counter() - start

        wer_txt = ""
        if ref:
            e, n = wer(ref, txt)
            tot_e += e
            tot_w += n
            wer_txt = f" err={e}" + (" (EXACT)" if e == 0 else "")

        enc_rtf = enc_elapsed / audio_dur
        total_rtf = late_wall / audio_dur

        elapse_enc_list.append(enc_elapsed)
        enc_rtf_list.append(enc_rtf)

        tot_wall += late_wall

        seg_txt = f" segs={len(spans)}" if len(spans) > 1 else ""
        print(
            f"[{name}] {audio_dur:6.2f}s{seg_txt} enc={enc_elapsed *1e3:8.2f}ms, late_wall={late_wall *1e3:8.2f} "
            f"total rtf={total_rtf:.4f}, encoder rtf={enc_rtf:.4f} (+ctc={elapsed_dec_ctc *1e3:4.1f}ms){wer_txt}"
        )
        print("  hyp:", txt)
        if ref:
            print("  ref:", ref)

    if tot_w:
        print(f"\nTOTAL WER = {int(tot_e)}/{int(tot_w)} = {tot_e / tot_w * 100:.2f}%")
    print(f"avg enc-rtf = {sum(enc_rtf_list)/max(len(enc_rtf_list),1):.4f}")


if __name__ == "__main__":
    main()
