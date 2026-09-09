"""Single MOdel Test : wenet torch module (WenetConformerASR) transcribing test (bypass veloxvoice.api).
Enable velox JIT kernel (fused_layernorm + silu_glu) with --use-jit / --no-jit.
Enable multi-process audio encoding under different GPU streams

usage:
  python tools/bench_wenet_multi_worker.py \
    --model-dir <bundle with final.pt file> --audio <audio with supported suffix> [--seconds N] [--use-jit | --no-jit]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

# NOTE (yiakwy) : add safty guard in mlx platform
import torch

from veloxvoice.audio import FrontendConfig
from veloxvoice.audio.frontend_torch import TorchGpuFrontend
from veloxvoice.audio.text_tokenizer import TextTokenizer
from veloxvoice.audio.vad_energy import frame_energy_db
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
    """wav/mp3/flac/... audio -> 16kHz mono audio resampled by ffmpeg."""
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


def _low_energy_spans(n_frames: int, max_frames: int, db) -> list[tuple[int, int]]:
    spans = []
    a = 0
    while n_frames - a > max_frames:
        nominal = a + max_frames

        lo = max(a + max_frames // 4, nominal - 3000)
        hi = min(nominal, len(db) - 1)

        cut = lo + int(np.argmin(db[lo:hi])) if hi > lo else nominal
        cut = max(cut, a + 1)

        spans.append((a, cut))
        a = cut

    spans.append((a, n_frames))
    return spans


def transcribe(
    m,
    pcm,
    cfg,
    text_tok,
    jit_on,
    device,
    max_mel_arg: int | None = None,
    fp16_on: bool = False,
    max_calls: int = max(int(os.environ.get("WENET_INPROC_MAX_CALLS", "9999")), 3),
):
    t0 = time.perf_counter()

    fe0 = TorchGpuFrontend(
        FrontendConfig(sample_rate=16000, num_mel_bins=cfg.input_dim), device
    )
    feats = torch.cat([fe0.accept(pcm), fe0.flush()], dim=0)

    t_fe = time.perf_counter() - t0

    t1 = time.perf_counter()
    max_len = max(getattr(m, "pos_max_len", 5000), 1000)
    hard_cap = (max_len - 200) * cfg.subsampling
    max_mel = args_max_mel(max_mel_arg, hard_cap)
    if feats.shape[0] <= max_mel:
        spans = [(0, feats.shape[0])]
    else:
        db = frame_energy_db(
            torch.as_tensor(pcm, dtype=torch.float32).reshape(-1).cpu()
        ).numpy()
        spans = _low_energy_spans(feats.shape[0], max_mel, db)
    t_spans = time.perf_counter() - t1

    dec = CtcGreedyDecoder()
    t_enc_ctc = t_worker = 0.0

    max_calls = int(os.environ.get("WENET_INPROC_MAX_CALLS", str(max_calls)))

    if len(spans) <= max_calls:
        t2 = time.perf_counter()

        for a, b in spans:
            enc = m.encode_utterance(feats[a:b][None])
            dec.push_logp(m.ctc_logp(enc)[0])
        ids = dec.tokens
        t_enc_ctc = time.perf_counter() - t2

        t3 = time.perf_counter()
        text = text_tok.ids_to_text(ids)
        t_decode_tokens = time.perf_counter() - t3
    else:
        pcm_np = np.asarray(pcm, dtype=np.float32).reshape(-1)
        tmpdir = tempfile.mkdtemp(prefix="transcribe_multi_worker")

        t2 = time.perf_counter()
        try:
            # TODO (yiakwy) : using numpy IPC (mmap) to share data cross processes
            pcm_path = os.path.join(tmpdir, "pcm.npy")
            np.save(pcm_path, pcm_np)

            PAR_ENV = os.environ.get("WENET_PAR_WORKERS", "serial")

            groups = (
                [spans]
                if PAR_ENV == "serial"
                else [spans[i : i + max_calls] for i in range(0, len(spans), max_calls)]
            )

            prev_ids = []
            done = 0

            # convert PAR_ENV to digits
            PAR = (
                int(PAR_ENV)
                if PAR_ENV.isdigit()
                else len(groups) if PAR_ENV == "serial" else max(2, len(groups) // 2)
            )

            gi = 0
            groups_of_wave = [groups[i : i + PAR] for i in range(0, len(groups), PAR)]
            job_store = []

            for wgi, wave in enumerate(groups_of_wave):
                procs = []
                for i, grp in enumerate(wave):

                    # TODO (yiakwy) : write to disk asynchronously
                    jpath = os.path.join(tmpdir, f"job{gi}.json")
                    opath = os.path.join(tmpdir, f"out{gi}.json")
                    lpath = os.path.join(tmpdir, f"worker{gi}.log")
                    job = {
                        "pcm": pcm_path,
                        "jit": jit_on,
                        "spans": [[int(a), int(b)] for a, b in grp],
                    }
                    with open(jpath, "w") as f:
                        json.dump(job, f)

                    print(
                        f"  [multi worker transcribe] [spawn@{time.time():.3f}] gm={gi} em-span={len(grp)}"
                    )
                    p = subprocess.Popen(
                        [
                            sys.executable,
                            os.path.abspath(__file__),
                            "--worker",
                            "--model-dir",
                            m.model_dir,
                            "--job",
                            jpath,
                            "--out",
                            opath,
                            "--job-id",
                            f"wave#{wgi}:seq#{i}",
                        ],
                        stdout=(
                            None
                            if os.environ.get("WENET_STREAM_WORKER", "0") == "1"
                            else open(lpath, "w")
                        ),
                        stderr=(
                            None
                            if os.environ.get("WENET_STREAM_WORKER", "0") == "1"
                            else subprocess.STDOUT
                        ),
                    )
                    procs.append((gi, grp, p, opath))

                    print(f"  [multi worker transcribe] [spawned] ts={time.time():.2f}")
                    gi += 1

                for pgi, grp, p, opath in procs:
                    start = time.time()
                    p.wait(timeout=900)
                    end = time.time()

                    print(
                        f"  [multi worker transcribe] spawn_wait={end - start:.2f}s rc={p.returncode}"
                    )
                    done += len(grp)

                    print(
                        f"  [multi worker transcribe] segments {done}/{len(spans)} "
                        f"(worker {done}/{len(spans)}, rc={p.returncode})",
                        flush=True,
                    )

                    if p.returncode != 0:
                        sys.exit(2)
                    job_store.append((pgi, opath))

            for pgi, opath in job_store:
                snap = json.load(open(opath))
                base = 0
                for entry in snap:
                    for t in entry["ids"][base:]:
                        if t != 0 and t != (prev_ids[-1] if prev_ids else 0):
                            prev_ids.append(t)
                    base = len(entry["ids"])
        finally:
            if not os.environ.get("WENET_KEEP_TMP"):
                if os.environ.get("WENET_KEEP_TMP", "0") != "1":
                    shutil.rmtree(tmpdir, ignore_errors=True)

        t_worker = time.perf_counter() - t2
        ids = prev_ids

        t3 = time.perf_counter()
        text = text_tok.ids_to_text(ids)
        t_decode_tokens = time.perf_counter() - t3

    return {
        "text": text.strip(),
        "idless": ids,
        "idspans": len(spans),
        "ids": ids,
        "spans": len(spans),
        "times": {
            "t_fe": t_fe,
            "t_spans": t_spans,
            "t_enc_ctc": t_enc_ctc,
            "t_worker": t_worker,
            "t_decode_tokens": t_decode_tokens,
        },
    }


def _worker_main(args):
    print(f"[worker] [prologue] ts={time.time():.3f}")

    # NOTE (yiakwy) : using IPC mmap
    job = json.load(open(args.job))
    job_id = args.job_id

    pcm = np.load(job["pcm"], mmap_mode="r")
    cfg = load_config(args.model_dir)

    start = time.perf_counter()
    m = WenetConformerASR(args.model_dir, device=args.device)
    m.set_jit(bool(job.get("jit", False)))
    print(
        f"[worker#{job_id}] load asr model, {(time.perf_counter() - start)*1e3:.0f}ms"
    )

    cfg_local = FrontendConfig(sample_rate=16000, num_mel_bins=cfg.input_dim)
    fe = TorchGpuFrontend(cfg_local, args.device)
    dec = CtcGreedyDecoder()

    results = []
    for a, b in job["spans"]:
        s = int(a) * fe.fs
        e = min(int(b) * fe.fs, pcm.shape[0])
        seg = np.ascontiguousarray(np.asarray(pcm[s:e], dtype=np.float32))
        feats = torch.cat([fe.accept(seg), fe.flush()], dim=0)

        enc_start = time.perf_counter()

        enc = m.encode_utterance(feats[None])
        logp = m.ctc_logp(enc)[0]
        print(
            f"[worker#{job_id}] encode span{[a,b]} {(time.perf_counter() - enc_start)*1e3:.0f}ms"
        )

        dec_start = time.perf_counter()
        dec.push_logp(logp)
        print(
            f"[worker#{job_id}] decode span{[a,b]} {(time.perf_counter() - dec_start)*1e3:.0f}ms"
        )

        results.append({"span": [a, b], "ids": [int(t) for t in dec.tokens]})

    dump_disk = time.perf_counter()

    with open(args.out, "w") as f:
        json.dump(results, f)

    print(f"[worker] dump data to disk { (time.perf_counter()- dump_disk)*1e3:.0f}ms")
    print(f"[worker] elapse ts={time.time():.3f}")
    print(f"[worker] decode tokens n={len(results)}")


def verify_correctness(cfg, text_tok, args, suffix=".wav"):
    wav0 = args.audio or None

    if wav0 is None and args.audio_dir:
        refs = read_trans(args.audio_dir)
        if refs:
            wav0 = os.path.join(args.audio_dir, sorted(refs)[0] + ".wav")

    if wav0 is None:
        raise Exception("No valid audio file")

    pcm0 = read_audio(wav0)
    if args.seconds:
        pcm0 = pcm0[: int(args.seconds * 16000)]

    m_on = WenetConformerASR(args.model_dir, device=args.device)
    m_on.set_jit(True)
    r_on = transcribe(
        m_on,
        pcm0,
        cfg,
        text_tok,
        jit_on=True,
        device=args.device,
        max_mel_arg=args.max_mel,
    )

    m_off = WenetConformerASR(args.model_dir, device=args.device)
    m_off.set_jit(False)
    r_off = transcribe(
        m_off,
        pcm0,
        cfg,
        text_tok,
        jit_on=False,
        device=args.device,
        max_mel_arg=args.max_mel,
    )

    same = r_on["ids"] == r_off["ids"]
    print(f"[verify: jit-on ≡ jit-off] tokens_same={same} n={len(r_on['ids'])}")
    assert same, "velox JIT kernel deviated"


def args_max_mel(arg, hard_cap):
    if arg is None:
        return min(15000, hard_cap)
    if arg > hard_cap:
        print(f"[warn] --max-mel={arg} > bundle upper limits ({hard_cap}).")
        return hard_cap
    return max(arg, hard_cap // 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--audio-dir", default=None)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument(
        "--max-mel",
        type=int,
        default=None,
        help="maximum mel frames up to 19200 (40GB / GPU budget)",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--use-jit", action=argparse.BooleanOptionalAction, default=False)

    # NOTE: fp8/mxfp8 paths removed (never validated); nvfp4/mxfp4 live in
    # `precision=` branch of DenseLinear (veloxvoice/models/wenet/torch_conformer.py).
    ap.add_argument("--fp16", action="store_true", help="use fp16 to encode audio")

    # NOTE (yiakwy) : used by transcribe functions, do not call it manually
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--job", default=None)
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--out", default=None)

    args = ap.parse_args()

    # called by transcribe function
    if args.worker:
        _worker_main(args)
        return

    cfg = load_config(args.model_dir)
    text_tok = TextTokenizer(os.path.join(args.model_dir, "units.txt"))

    verify_correctness(cfg, text_tok, args)

    m = WenetConformerASR(args.model_dir, device=args.device)
    m.set_jit(args.use_jit)

    jobs = []
    refs = read_trans(args.audio_dir) if args.audio_dir else {}
    if args.audio:
        base = os.path.splitext(os.path.basename(args.audio))[0]
        jobs.append((base, args.audio, refs.get(base)))
    elif args.audio_dir:
        for f in sorted(os.listdir(args.audio_dir)):
            if os.path.splitext(f)[1].lower() in _AUDIO_EXTS:
                base = os.path.splitext(f)[0]
                jobs.append((base, os.path.join(args.audio_dir, f), refs.get(base)))

    print(f"[jit={'ON' if args.use_jit else 'OFF'}] jobs={len(jobs)}")
    tot_e = tot_w = 0.0
    rtf_list = []
    for name, path, ref in jobs:
        pcm = read_audio(path)
        if args.seconds:
            pcm = pcm[: int(args.seconds * 16000)]
        t_audio = len(pcm) / 16000.0

        for it in range(args.iters):
            t0 = time.perf_counter()
            r = transcribe(
                m,
                pcm,
                cfg,
                text_tok,
                jit_on=args.use_jit,
                device=args.device,
                max_mel_arg=args.max_mel,
            )
            if it == 0:
                wall0 = time.perf_counter() - t0

                t = times = r["times"]

                total = sum(times.values())
                wer_txt = ""
                txt = r["text"].lower()
                if ref:
                    e, n = wer(ref, txt)
                    tot_e += e
                    tot_w += n
                    wer_txt = f" err={e}" + (" (EXACT)" if e == 0 else "")
                rtf = wall0 / t_audio
                rtf_list.append(rtf)
                print(
                    f"[{name}] {t_audio:7.1f}s spans={r['spans']} "
                    f"fe={t['t_fe']*1e3:6.1f}ms spans_tm={t['t_spans']*1e3:6.1f}ms "
                    f"enc_ctc={t['t_enc_ctc']*1e3:8.1f}ms worker={t['t_worker']*1e3:8.1f}ms "
                    f"decode={t['t_decode_tokens']*1e3:6.2f}ms  wall={wall0*1e3:9.1f}ms "
                    f"rtf={rtf:.4f}{wer_txt}"
                )
                print("  hyp:", txt)
                if ref:
                    print("  ref:", ref)
            else:
                transcribe(
                    m,
                    pcm,
                    cfg,
                    text_tok,
                    jit_on=args.use_jit,
                    device=args.device,
                    max_mel_arg=args.max_mel,
                )

    if tot_w:
        print(f"\nTOTAL WER = {int(tot_e)}/{int(tot_w)} = {tot_e / tot_w * 100:.2f}%")
    if rtf_list:
        print(f"avg rtf = {sum(rtf_list) / len(rtf_list):.4f}")


if __name__ == "__main__":
    main()
