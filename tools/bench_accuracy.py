"""ASR accuracy benchmark: WER / CER / cpWER across public meeting + dialect sets.

Datasets (utterance-level scoring, single-speaker segments):
  - AMI IHM   (edinburghcstr/ami, ihm, eval)        individual headset mics, English
  - AMI SDM   (edinburghcstr/ami, sdm1, eval)       single distant mic, English
  - AISHELL-4 (AISHELL/AISHELL-4, test)             Mandarin meetings (far mic)
  - AliMeeting( playwithmino/alimeeting-eval-8k )   Mandarin far-field meetings
  - CommonVoice yue (mozilla-foundation/common_voice_17_0)  Cantonese read speech

Inspired by VibeVoice's ASR-scored evaluation harness: transcribe -> normalize
-> edit-distance metrics vs references. cpWER is the speaker-attributed
(concatenated permutation) variant; with ASR-only output every scored segment
is single-speaker, so cpWER reduces to utterance WER (labeled cpWER-oracle).

Usage:
  python tools/bench_accuracy.py --samples 50 [--datasets ami_ihm,ami_sdm,aishell4,alimeeting,cv_yue]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wer_metrics import cer, cpwer, normalize, wer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmark" / "accuracy"
HF_TOKEN = (
    open("/home/yiakwang/token.txt").read().strip()
    if os.path.exists("/home/yiakwang/token.txt")
    else None
)


# ---------------------------------------------------------------------------
# Transcription (VeloxVoice pipeline: frontend + fp16 autocast encoder + CTC)
# ---------------------------------------------------------------------------

_ASR = None
_FE = None


def _get_asr(model_dir: str):
    global _ASR, _FE
    if _ASR is None:
        from veloxvoice.audio.frontend_torch import FrontendConfig, TorchGpuFrontend
        from veloxvoice.models.wenet.torch_conformer import (
            WenetConformerASR,
            load_config,
        )

        cfg = load_config(model_dir)
        m = WenetConformerASR(model_dir, device="cuda")
        m.set_jit(True)
        m.set_fp16(True)
        _ASR = m
        _FE_CLS_FE = (FrontendConfig, TorchGpuFrontend, cfg.input_dim)
    return _ASR


def transcribe_pcm(pcm: np.ndarray, sr: int, model_dir: str) -> str:
    """pcm float32 [-1,1] at sr -> transcript."""
    import torch

    from veloxvoice.audio.frontend_torch import FrontendConfig, TorchGpuFrontend

    if sr != 16000:
        import torchaudio.functional as AF

        t = torch.from_numpy(pcm).float()
        t = AF.resample(t, sr, 16000)
        pcm = t.numpy()

    m = _get_asr(model_dir)
    fe = TorchGpuFrontend(
        FrontendConfig(sample_rate=16000, num_mel_bins=m.cfg.input_dim), "cuda"
    )
    pcm = np.ascontiguousarray(pcm.astype(np.float32))
    feats = torch.cat([fe.accept(pcm), fe.flush()], dim=0)
    if feats.shape[0] < 8:
        return ""
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        enc = m.encode_utterance(feats[None])
    logp = m.ctc_logp(enc)[0]
    ids = logp.argmax(-1)
    prev = torch.cat([ids.new_zeros(1), ids[:-1]])
    keep = (ids != 0) & (ids != prev)
    kept = ids[keep.nonzero().squeeze(-1)].cpu().tolist()
    # collapse blank-separated repeats like CtcGreedyDecoder.push_ids
    out, last = [], 0
    for t in kept:
        t = int(t)
        if t != last and t != 0:
            out.append(t)
        last = t
    from veloxvoice.audio.text_tokenizer import TextTokenizer

    tok = getattr(m, "_text_tok", None)
    if tok is None:
        tok = TextTokenizer(os.path.join(model_dir, "units.txt"))
        m._text_tok = tok
    return tok.ids_to_text(out).strip()


# ---------------------------------------------------------------------------
# Dataset loaders: yield (utt_id, pcm float32 sr, ref_text, speaker)
# ---------------------------------------------------------------------------


def _decode_audio_bytes(raw: bytes):
    """Decode compressed audio bytes without torchcodec (crashes on this box)."""
    import io

    import soundfile as sf

    data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if data.ndim > 1:
        data = data.mean(-1)
    return data.astype(np.float32), sr


def load_ami(config: str, n: int):
    from datasets import Audio, load_dataset

    ds = load_dataset(
        "edinburghcstr/ami", config, split="test", streaming=True, token=HF_TOKEN
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    for row in ds:
        data, sr = _decode_audio_bytes(row["audio"]["bytes"])
        # rows carry full-meeting audio with utterance offsets (seconds)
        b, e = row.get("begin_time"), row.get("end_time")
        if b is not None and e is not None and e > b:
            b_i, e_i = int(b * sr), int(e * sr)
            if 0 <= b_i < e_i <= data.shape[0]:
                data = data[b_i:e_i]
        if data.shape[0] < sr * 1.0:  # skip backchannel snippets (<1s)
            continue
        out.append(
            (row["audio_id"], data, sr, row["text"], row.get("speaker_id", "spk"))
        )
        if len(out) >= n:
            break
    return out


def load_aishell4(n: int):
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    session = "L_R003S01C02"
    tg_path = hf_hub_download(
        "AISHELL/AISHELL-4",
        f"test/TextGrid/{session}.TextGrid",
        repo_type="dataset",
        token=HF_TOKEN,
    )
    wav_path = hf_hub_download(
        "AISHELL/AISHELL-4",
        f"test/wav/{session}.flac",
        repo_type="dataset",
        token=HF_TOKEN,
    )

    # parse TextGrid: per-speaker IntervalTiers
    text = open(tg_path, encoding="utf-8").read()
    tiers = re_tier = __import__("re").findall(
        r'item \[\d+\]:\s*class = "IntervalTier"\s*name = "([^"]+)"(.*?)item \[',
        text,
        __import__("re").S,
    )
    utts = []
    for spk, body in tiers:
        for m in __import__("re").finditer(
            r"xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = \"([^\"]*)\"", body
        ):
            x0, x1, txt = float(m.group(1)), float(m.group(2)), m.group(3)
            if not txt.strip() or txt.startswith("<"):
                continue
            if x1 - x0 < 0.4:
                continue
            utts.append((x0, x1, txt, spk))
    info = sf.info(wav_path)
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(-1)
    out = []
    for i, (x0, x1, txt, spk) in enumerate(utts):
        if len(out) >= n:
            break
        seg = data[int(x0 * sr) : int(x1 * sr)]
        if seg.shape[0] < sr // 4:
            continue
        out.append((f"{session}_{i:03d}", seg.astype(np.float32), sr, txt, spk))
    return out


def load_alimeeting(n: int, manifest: str = "eval_n100_headset"):
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    man_path = hf_hub_download(
        "playwithmino/alimeeting-eval-8k",
        f"manifests/{manifest}.jsonl",
        repo_type="dataset",
        token=HF_TOKEN,
    )
    rows = [json.loads(l) for l in open(man_path, encoding="utf-8")]
    rows = [
        r
        for r in rows
        if r.get("num_speakers") == 1
        and r.get("overlap_ratio", 1) == 0
        and r.get("sources")
    ]
    out = []
    for r in rows:
        if len(out) >= n:
            break
        src = r["sources"][0]
        try:
            wav = hf_hub_download(
                "playwithmino/alimeeting-eval-8k",
                r["mix_path"],
                repo_type="dataset",
                token=HF_TOKEN,
            )
            data, sr = sf.read(wav, dtype="float32")
        except Exception:
            continue
        if data.ndim > 1:
            data = data.mean(-1)
        if data.shape[0] < sr // 4:
            continue
        out.append(
            (r["sample_id"], data.astype(np.float32), sr, src["text"], src["speaker"])
        )
    return out


def load_cantonese(n: int):
    """ziyou-li/cantonese_daily: Cantonese read/conversational clips + transcripts."""
    import soundfile as sf
    from huggingface_hub import hf_hub_download, list_repo_files

    meta_path = hf_hub_download(
        "ziyou-li/cantonese_daily", "metadata.jsonl", repo_type="dataset"
    )
    rows = [json.loads(l) for l in open(meta_path, encoding="utf-8-sig")]
    out = []
    for r in rows:
        if len(out) >= n:
            break
        try:
            wav = hf_hub_download(
                "ziyou-li/cantonese_daily", r["file_name"], repo_type="dataset"
            )
            data, sr = sf.read(wav, dtype="float32")
        except Exception:
            continue
        if data.ndim > 1:
            data = data.mean(-1)
        if data.shape[0] < sr // 4:
            continue
        out.append(
            (
                os.path.basename(r["file_name"]),
                data.astype(np.float32),
                sr,
                r["sentence"],
                "yue",
            )
        )
    return out


LOADERS = {
    "ami_ihm": lambda n: load_ami("ihm", n),
    "ami_sdm": lambda n: load_ami("sdm", n),
    "aishell4": load_aishell4,
    # NOTE: eval_n100 and eval_n100_headset contain the SAME clips (same
    # sample_ids and mix paths); eval_all is the representative full set.
    "alimeeting": lambda n: load_alimeeting(n, "eval_all"),
    "alimeeting_headset": lambda n: load_alimeeting(n, "eval_n100_headset"),
    "alimeeting_far": lambda n: load_alimeeting(n, "eval_n100"),
    "cantonese": load_cantonese,
}


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


def run_dataset(name, samples, model_dir, verbose=False):
    rows = LOADERS[name](samples)
    recs = []
    t_all = t_audio = 0.0
    for utt_id, pcm, sr, ref, spk in rows:
        t0 = time.perf_counter()
        try:
            hyp = transcribe_pcm(pcm, sr, model_dir)
        except Exception as e:
            if verbose:
                print(f"  [warn] {utt_id}: {e}")
            continue
        dt = time.perf_counter() - t0
        t_all += dt
        t_audio += len(pcm) / sr
        # session = meeting-level grouping for cpWER speaker attribution
        if name.startswith("ami"):
            session = utt_id.split("_")[1]  # AMI_<meeting>_H.. -> meeting id
        elif name == "alimeeting":
            session = utt_id.rsplit("_", 3)[0]
        else:
            session = utt_id.split("_")[0] if name == "aishell4" else utt_id
        recs.append(
            {"id": utt_id, "ref": ref, "hyp": hyp, "spk": spk, "session": session}
        )
        if verbose:
            print(f"  {utt_id}: ref={normalize(ref)[:40]} | hyp={normalize(hyp)[:40]}")

    wers, cers = [], []
    for r in recs:
        wers.append(wer(r["ref"], r["hyp"]))
        cers.append(cer(r["ref"], r["hyp"]))
    # cpWER: speaker-attributed (oracle) scoring per SESSION — group each
    # meeting's per-speaker refs/hyps, minimize the speaker permutation
    from collections import defaultdict

    sess_w, sess_c, n_sess = [], [], 0
    by_session = defaultdict(list)
    for r in recs:
        by_session[r["session"]].append(r)
    for sess, srecs in by_session.items():
        refs_by_spk, hyps_by_spk = {}, {}
        for r in srecs:
            refs_by_spk[r["spk"]] = refs_by_spk.get(r["spk"], "") + " " + r["ref"]
            hyps_by_spk[r["spk"]] = hyps_by_spk.get(r["spk"], "") + " " + r["hyp"]
        w, c = cpwer(refs_by_spk, hyps_by_spk)
        sess_w.append(w)
        sess_c.append(c)
        n_sess += 1
    cp = (
        float(np.mean(sess_w)) if sess_w else float("nan"),
        float(np.mean(sess_c)) if sess_c else float("nan"),
    )
    stats = {
        "dataset": name,
        "n": len(recs),
        "wer": float(np.mean(wers)) if wers else float("nan"),
        "cer": float(np.mean(cers)) if cers else float("nan"),
        "cpwer": cp[0],
        "cpwer_cer": cp[1],
        "rtf": t_all / t_audio if t_audio else float("nan"),
        "audio_s": t_audio,
    }
    return stats, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "data/models/asr_model"))
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument(
        "--datasets", default="ami_ihm,ami_sdm,aishell4,alimeeting,cantonese"
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = [s.strip() for s in args.datasets.split(",") if s.strip()]

    all_stats = []
    for name in names:
        print(f"=== {name} ({args.samples} samples) ===", flush=True)
        try:
            stats, recs = run_dataset(name, args.samples, args.model_dir, args.verbose)
        except Exception as e:
            print(f"  [FAILED] {name}: {e}")
            continue
        print(
            f"  n={stats['n']}  WER={stats['wer']*100:.2f}%  CER={stats['cer']*100:.2f}%  "
            f"cpWER={stats['cpwer']*100:.2f}%  RTF={stats['rtf']:.4f}"
        )
        all_stats.append(stats)
        with open(OUT_DIR / f"{name}_results.json", "w", encoding="utf-8") as f:
            json.dump(
                {"stats": stats, "records": recs}, f, ensure_ascii=False, indent=1
            )

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=1)

    # table + plot
    print(
        f"\n{'dataset':<12s} {'n':>5s} {'WER%':>7s} {'CER%':>7s} {'cpWER%':>7s} {'RTF':>8s}"
    )
    for s in all_stats:
        print(
            f"{s['dataset']:<12s} {s['n']:>5d} {s['wer']*100:>7.2f} {s['cer']*100:>7.2f} "
            f"{s['cpwer']*100:>7.2f} {s['rtf']:>8.4f}"
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ds = [s["dataset"] for s in all_stats]
        x = np.arange(len(ds))
        w = 0.25
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w, [s["wer"] * 100 for s in all_stats], w, label="WER")
        ax.bar(x, [s["cer"] * 100 for s in all_stats], w, label="CER")
        ax.bar(x + w, [s["cpwer"] * 100 for s in all_stats], w, label="cpWER(oracle)")
        ax.set_xticks(x, ds)
        ax.set_ylabel("error rate (%)")
        ax.set_title(f"VeloxVoice ASR accuracy ({args.samples} samples/dataset)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        for i, s in enumerate(all_stats):
            ax.text(
                i - w,
                s["wer"] * 100,
                f"{s['wer']*100:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.text(
                i,
                s["cer"] * 100,
                f"{s['cer']*100:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.tight_layout()
        png = OUT_DIR / "accuracy_benchmark.png"
        fig.savefig(png, dpi=140)
        print(f"\nchart: {png}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")


if __name__ == "__main__":
    main()
