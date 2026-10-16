"""ASR accuracy benchmark on the LOCAL AISHELL-4 testset (test_data/testset/).

Scores WER / CER of the VeloxVoice pipeline against test_data/testset/text

Outputs (benchmark/accuracy/):
  - aishell4_local_results.json  (aggregate stats + per-utterance records)
  - aishell4_local.png           (WER/CER bars + per-utterance CER histogram)

With --log-diffs, per-utterance ref-vs-hyp diffs land in
logs/<dataset-relpath>/<utt_id>.txt (mirroring the testset layout) plus a
mismatch_summary.txt index sorted by CER.

Usage:
  python tools/bench_accuracy_local.py [--samples 0] [--verbose]
    [--dataset-dir test_data/testset/aishell4] [--manifest test_data/testset/text]
    [--log-diffs [--log-dir logs]]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wer_metrics import _chars, _edit_distance, cer, normalize, wer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmark" / "accuracy"


# ---------------------------------------------------------------------------
# Transcription (VeloxVoice pipeline: frontend + fp16 autocast encoder + CTC)
# ---------------------------------------------------------------------------

_ASR = None
_FE = None

# dedup switch: False (default)
_DEDUP = False


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
        # Precision lane (A/B): VELOXVOICE_PRECISION in {fp16, bf16, fp32}.
        #   fp16 = autocast-fp16 encoder on fp32 weights (historical default)
        #   bf16 = bf16 weights+encoder
        #   fp32 = full fp32 (no autocast)
        precision = os.environ.get("VELOXVOICE_PRECISION", "fp16").lower()
        if precision == "fp16":
            m.set_fp16(True)
        elif precision == "bf16":
            m.set_precision("bf16")
        elif precision == "fp32":
            m.set_precision("fp32")
        else:
            raise SystemExit(f"unknown VELOXVOICE_PRECISION={precision!r}")
        _ASR = m
        _FE = TorchGpuFrontend(
            FrontendConfig(sample_rate=16000, num_mel_bins=cfg.input_dim), "cuda"
        )
    return _ASR


def transcribe_pcm(pcm: np.ndarray, sr: int) -> str:
    """pcm float32 [-1,1] at sr -> transcript (one shared frontend)."""
    import torch

    if sr != 16000:
        import torchaudio.functional as AF

        t = torch.from_numpy(pcm).float()
        t = AF.resample(t, sr, 16000)
        pcm = t.numpy()

    m = _ASR
    pcm = np.ascontiguousarray(pcm.astype(np.float32))
    feats = torch.cat([_FE.accept(pcm), _FE.flush()], dim=0)
    if feats.shape[0] < 8:
        return ""
    enc = m.encode_utterance(feats[None])
    logp = m.ctc_logp(enc)[0]

    # RESCORE=1 routes the default lane to attention rescoring; 
    # explicit VELOXVOICE_DECODE overrides.
    default_decode = "greedy"
    if os.environ.get("RESCORE", "0") == "1" and not os.environ.get(
        "VELOXVOICE_DECODE"
    ):
        default_decode = "att_rescore"
    mode = os.environ.get("VELOXVOICE_DECODE", default_decode).lower()
    if mode == "att_rescore":
        from veloxvoice.stream.rescore import get_att_rescorer

        decoder = getattr(m, "attention_decoder", None)
        if decoder is not None:
            rs = get_att_rescorer(decoder)
            with torch.inference_mode():
                ids = rs.decode(enc, logp)
            return _ids_to_text(m, ids).strip()
        print("[warn] attention decoder unavailable; falling back to greedy")

    ids = logp.argmax(-1)
    # single standard CTC collapse: adjacent equal frames merge, blank resets.
    prev = torch.cat([ids.new_zeros(1), ids[:-1]])
    keep = (ids != 0) & (ids != prev)
    kept = ids[keep.nonzero().squeeze(-1)].cpu().tolist()
    return _ids_to_text(m, kept).strip()


def _ids_to_text(m, kept) -> str:
    if not _DEDUP:
        out = kept
    else:
        # legacy second collapse (adjacent kept ids deduped again)
        out, last = [], 0
        for t in kept:
            t = int(t)
            if t != last and t != 0:
                out.append(t)
            last = t
    from veloxvoice.audio.text_tokenizer import TextTokenizer

    tok = getattr(m, "_text_tok", None)
    if tok is None:
        tok = TextTokenizer(os.path.join(m.model_dir, "units.txt"))
        m._text_tok = tok
    return tok.ids_to_text(out).strip()


# ---------------------------------------------------------------------------
# Local dataset loader: test_data/testset/aishell4/*.wav + testset/text manifest
# ---------------------------------------------------------------------------


def load_local_aishell4(dataset_dir: str, manifest: str, n: int):
    refs = {}
    for line in open(manifest, encoding="utf-8"):
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            refs[parts[0]] = parts[1]

    import soundfile as sf

    wavs = sorted(Path(dataset_dir).glob("*.wav"))
    out = []
    for wav in wavs:
        if n and len(out) >= n:
            break
        utt_id = wav.stem
        if utt_id not in refs:
            continue
        data, sr = sf.read(str(wav), dtype="float32")
        if data.ndim > 1:
            data = data.mean(-1)
        if data.shape[0] < sr // 4:  # skip <0.25 s snippets
            continue
        # session = AISHELL-4 meeting id (e.g. L_R003S01C02)
        session = "_".join(utt_id.split("_")[:2])
        out.append(
            (utt_id, data.astype(np.float32), sr, refs[utt_id], session, str(wav))
        )
    return out


# ---------------------------------------------------------------------------
# Mismatch logging: char-level diff of ref vs hyp, one file per utterance,
# mirroring the testset layout under <log-dir>/<dataset-relpath>/<utt_id>.txt
# ---------------------------------------------------------------------------

# error taxonomy: acoustically undecidable / annotation-convention / true errors
_FILLERS = set("呃嗯啊呢唉哦哎哈")
_FUNC = set("的了是一個个也你我他她它这那就都还在有不吧吗嘛们")
CAT_DESC = {
    "filler": "填充词/语气词 (呃嗯啊呢) — 声学不可分, 标注口径随机",
    "erhua": "儿化音 (儿) — 发音与标注不一致",
    "func_sub": "近音功能词替换 (这/那, 的/了/呢, 吗/嘛)",
    "func_del": "弱读功能词脱落 (一/是/也/的/你/我...)",
    "func_ins": "多识功能词",
    "content": "内容词误识 — 真实的识别错误",
    "boundary": "整句错配 (CER>50%) — TextGrid 切分边界/说话人重叠",
}


_TAG_MAP = {"replace": "SUB", "delete": "DEL", "insert": "INS"}


def char_ops(ref: str, hyp: str) -> list[tuple[str, int, str, str]]:
    """[(tag, pos, ref_txt, hyp_txt)] opcodes after ASR-eval normalization.

    tag is one of SUB / DEL / INS (ref-side perspective: DEL = ref text the
    hypothesis missed, INS = extra text the hypothesis produced).
    """
    import difflib

    r = [c for c in normalize(ref) if not c.isspace()]
    h = [c for c in normalize(hyp) if not c.isspace()]
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=r, b=h, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        ops.append((_TAG_MAP[tag], i1, "".join(r[i1:i2]), "".join(h[j1:j2])))
    return ops


def format_ops(ops) -> list[str]:
    lines = []
    for tag, pos, rs, hs in ops:
        if tag == "SUB":
            lines.append(f"  @{pos:<4d} SUB  ref「{rs}」 -> hyp「{hs}」")
        elif tag == "DEL":
            lines.append(f"  @{pos:<4d} DEL  ref「{rs}」 not in hyp")
        else:
            lines.append(f"  @{pos:<4d} INS  hyp「{hs}」 not in ref")
    return lines


def op_category(tag: str, rs: str, hs: str) -> str:
    # filler/erhua are annotation-convention classes: only for short ops
    # (a long op that merely contains 儿/嗯 is a real content error)
    if max(len(rs), len(hs)) <= 2:
        sides = [t for t in (rs, hs) if t]
        if any(set(t) and set(t) <= _FILLERS for t in sides):
            return "filler"
        if any("儿" in t for t in sides):
            return "erhua"
    if tag == "SUB" and len(rs) == 1 and len(hs) == 1 and (rs in _FUNC or hs in _FUNC):
        return "func_sub"
    if tag == "DEL" and set(rs) <= _FUNC:
        return "func_del"
    if tag == "INS" and set(hs) <= _FUNC:
        return "func_ins"
    return "content"


def utt_category(rec: dict) -> str:
    """Utterance-level label: boundary artifact if severe, else dominant op class."""
    if rec["cer"] > 0.5:
        return "boundary"
    from collections import Counter

    cats = Counter(op_category(t, rs, hs) for t, _, rs, hs in rec.get("ops", []))
    if not cats:
        return "content"
    pref = ["filler", "erhua", "func_del", "func_sub", "func_ins", "content"]
    return max(cats, key=lambda c: (cats[c], -pref.index(c)))


def write_diff_logs(recs, log_dir: Path, dataset_dir: str, n_total: int):
    """Per-utterance diff files (mismatches only) + error taxonomy summary."""
    d = Path(dataset_dir)
    try:
        rel = d.resolve().relative_to(ROOT)
    except ValueError:
        rel = Path(d.name)
    out_root = log_dir / rel
    out_root.mkdir(parents=True, exist_ok=True)

    bad = [r for r in recs if r["wer"] > 0 or r["cer"] > 0]
    for r in bad:
        r["ops"] = char_ops(r["ref"], r["hyp"])
        r["category"] = utt_category(r)
        lines = [
            f"utt_id:   {r['id']}",
            f"wav:      {r.get('wav', '?')}",
            f"session:  {r['session']}",
            f"WER: {r['wer']*100:.2f}%   CER: {r['cer']*100:.2f}%",
            f"category: {r['category']} — {CAT_DESC[r['category']]}",
            "",
            f"REF: {normalize(r['ref'])}",
            f"HYP: {normalize(r['hyp'])}",
            "",
            "diff (char-level, after normalization):",
        ]
        lines += format_ops(r["ops"]) or ["  (word-level mismatch only)"]
        (out_root / f"{r['id']}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    bad_sorted = sorted(bad, key=lambda r: -r["cer"])
    summary = [
        f"# {len(bad)}/{n_total} utterances with mismatches; sorted by CER desc",
        f"# per-utterance diffs: {out_root}/<utt_id>.txt",
        f"{'CER%':>7s} {'WER%':>7s}  {'category':<10s} utt_id",
    ]
    for r in bad_sorted:
        summary.append(
            f"{r['cer']*100:>7.2f} {r['wer']*100:>7.2f}  {r['category']:<10s} {r['id']}"
        )
    (out_root / "mismatch_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    write_error_taxonomy(recs, bad, out_root, n_total)
    return out_root, bad


def write_error_taxonomy(recs, bad, out_root: Path, n_total: int):
    """Aggregate the per-op diff annotations into logs/.../error_taxonomy.txt."""
    from collections import Counter

    op_cats = Counter()
    sub_pairs = Counter()          # confusion pairs among true content errors
    utt_cats = Counter()
    examples = {}                  # category -> [(cer, utt_id, line)]
    for r in bad:
        utt_cats[r["category"]] += 1
        for tag, pos, rs, hs in r.get("ops", []):
            cat = op_category(tag, rs, hs)
            op_cats[cat] += 1
            if cat == "content":
                sub_pairs[f"{tag} ref「{rs}」->hyp「{hs}」"] += 1
            line = f"{tag} @{pos} ref「{rs}」 hyp「{hs}」"
            examples.setdefault(cat, []).append((r["cer"], r["id"], line))

    tot_ops = sum(op_cats.values())
    lines = [
        "# AISHELL-4 local testset — error taxonomy",
        f"# utterances: {n_total} total, {len(bad)} with mismatches",
        f"# diff ops: {tot_ops}  (SUB/DEL/INS, char-level, after normalization)",
        "# same model everywhere: JIT-ON ≡ JIT-OFF bit-exact, errors are",
        "# model/data-inherent, not kernel/platform-related",
        "#",
        "# NOTE: a few DELs are REFERENCE-side errors, not ASR errors",
        "# (user-verified example: L_R003S01C02_interval_83 — audio says 咱,",
        "# ref writes 咱们), so content/func_del slightly overcount true",
        "# ASR errors.",
        "",
        "== per-op categories ==",
        f"{'category':<10s} {'ops':>5s} {'share':>7s}  description",
    ]
    for cat in CAT_DESC:
        n = op_cats.get(cat, 0)
        if n:
            lines.append(
                f"{cat:<10s} {n:>5d} {n / max(tot_ops, 1) * 100:>6.1f}%  {CAT_DESC[cat]}"
            )
    n_b = utt_cats.get("boundary", 0)
    if n_b:
        lines.append(
            f"{'boundary':<10s} {n_b:>5d} {'':>7s}  {CAT_DESC['boundary']} (utterance-level)"
        )

    lines += ["", "== top confusion pairs (content errors) =="]
    for pair, c in sub_pairs.most_common(15):
        lines.append(f"  x{c:<3d} {pair}")

    lines += ["", "== examples per category (worst first) =="]
    for cat in CAT_DESC:
        exs = sorted(examples.get(cat, []), key=lambda e: -e[0])[:5]
        if not exs:
            continue
        lines.append(f"-- {cat}: {CAT_DESC[cat]}")
        for cer_v, utt, ex in exs:
            lines.append(f"   [{cer_v*100:5.1f}%] {utt}: {ex}")

    lines += ["", "== per-session CER =="]
    from collections import defaultdict

    by_sess = defaultdict(list)
    for r in recs:
        by_sess[r["session"]].append(r["cer"])
    for s, cs in sorted(by_sess.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        lines.append(f"  {s:<14s} CER={sum(cs) / len(cs) * 100:6.2f}%  n={len(cs)}")

    (out_root / "error_taxonomy.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


def run(model_dir, dataset_dir, manifest, samples, verbose=False, log_diffs=False,
        log_dir=None, dedup=False):
    global _DEDUP
    _DEDUP = bool(dedup)
    _get_asr(model_dir)
    rows = load_local_aishell4(dataset_dir, manifest, samples)
    print(f"loaded {len(rows)} utterances from {dataset_dir}", flush=True)

    recs = []
    t_all = t_audio = 0.0
    edits = refchars = exact = 0
    for i, (utt_id, pcm, sr, ref, session, wav_path) in enumerate(rows):
        t0 = time.perf_counter()
        try:
            hyp = transcribe_pcm(pcm, sr)
        except Exception as e:
            if verbose:
                print(f"  [warn] {utt_id}: {e}")
            continue
        dt = time.perf_counter() - t0
        t_all += dt
        t_audio += len(pcm) / sr
        _r, _h = _chars(normalize(ref)), _chars(normalize(hyp))
        _d = _edit_distance(_r, _h)
        edits += _d
        refchars += len(_r)
        exact += int(_d == 0)
        recs.append(
            {
                "id": utt_id,
                "ref": ref,
                "hyp": hyp,
                "session": session,
                "wav": wav_path,
                "wer": wer(ref, hyp),
                "cer": cer(ref, hyp),
            }
        )
        if verbose:
            print(f"  {utt_id}: ref={normalize(ref)[:40]} | hyp={normalize(hyp)[:40]}")
        elif (i + 1) % 100 == 0:
            w = float(np.mean([r["wer"] for r in recs])) * 100
            c = float(np.mean([r["cer"] for r in recs])) * 100
            print(f"  [{i + 1}/{len(rows)}] running WER={w:.2f}% CER={c:.2f}%", flush=True)

    wers = [r["wer"] for r in recs]
    cers = [r["cer"] for r in recs]
    stats = {
        "dataset": "aishell4_local",
        "dataset_dir": str(dataset_dir),
        "dedup": bool(_DEDUP),
        "n": len(recs),
        "wer": float(np.mean(wers)) if wers else float("nan"),
        "cer": float(np.mean(cers)) if cers else float("nan"),
        "wer_median": float(np.median(wers)) if wers else float("nan"),
        "cer_median": float(np.median(cers)) if cers else float("nan"),
        "cer_corpus": edits / refchars if refchars else float("nan"),
        "edits": edits,
        "refchars": refchars,
        "exact": exact,
        "cer_p90": float(np.percentile(cers, 90)) if cers else float("nan"),
        "cer_p99": float(np.percentile(cers, 99)) if cers else float("nan"),
        "rtf": t_all / t_audio if t_audio else float("nan"),
        "audio_s": t_audio,
    }
    return stats, recs


def plot(stats, recs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(2)
    vals = [stats["wer"] * 100, stats["cer"] * 100]
    bars = ax1.bar(x, vals, 0.5, color=["#4C72B0", "#DD8452"])
    ax1.set_xticks(x, ["WER", "CER"])
    ax1.set_ylabel("error rate (%)")
    ax1.set_title(
        f"VeloxVoice ASR on local AISHELL-4 testset (n={stats['n']}, "
        f"RTF={stats['rtf']:.4f})"
    )
    ax1.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")

    cers = [r["cer"] * 100 for r in recs]
    ax2.hist(cers, bins=50, color="#4C72B0", alpha=0.85)
    ax2.axvline(
        stats["cer"] * 100, color="#DD8452", linestyle="--",
        label=f"mean CER {stats['cer']*100:.2f}%",
    )
    ax2.set_xlabel("per-utterance CER (%)")
    ax2.set_ylabel("utterances")
    ax2.set_title("CER distribution")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    png = OUT_DIR / "aishell4_local.png"
    fig.savefig(png, dpi=140)
    print(f"\nchart: {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "data/models/asr_model"))
    ap.add_argument(
        "--dataset-dir",
        default=str(ROOT / "test_data/testset/aishell4"),
        help="directory with pre-cut AISHELL-4 wav files",
    )
    ap.add_argument(
        "--manifest",
        default=str(ROOT / "test_data/testset/text"),
        help="Kaldi-style `utt_id transcript` manifest",
    )
    ap.add_argument(
        "--samples", type=int, default=0, help="0 = all utterances in the testset"
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--dedup",
        action="store_true",
        help="enable the legacy extra CTC dedup pass (drops real doubled "
        "symbols like 看看->看); default OFF = single correct collapse",
    )
    ap.add_argument(
        "--log-diffs",
        action="store_true",
        help="write per-utterance ref-vs-hyp diffs to <log-dir>/<dataset-relpath>/"
        " (one file per mismatching utterance + mismatch_summary.txt)",
    )
    ap.add_argument(
        "--log-dir",
        default=str(ROOT / "logs"),
        help="root dir for diff logs (default: <repo>/logs)",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats, recs = run(
        args.model_dir, args.dataset_dir, args.manifest, args.samples, args.verbose,
        log_diffs=args.log_diffs, log_dir=Path(args.log_dir), dedup=args.dedup,
    )
    print(
        f"\n=== {stats['dataset']} (dedup={'ON' if args.dedup else 'OFF'}) ===\n"
        f"n={stats['n']}  WER={stats['wer']*100:.2f}%  "
        f"mean CER={stats['cer']*100:.2f}%  median CER={stats['cer_median']*100:.2f}%  "
        f"p90/p99 CER={stats['cer_p90']*100:.1f}%/{stats['cer_p99']*100:.1f}%  "
        f"RTF={stats['rtf']:.4f}\n"
        f"corpus CER (total edits / total ref chars) = "
        f"{stats['cer_corpus']*100:.2f}% ({stats['edits']}/{stats['refchars']})  "
        f"exact={stats['exact']}/{stats['n']} "
        f"({stats['exact']/stats['n']*100:.1f}%)"
    )

    if args.log_diffs:
        out_root, bad = write_diff_logs(
            recs, Path(args.log_dir), args.dataset_dir, stats["n"]
        )
        print(
            f"diff logs: {len(bad)} mismatching utterances -> {out_root}/<utt_id>.txt"
            f" (+ {out_root / 'mismatch_summary.txt'})"
        )

    with open(OUT_DIR / "aishell4_local_results.json", "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "records": recs}, f, ensure_ascii=False, indent=1)
    print(f"results: {OUT_DIR / 'aishell4_local_results.json'}")

    plot(stats, recs)


if __name__ == "__main__":
    main()
