#!/usr/bin/env python3
"""Vocal / accompaniment separation for the music lane.

Default model: ``htdemucs_ft`` (Demucs v4 fine-tuned).  It is a strong,
fast local baseline.  For maximum vocal purity use a BS-RoFormer /
Mel-Band RoFormer checkpoint through UVR or ``audio-separator`` instead;
those are the current SOTA models for vocal extraction (see the project
notes).  Demucs is deliberately kept as the zero-extra-download fallback.

Outputs two stems next to the output directory root:

    vocals.wav         singing voice only
    instrumental.wav   accompaniment (drums + bass + other mixed)

Usage:
    python tools/music/separate_vocals.py song.mp3 -o out_dir [--device cuda]

Used by the AMR demo lyrics lane (ASR runs on the vocal stem); also usable
standalone.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

MODEL = os.environ.get("VELOX_DEMUCS_MODEL", "htdemucs_ft")


def _patch_torchaudio_save() -> None:
    """Save with soundfile instead of TorchCodec.

    torchaudio >= 2.9 routes ``save`` through torchcodec, which is not
    installed in the VeloxVoice venv.  Demucs only needs to write ordinary
    PCM WAV stems, and soundfile is already a dependency, so replace the
    writer instead of pulling in another native codec stack.
    """
    import soundfile as sf
    import torchaudio

    def _save(path, src, sample_rate, **kwargs):
        array = src.detach().cpu().float().numpy()
        if array.ndim == 1:
            array = array[None, :]
        bits = int(kwargs.get("bits_per_sample") or 16)
        subtype = "PCM_24" if bits >= 24 else "PCM_16"
        sf.write(str(path), array.T, int(sample_rate), subtype=subtype)

    torchaudio.save = _save


def separate(audio: str | Path, out_dir: str | Path,
             device: str = "cuda", model: str = MODEL) -> tuple[Path, Path]:
    """Separate ``audio`` into vocals.wav + instrumental.wav under ``out_dir``.

    Returns (vocals_path, instrumental_path). Cached: existing stems are
    reused when they were produced by the same model, so repeated calls
    (e.g. server retries) cost nothing.
    """
    audio, out_dir = Path(audio), Path(out_dir)
    if not audio.is_file():
        raise FileNotFoundError(f"audio not found: {audio}")
    out_dir.mkdir(parents=True, exist_ok=True)
    vocals = out_dir / "vocals.wav"
    instrumental = out_dir / "instrumental.wav"
    marker = out_dir / "vocals.model"
    cached_model = (marker.read_text(encoding="utf-8").strip()
                    if marker.is_file() else None)
    if (vocals.is_file() and instrumental.is_file() and
            cached_model == model):
        return vocals, instrumental

    from demucs.separate import main as demucs_main

    _patch_torchaudio_save()
    work = out_dir / "_demucs_work"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    t0 = time.monotonic()
    demucs_main(["--two-stems", "vocals", "-n", model, "-d", device,
                 "--int24", "-o", str(work), str(audio)])
    elapsed = time.monotonic() - t0

    stems = work / model / audio.stem
    raw_vocals = stems / "vocals.wav"
    raw_no_vocals = stems / "no_vocals.wav"
    if not raw_vocals.is_file() or not raw_no_vocals.is_file():
        raise RuntimeError(f"demucs produced no stems under {stems}")
    shutil.move(str(raw_vocals), vocals)
    shutil.move(str(raw_no_vocals), instrumental)
    marker.write_text(model + "\n", encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(f"[separate] {model}: {elapsed:.1f}s -> {vocals.name}, "
          f"{instrumental.name}", file=sys.stderr)
    return vocals, instrumental


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", help="input song (mp3/wav/flac/...)")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="output directory (default: alongside the input)")
    ap.add_argument("-d", "--device", default="cuda", help="cuda | cpu")
    ap.add_argument("-n", "--model", default=MODEL, help="demucs model name")
    args = ap.parse_args()
    audio = Path(args.audio)
    out_dir = Path(args.out_dir) if args.out_dir else audio.parent
    vocals, instrumental = separate(audio, out_dir, device=args.device,
                                    model=args.model)
    print(vocals)
    print(instrumental)


if __name__ == "__main__":
    main()
