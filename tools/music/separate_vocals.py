#!/usr/bin/env python3
"""Vocal / accompaniment separation for the music lane (demucs, htdemucs).

Input a song, output two stems next to the output directory root:

    vocals.wav         singing voice only
    instrumental.wav   accompaniment (drums + bass + other mixed)

Usage:
    python tools/music/separate_vocals.py song.mp3 -o out_dir [--device cuda]

Used by the AMR demo lyrics lane (ASR runs on the vocal stem); also usable
standalone.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

MODEL = "htdemucs"


def separate(audio: str | Path, out_dir: str | Path,
             device: str = "cuda", model: str = MODEL) -> tuple[Path, Path]:
    """Separate ``audio`` into vocals.wav + instrumental.wav under ``out_dir``.

    Returns (vocals_path, instrumental_path). Cached: existing stems are
    reused, so repeated calls (e.g. server retries) cost nothing.
    """
    audio, out_dir = Path(audio), Path(out_dir)
    if not audio.is_file():
        raise FileNotFoundError(f"audio not found: {audio}")
    out_dir.mkdir(parents=True, exist_ok=True)
    vocals = out_dir / "vocals.wav"
    instrumental = out_dir / "instrumental.wav"
    if vocals.is_file() and instrumental.is_file():
        return vocals, instrumental

    from demucs.separate import main as demucs_main

    work = out_dir / "_demucs_work"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    t0 = time.monotonic()
    demucs_main(["--two-stems", "vocals", "-n", model, "-d", device,
                 "-o", str(work), str(audio)])
    elapsed = time.monotonic() - t0

    stems = work / model / audio.stem
    raw_vocals = stems / "vocals.wav"
    raw_no_vocals = stems / "no_vocals.wav"
    if not raw_vocals.is_file() or not raw_no_vocals.is_file():
        raise RuntimeError(f"demucs produced no stems under {stems}")
    shutil.move(str(raw_vocals), vocals)
    shutil.move(str(raw_no_vocals), instrumental)
    shutil.rmtree(work, ignore_errors=True)
    print(f"[separate] {model}: {elapsed:.1f}s -> {vocals.name}, {instrumental.name}",
          file=sys.stderr)
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
