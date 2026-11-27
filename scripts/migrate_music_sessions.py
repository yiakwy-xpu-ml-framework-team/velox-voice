#!/usr/bin/env python3
"""Copy legacy music sessions into the persistent root and build metadata.

Legacy sessions lived in /tmp/veloxvoice_music_output and disappeared on a
reboot.  This script copies them to output/music_sessions (or --dst), writes
session_meta.json for each upload, and builds the sha256 -> session index used
by /v1/audio/music/upload for reuse.

Usage:
    python scripts/migrate_music_sessions.py
    python scripts/migrate_music_sessions.py --src /tmp/veloxvoice_music_output \
        --dst output/music_sessions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
DERIVED_NAMES = {
    "vocals.wav", "instrumental.wav", "no_vocals.wav",
    "piano_A_synth.wav", "piano_B.wav",
    "piano_mix.wav", "piano_vocal.wav", "piano_instrumental.wav",
}


def is_original_audio(path: Path) -> bool:
    return (path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES and
            path.name not in DERIVED_NAMES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="/tmp/veloxvoice_music_output")
    parser.add_argument("--dst", default=str(
        Path(__file__).resolve().parents[1] / "output" / "music_sessions"))
    parser.add_argument("--move", action="store_true",
                        help="move instead of copy (default: copy)")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        raise SystemExit(f"source not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    index_path = dst / "_index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(index, dict):
            index = {}
    except Exception:
        index = {}

    copied = skipped = 0
    for entry in sorted(src.iterdir()):
        if not entry.is_dir():
            continue
        if len(entry.name) != 12 or any(c not in "0123456789abcdef"
                                        for c in entry.name):
            continue
        target = dst / entry.name
        if not target.exists():
            if args.move:
                shutil.move(str(entry), str(target))
            else:
                shutil.copytree(entry, target)
            copied += 1
        else:
            skipped += 1
        # Drop failed demucs work dirs if they were copied.
        work = target / "_demucs_work"
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        audio_files = [p for p in target.iterdir() if is_original_audio(p)]
        if not audio_files:
            continue
        audio = audio_files[0]
        digest = sha256_file(audio)
        meta_path = target / "session_meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        if not meta.get("sha256"):
            result_path = target / "result.json"
            stamp_source = (result_path if result_path.exists()
                            else target)
            created = datetime.fromtimestamp(
                stamp_source.stat().st_mtime, timezone.utc).isoformat()
            meta = {
                "session": entry.name,
                "original_filename": meta.get("original_filename") or
                                     audio.name,
                "saved_filename": audio.name,
                "size": audio.stat().st_size,
                "sha256": digest,
                "created_at": meta.get("created_at") or created,
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False,
                                            indent=1), encoding="utf-8")
        index[digest] = {
            "session": entry.name,
            "filename": meta.get("original_filename") or audio.name,
            "saved_filename": audio.name,
            "size": audio.stat().st_size,
            "created_at": meta.get("created_at"),
        }

    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    print(f"source: {src}")
    print(f"target: {dst}")
    print(f"copied: {copied}, already present: {skipped}, "
          f"indexed: {len(index)}")
    print(f"index:  {index_path}")


if __name__ == "__main__":
    main()
