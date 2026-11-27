"""Transcribe music audio through the public VeloxVoice API (SheetSage2 lane).

Examples:
  python examples/transcribe_music.py --model-dir /raid/yiakwy/SheetSage2 \
      --base-model-dir /raid/yiakwy/MERT-v2-FullSong \
      --audio test_data/music/bohemian_rhapsody_queen.mp3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


MUSIC_SAMPLE_RATE = 24_000


def ffmpeg_decode_f32(path: str, seconds: float | None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error"]
    if seconds is not None:
        cmd += ["-t", str(seconds)]
    cmd += ["-i", path, "-ac", "1", "-ar", str(MUSIC_SAMPLE_RATE),
            "-f", "f32le", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, check=True,
                              timeout=600)
        return np.frombuffer(proc.stdout, dtype=np.float32)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        import soundfile as sf
        from scipy.signal import resample_poly

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        pcm = data.mean(axis=1)
        g = np.gcd(int(sr), MUSIC_SAMPLE_RATE)
        pcm = resample_poly(pcm, MUSIC_SAMPLE_RATE // g, sr // g)
        if seconds is not None:
            pcm = pcm[: int(seconds * MUSIC_SAMPLE_RATE)]
        return pcm.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model-dir", required=True,
                    help="SheetSage2 checkpoint dir")
    ap.add_argument("--base-model-dir", required=True,
                    help="MERT-v2-FullSong encoder dir")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--output", default=None,
                    help="output dir for ABC/MIDI (default: none)")
    args = ap.parse_args()

    import veloxvoice

    pcm = ffmpeg_decode_f32(args.audio, args.seconds)
    audio_t = len(pcm) / MUSIC_SAMPLE_RATE

    vx = veloxvoice.Velox.load(
        args.model_dir,
        base_model_dir=args.base_model_dir,
        backend="torch",
        native_kernels=True,
        kernel_precision=args.precision,
        enable_graphs=True,
    )
    
    #  before timing to build jit kernels, and graph captures
    vx.transcribe_pcm(pcm)

    # VeloxMusic.transcribe_pcm resamples to the music lane's 24 kHz grid
    result = vx.transcribe_pcm(pcm, output_dir=args.output)

    platform = getattr(vx, "platform", None)
    platform_name = getattr(platform, "name", "cuda")
    backend = getattr(platform, "module_backend", "torch")
    print(
        f"[platform={platform_name} backend={backend} "
        f"precision={args.precision} audio={audio_t:.1f}s]"
    )
    if not isinstance(result, dict):
        result = dict(result)
    events = result.get("num_events")
    duration = result.get("duration_seconds") or audio_t
    elapsed = result.get("elapsed_seconds", 0.0)
    print(f"\nevents: {events}")
    print(f"\ntranscribe: {elapsed:.3f}s  RTF={elapsed / duration:.5f}")
    if args.output:
        print(f"output: {args.output}")


if __name__ == "__main__":
    main()
