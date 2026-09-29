"""Transcribe audio through the public VeloxVoice API.

Examples:
  # full-context transcription (same speed path as tools/bench_wenet_multi_worker.py)
  python examples/transcribe.py --model-dir data/models/asr_model \
      --audio speaker_test/meeting-test.wav

  # streaming partial results while audio is consumed
  python examples/transcribe.py --model-dir data/models/asr_model \
      --audio speaker_test/meeting-test.wav --stream
"""

from __future__ import annotations

import argparse
import subprocess

import numpy as np

SAMPLING_RATE = 16_000


def ffmpeg_decode_f32(path: str, seconds: float | None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error"]
    if seconds is not None:
        cmd += ["-t", str(seconds)]
    cmd += ["-i", path, "-ac", "1", "-ar", str(SAMPLING_RATE), "-f", "f32le", "-"]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, check=True, timeout=120)
        return np.frombuffer(proc.stdout, dtype=np.float32)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        import wave

        with wave.open(path, "rb") as w:
            sr, ch = w.getframerate(), w.getnchannels()
            pcm = (
                np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(
                    np.float32
                )
                / 32768.0
            )
            if ch > 1:
                pcm = pcm.reshape(-1, ch).mean(-1)

        from scipy.signal import resample_poly

        g = np.gcd(int(sr), SAMPLING_RATE)
        pcm = resample_poly(pcm, SAMPLING_RATE // g, sr // g).astype(np.float32)
        if seconds is not None:
            pcm = pcm[: int(seconds * SAMPLING_RATE)]
        return pcm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--enable-graphs", action="store_true")
    ap.add_argument("--pool-seconds", type=float, default=20.0)
    ap.add_argument("--rtf-goal", type=float, default=0.05)
    ap.add_argument(
        "--stream",
        action="store_true",
        help="emit partial transcriptions from pooled audio",
    )
    args = ap.parse_args()

    import veloxvoice

    pcm = ffmpeg_decode_f32(args.audio, args.seconds)
    audio_t = len(pcm) / SAMPLING_RATE

    vx = veloxvoice.Velox.load(
        args.model_dir,
        backend="torch",
        native_kernels=True,
        kernel_precision=args.precision,
        enable_graphs=args.enable_graphs,
    )

    if not args.stream:
        result = vx.transcribe_pcm(pcm)
        print(
            f"[platform={vx.platform.name} backend={vx.platform.module_backend} "
            f"precision={args.precision} audio={audio_t:.1f}s]"
        )
        print(f"\nfinal ({len(result.text)} chars):\n  {result.text}")
        print(
            f"\nencode+ctc: {result.elapsed_seconds:.3f}s  "
            f"RTF={result.rtf:.4f} target={args.rtf_goal} "
            f"{'PASS' if result.rtf < args.rtf_goal else 'FAIL'}"
        )
        return

    def chunks():
        step = SAMPLING_RATE
        for i in range(0, len(pcm), step):
            yield pcm[i : i + step]

    print(
        f"[platform={vx.platform.name} backend={vx.platform.module_backend} "
        f"precision={args.precision} pool={args.pool_seconds:.1f}s "
        f"audio={audio_t:.1f}s]"
    )
    for result in vx.transcribe_pcm_iter(chunks(), pool_seconds=args.pool_seconds):
        print(
            f"[partial {result.audio_seconds:6.2f}s rtf={result.rtf:.4f}] "
            f"{result.text}",
            flush=True,
        )


if __name__ == "__main__":
    main()
