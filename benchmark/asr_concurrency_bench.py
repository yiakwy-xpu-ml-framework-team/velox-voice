#!/usr/bin/env python3
"""Concurrency / RTF benchmark for the VeloxVoice ASR server.

Measures, per concurrency level:
  * e2e request latency (client wall) and throughput
  * server-side RTF from the response fields (elapsed_seconds / audio_seconds)
  * peak GPU memory + utilization (sampled while the level runs)

Example:
    python benchmark/asr_concurrency_bench.py --server http://127.0.0.1:8000 \
        --audio test_data/speaker/meeting-test.wav --levels 1 2 4 8 16 32
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(
        len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) + 1))) - 1)
    )
    return ordered[index]


class GpuSampler:
    """Samples GPU memory/utilization on a background thread."""

    def __init__(self, gpu: int, interval: float = 0.5):
        self.gpu = gpu
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        str(self.gpu),
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                mem, util = (int(v) for v in out.strip().split(", "))
                self.samples.append((mem, util))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def summary(self) -> dict:
        if not self.samples:
            return {"gpu_mem_max_mib": 0, "gpu_util_max_pct": 0}
        return {
            "gpu_mem_max_mib": max(m for m, _ in self.samples),
            "gpu_util_max_pct": max(u for _, u in self.samples),
        }


def one_request(
    server: str, audio_paths: list[Path], audio_seconds: float, timeout: float
) -> tuple[float, bool, dict]:
    one_request.index = getattr(one_request, "index", 0)
    audio_path = audio_paths[one_request.index % len(audio_paths)]
    one_request.index += 1
    started = time.perf_counter()
    try:
        with open(audio_path, "rb") as handle:
            response = requests.post(
                f"{server}/v1/audio/transcriptions",
                files={"file": (audio_path.name, handle, "audio/wav")},
                data={"response_format": "json"},
                timeout=timeout,
            )
        wall = time.perf_counter() - started
        if not response.ok:
            return wall, False, {"error": response.text[:200]}
        payload = response.json()
        elapsed_s = float(payload.get("elapsed_seconds") or 0.0)
        audio_s = float(payload.get("audio_seconds") or 0.0)

        # Breakdown: server processing (elapsed_seconds, GPU transcribe) vs
        # e2e; the residual carries upload + decode + queue + download.
        return (
            wall,
            True,
            {
                "audio_seconds": audio_s,
                "server_elapsed_s": elapsed_s,
                "server_rtf": elapsed_s / audio_s if audio_s else 0.0,
                "e2e_rtf": wall / audio_seconds if audio_seconds else 0.0,
            },
        )
    except Exception as exc:
        return (
            time.perf_counter() - started,
            False,
            {"error": f"{type(exc).__name__}: {exc}"},
        )


def audio_duration_seconds(audio_path: Path) -> float:
    if audio_path.suffix.lower() == ".wav":
        import wave

        with wave.open(str(audio_path)) as handle:
            return handle.getnframes() / handle.getframerate()
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        text=True,
    )
    return float(out.strip())


def run_level(
    server: str,
    audio_paths: list[Path],
    audio_seconds: float,
    requests_n: int,
    concurrency: int,
    timeout: float,
    gpu: int,
) -> dict:
    with GpuSampler(gpu) as sampler:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(
                pool.map(
                    lambda _: one_request(server, audio_paths, audio_seconds, timeout),
                    range(requests_n),
                )
            )

        wall = time.perf_counter() - started

    ok = [r for _, success, r in results if success]
    latencies = [w for w, success, _ in results if success]
    e2e_rtfs = [r.get("e2e_rtf", 0.0) for r in ok if r.get("audio_seconds")]
    server_rtfs = [r["server_rtf"] for r in ok if r.get("audio_seconds")]
    server_elapsed = [r["server_elapsed_s"] for r in ok if r.get("audio_seconds")]
    audio_s = ok[0].get("audio_seconds", 0.0) if ok else audio_seconds

    return {
        "concurrency": concurrency,
        "requests": requests_n,
        "success": len(latencies),
        "failed": requests_n - len(latencies),
        "wall_s": round(wall, 3),
        "throughput_rps": round(len(latencies) / wall, 3) if wall > 0 else 0.0,
        "audio_throughput_xrealtime": (
            round((len(latencies) * audio_seconds) / wall, 1) if wall > 0 else 0.0
        ),
        "latency_p50_s": round(percentile(latencies, 50), 3) if latencies else 0.0,
        "latency_p95_s": round(percentile(latencies, 95), 3) if latencies else 0.0,
        "server_process_mean_s": (
            round(statistics.mean(server_elapsed), 4) if server_elapsed else 0.0
        ),
        "server_rtf_mean": (
            round(statistics.mean(server_rtfs), 5) if server_rtfs else 0.0
        ),
        "e2e_rtf_mean": round(statistics.mean(e2e_rtfs), 4) if e2e_rtfs else 0.0,
        "e2e_rtf_p95": round(percentile(e2e_rtfs, 95), 4) if e2e_rtfs else 0.0,
        "upload_decode_residual_mean_s": (
            round(statistics.mean(latencies) - statistics.mean(server_elapsed), 4)
            if latencies and server_elapsed
            else 0.0
        ),
        "audio_seconds": round(audio_s, 2),
        "gpu": sampler.summary(),
        "failures": [r for _, success, r in results if not success][:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--audio",
        default="test_data/speaker/meeting-test.wav",
        help="audio file, or a glob for rotating inputs",
    )
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument(
        "--requests",
        type=int,
        default=0,
        help="requests per level (0 = max(concurrency, 8))",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--gpu", type=int, default=5)
    args = parser.parse_args()

    if any(ch in args.audio for ch in "*?["):
        audio_paths = sorted(Path().glob(args.audio))
    else:
        audio_paths = [Path(args.audio)]
    for path in audio_paths:
        if not path.exists():
            raise SystemExit(f"audio not found: {path}")
    audio_seconds = statistics.mean(
        audio_duration_seconds(path) for path in audio_paths
    )
    print(
        f"server={args.server} audio={len(audio_paths)} file(s) "
        f"duration={audio_seconds:.2f}s avg",
        flush=True,
    )
    for level in args.levels:
        requests_n = args.requests or max(level, 8)
        result = run_level(
            args.server,
            audio_paths,
            audio_seconds,
            requests_n,
            level,
            args.timeout,
            args.gpu,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
