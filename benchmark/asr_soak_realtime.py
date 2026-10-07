#!/usr/bin/env python3
"""Real-time multi-channel soak test for VeloxVoice ASR (WS lane).

N independent ws sessions stream audio PACED AT 1x REALTIME,
looping a source file seamlessly, for a configurable wall duration.

Channels are ramped in safe order (default 8 → 16 → 32); the sweep stops at the
first failing level.

python benchmark/asr_soak_realtime.py --levels 8 16 32 --duration 120
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import threading
import time
import wave
from pathlib import Path

import aiohttp

CHUNK_SECONDS = 1.0


def load_pcm(path: Path) -> tuple[bytes, int, int, float]:
    """Decode to raw s16le mono 16 kHz.

    The server interprets `-ar/-ac` as the RAW INPUT layout,
    so the client must hand over true 16 kHz mono PCM for 1 paced
    second to equal 1 audio second.
    """
    tmp = Path("/tmp") / f"soak_{path.stem}_16k_mono.pcm"
    seconds = 0.0
    if not tmp.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-f",
                "s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(tmp),
            ],
            check=True,
        )
    with wave.open(str(path)) as handle:
        seconds = handle.getnframes() / handle.getframerate()
    return tmp.read_bytes(), 16000, 1, seconds


class Channel:
    """One paced ws session; tracks backlog drift against the live clock."""

    def __init__(
        self,
        index: int,
        server: str,
        pcm: bytes,
        rate: int,
        channels: int,
        duration: float,
        pool_seconds: float,
    ):
        self.index = index
        self.server = server
        self.pcm = pcm
        self.rate = rate
        self.channels = channels
        self.duration = duration
        self.pool_seconds = pool_seconds
        self.pools = 0
        self.audio_processed = 0.0
        self.engine_seconds = 0.0
        self.disconnects = 0
        self.error: str | None = None

    async def run(self, results: list[dict]) -> None:
        chunk_bytes = int(CHUNK_SECONDS * self.rate) * self.channels * 2
        sent_seconds = 0.0
        started = time.perf_counter()
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=None, sock_connect=30, sock_read=None
                )
            ) as session:
                async with session.ws_connect(f"{self.server}/v1/audio/ws") as ws:
                    await ws.send_str(
                        json.dumps(
                            {
                                "type": "start",
                                "filename": f"soak_ch{self.index}.pcm",
                                "format": "s16le",
                                "sample_rate": self.rate,
                                "pool_seconds": self.pool_seconds,
                            }
                        )
                    )
                    done = asyncio.Event()

                    async def drain() -> None:
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            obj = json.loads(msg.data)
                            if obj.get("type") == "error":
                                self.error = str(obj.get("detail"))[:200]
                                done.set()
                                break
                            if obj.get("type") in ("partial", "final"):
                                self.pools += 1
                                if obj.get("type") == "final":
                                    # authoritative session total
                                    self.audio_processed = float(
                                        obj.get("audio_seconds") or self.audio_processed
                                    )
                                else:
                                    # each partial covers one distinct pool
                                    self.audio_processed += float(
                                        obj.get("audio_seconds") or 0
                                    )
                                self.engine_seconds += float(
                                    obj.get("elapsed_seconds") or 0
                                )
                                if obj.get("type") == "final":
                                    done.set()
                                    break

                    reader = asyncio.create_task(drain())
                    pos = 0
                    while (
                        time.perf_counter() - started < self.duration
                        and not done.is_set()
                    ):
                        send_t = time.perf_counter()
                        chunk = self.pcm[pos : pos + chunk_bytes]
                        pos = (pos + chunk_bytes) % len(self.pcm)
                        if chunk:
                            await ws.send_bytes(chunk)
                        sent_seconds += len(chunk) / (self.rate * self.channels * 2)
                        # pace to 1x realtime
                        delay = CHUNK_SECONDS - (time.perf_counter() - send_t)
                        if delay > 0:
                            await asyncio.sleep(delay)
                    await ws.send_str(json.dumps({"type": "stop"}))
                    try:
                        await asyncio.wait_for(reader, timeout=self.pool_seconds * 3)
                    except asyncio.TimeoutError:
                        self.error = "final not received before timeout"
            self.disconnects = 0
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        self.wall = time.perf_counter() - started
        # backlog: audio sent into the pipe that the engine has not processed yet
        self.backlog_s = sent_seconds - self.audio_processed


class GpuSampler:
    def __init__(self, gpu: int, interval: float = 5.0):
        self.gpu, self.interval = gpu, interval
        self.samples: list[tuple[int, int]] = []
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

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

    def stop(self) -> dict:
        self._stop.set()
        if not self.samples:
            return {"gpu_mem_max_mib": 0, "gpu_util_max_pct": 0}
        return {
            "gpu_mem_max_mib": max(m for m, _ in self.samples),
            "gpu_util_max_pct": max(u for _, u in self.samples),
        }


def summarize(level: int, channels: "list[Channel]", wall: float, gpu: dict) -> dict:
    pool_seconds = channels[0].pool_seconds if channels else 20.0
    ok = [c for c in channels if c.error is None]
    backlogs = [c.backlog_s for c in ok]
    pools = [c.pools for c in ok]
    rtfs = [c.engine_seconds / c.audio_processed for c in ok if c.audio_processed > 1]
    return {
        "channels": level,
        "ok": len(ok),
        "failed": level - len(ok),
        "wall_s": round(wall, 1),
        "errors": [c.error for c in channels if c.error][:3],
        "backlog_s_mean": round(statistics.mean(backlogs), 2) if backlogs else None,
        "backlog_s_max": round(max(backlogs), 2) if backlogs else None,
        "pools_per_channel_mean": round(statistics.mean(pools), 1) if pools else 0,
        "engine_rtf_mean": round(statistics.mean(rtfs), 5) if rtfs else None,
        "realtime_safe": bool(ok) and backlogs and max(backlogs) < 2 * pool_seconds,
        "gpu": gpu,
    }


async def run_level(
    server: str,
    pcm: bytes,
    rate: int,
    level: int,
    duration: float,
    pool_seconds: float,
    gpu: int,
) -> dict:
    channels_n = level
    sampler = GpuSampler(gpu)
    started = time.perf_counter()
    # pcm is 16 kHz MONO (load_pcm guarantees it); channels=1 is the PCM layout,
    # NOT the soak level.
    channels = [
        Channel(i, server, pcm, rate, 1, duration, pool_seconds)
        for i in range(channels_n)
    ]
    await asyncio.gather(*(ch.run([]) for ch in channels))
    wall = time.perf_counter() - started
    gpu_stats = sampler.stop()
    return summarize(level, channels, wall, gpu_stats)


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--audio", default="test_data/speaker/meeting-test.wav")
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="soak wall duration per level (seconds)",
    )
    parser.add_argument("--pool-seconds", type=float, default=20.0)
    parser.add_argument("--gpu", type=int, default=5)
    args = parser.parse_args()

    wav = Path(args.audio)
    if not wav.exists():
        raise SystemExit(f"audio not found: {wav}")
    pcm, rate, channels_wav, seconds = load_pcm(wav)
    print(
        f"server={args.server} source={wav.name} {seconds:.1f}s {rate}Hz x{channels_wav} "
        f"→ paced 1x realtime, {args.duration:.0f}s soak per level",
        flush=True,
    )
    for level in args.levels:
        row = await run_level(
            args.server, pcm, rate, level, args.duration, args.pool_seconds, args.gpu
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if not row["realtime_safe"] or row["failed"]:
            print(
                f"stopping: {level} channels not real-time safe (safe mode)", flush=True
            )
            break


if __name__ == "__main__":
    asyncio.run(main_async())
