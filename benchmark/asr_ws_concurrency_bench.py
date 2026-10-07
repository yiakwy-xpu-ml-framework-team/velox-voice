#!/usr/bin/env python3
"""WebRTC/WS streaming concurrency benchmark for VeloxVoice ASR.

Runs N independent ``/v1/audio/ws`` sessions (the streaming lane the test
webpage uses). Each session streams a whole audio file, collects the final
result and reports the timing breakdown separately from the HTTP lane:

  * server processing time / RTF  (final payload: elapsed_seconds / audio_seconds)
  * user end-to-end time / RTF    (session wall / audio_seconds)
  * upload+decode residual        (e2e - server processing)

Concurrency is increased in safe steps; every level must pass before the
next one runs.

    python benchmark/asr_ws_concurrency_bench.py --levels 1 2 4 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp


async def ws_session(
    session: aiohttp.ClientSession,
    server: str,
    wav: Path,
    audio_seconds: float,
    pool_seconds: float,
) -> dict:
    started = time.perf_counter()
    async with session.ws_connect(f"{server}/v1/audio/ws") as ws:
        await ws.send_str(
            json.dumps(
                {"type": "start", "filename": wav.name, "pool_seconds": pool_seconds}
            )
        )
        data = wav.read_bytes()
        for i in range(0, len(data), 64 * 1024):
            await ws.send_bytes(data[i : i + 64 * 1024])
        await ws.send_str(json.dumps({"type": "stop"}))
        final = None
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            obj = json.loads(msg.data)
            if obj.get("type") in ("final", "error"):
                final = obj
                break
    e2e = time.perf_counter() - started
    if final is None or final.get("type") == "error":
        return {
            "ok": False,
            "error": (final or {}).get("detail", "no final message"),
            "e2e_s": round(e2e, 3),
        }
    server_elapsed = float(final.get("elapsed_seconds") or 0.0)
    server_audio = float(final.get("audio_seconds") or audio_seconds)
    return {
        "ok": True,
        "e2e_s": round(e2e, 3),
        "server_process_s": round(server_elapsed, 4),
        "server_rtf": round(server_elapsed / server_audio, 5) if server_audio else 0.0,
        "e2e_rtf": round(e2e / audio_seconds, 4) if audio_seconds else 0.0,
        "upload_decode_residual_s": round(e2e - server_elapsed, 3),
        "audio_seconds": round(server_audio, 2),
    }


def summarize(level: int, results: list[dict], wall: float) -> dict:
    ok = [r for r in results if r["ok"]]
    row = {
        "concurrency": level,
        "sessions": len(results),
        "success": len(ok),
        "failed": len(results) - len(ok),
        "wall_s": round(wall, 3),
        "throughput_sessions_per_s": round(len(ok) / wall, 3) if wall else 0.0,
    }
    if ok:
        for key in (
            "e2e_s",
            "server_process_s",
            "server_rtf",
            "e2e_rtf",
            "upload_decode_residual_s",
        ):
            values = [r[key] for r in ok]
            row[f"{key}_mean"] = round(statistics.mean(values), 4)
            row[f"{key}_max"] = round(max(values), 4)
    row["failures"] = [r.get("error") for r in results if not r["ok"]][:3]
    return row


async def run_level(
    server: str,
    wav: Path,
    audio_seconds: float,
    level: int,
    pool_seconds: float,
    timeout: float,
) -> dict:
    async def one(_index: int) -> dict:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            return await ws_session(session, server, wav, audio_seconds, pool_seconds)

    started = time.perf_counter()
    results = await asyncio.gather(*(one(i) for i in range(level)))
    return summarize(level, list(results), time.perf_counter() - started)


def audio_duration_seconds(audio_path: Path) -> float:
    if audio_path.suffix.lower() == ".wav":
        import wave

        with wave.open(str(audio_path)) as handle:
            return handle.getnframes() / handle.getframerate()
    out = __import__("subprocess").check_output(
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


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--audio", default="test_data/speaker/meeting-test.wav")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--pool-seconds", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    wav = Path(args.audio)
    if not wav.exists():
        raise SystemExit(f"audio not found: {wav}")
    audio_seconds = audio_duration_seconds(wav)
    print(
        f"server={args.server} audio={wav.name} duration={audio_seconds:.1f}s "
        f"(WS streaming lane, safe increments)",
        flush=True,
    )
    for level in args.levels:
        row = await run_level(
            args.server, wav, audio_seconds, level, args.pool_seconds, args.timeout
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if row["failed"]:
            print(f"stopping: level {level} had failures (safe mode)", flush=True)
            break


if __name__ == "__main__":
    asyncio.run(main_async())
