"""SGLang-Omni-compatible ASR API.

Routes
------
* ``GET  /``                                redirect to OpenAPI docs
* ``GET  /docs``                            Swagger UI
* ``GET  /client``                          optional browser test client
* ``GET  /openapi.json``                    machine-readable API schema
* ``GET  /health``                          health testing
* ``GET  /v1/models``                       model list
* ``POST /v1/audio/transcriptions``         multipart transcription/SSE
* ``POST /v1/audio/uploads``                upload-only staging
* ``WS   /v1/audio/ws``                     websocket streaming transcription
"""

from __future__ import annotations

import argparse
import os
import asyncio
import json
import math
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from ..api import TranscriptionResult, Velox

SAMPLE_RATE = 16000

UPLOAD_DIR = Path("logs/user_data")
TRANSCRIPTION_RESPONSE_FORMATS = frozenset(
    {"json", "text", "verbose_json", "srt", "vtt"}
)
STREAMING_RESPONSE_FORMATS = frozenset({"json", "text"})


def _safe_filename(filename: str | None, fallback: str = "upload.wav") -> str:
    name = Path(filename or fallback).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or fallback


def _save_upload(data: bytes, filename: str | None, upload_dir: Path) -> Path:
    """Persist an uploaded audio file under logs/user_data/YYYYMMDD/."""
    day_dir = upload_dir / time.strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    path = day_dir / f"{stamp}_{uuid4().hex[:8]}_{_safe_filename(filename)}"
    path.write_bytes(data)
    return path


def _write_result_metadata(path: Path, result: TranscriptionResult) -> None:
    metadata = path.with_suffix(path.suffix + ".json")
    metadata.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _decode_audio_bytes(data: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".upload") as tmp:
        tmp.write(data)
        tmp.flush()
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                tmp.name,
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-f",
                "f32le",
                "-",
            ],
            stdout=subprocess.PIPE,
            check=True,
        )
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _offset_spans(spans: list[list[int]], frame_offset: int) -> list[list[int]]:
    return [[a + frame_offset, b + frame_offset] for a, b in spans]


def _join_text(parts: list[str]) -> str:
    """Join ASR pools without inserting spaces between CJK tokens."""
    out = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not out:
            out = part
        elif out[-1].isspace() or part[0].isspace():
            out += part
        elif any(0x4E00 <= ord(ch) <= 0x9FFF for ch in (out[-1], part[0])):
            out += part
        else:
            out += " " + part
    return out


def _duration_usage(audio_seconds: float) -> dict[str, Any] | None:
    if audio_seconds <= 0:
        return None
    return {"type": "duration", "seconds": math.ceil(audio_seconds)}


def _format_subtitle_timestamp(
    seconds: float, *, always_hours: bool, decimal: str
) -> str:
    milliseconds = round(max(0.0, seconds) * 1000.0)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    hours_text = f"{hours:02d}:" if always_hours or hours else ""
    return f"{hours_text}{minutes:02d}:{secs:02d}{decimal}{milliseconds:03d}"


def _subtitle_text(text: str) -> str:
    return text.strip().replace("-->", "->")


def _segments_for_result(result: TranscriptionResult) -> list[dict[str, Any]]:
    return [
        {
            "id": 0,
            "start": 0.0,
            "end": max(0.0, result.audio_seconds),
            "text": result.text,
        }
    ]


def _transcription_response(
    result: TranscriptionResult, response_format: str, language: str | None = None
) -> Response:

    result.text = result.text.strip()
    response_format = response_format.lower()

    if response_format == "text":
        return PlainTextResponse(result.text, media_type="text/plain; charset=utf-8")

    if response_format == "srt":
        segment = _segments_for_result(result)[0]

        start = _format_subtitle_timestamp(
            segment["start"], always_hours=True, decimal=","
        )

        end = _format_subtitle_timestamp(segment["end"], always_hours=True, decimal=",")
        content = f"1\n{start} --> {end}\n{_subtitle_text(result.text)}\n\n"

        return PlainTextResponse(
            content, media_type="application/x-subrip; charset=utf-8"
        )

    if response_format == "vtt":
        segment = _segments_for_result(result)[0]

        start = _format_subtitle_timestamp(
            segment["start"], always_hours=False, decimal="."
        )

        end = _format_subtitle_timestamp(
            segment["end"], always_hours=False, decimal="."
        )

        content = "WEBVTT\n\n" f"{start} --> {end}\n{_subtitle_text(result.text)}\n\n"

        return PlainTextResponse(content, media_type="text/vtt; charset=utf-8")

    usage = _duration_usage(result.audio_seconds)
    if response_format == "verbose_json":
        payload: dict[str, Any] = {
            "task": "transcribe",
            "text": result.text,
            "segments": _segments_for_result(result),
        }

        if language is not None:
            payload["language"] = language
        if result.audio_seconds > 0:
            payload["duration"] = result.audio_seconds
        if usage is not None:
            payload["usage"] = usage

        return JSONResponse(payload)

    payload = {"text": result.text}

    if result.audio_seconds > 0:
        payload["audio_seconds"] = result.audio_seconds
        payload["elapsed_seconds"] = result.elapsed_seconds
        payload["rtf"] = result.rtf
    if usage is not None:
        payload["usage"] = usage
    return JSONResponse(payload)


class _AudioStreamSession:
    """Decode a live audio stream and transcribe it in large pooled chunks.

    Incoming audio bytes are fed to stdin while decoded PCM is consumed from stdout.
    A full pool is transcribed while the client can continue uploading the remainder.
    """

    def __init__(
        self,
        vx: Velox,
        pool_seconds: float,
        input_format: str = "auto",
        sample_rate: int = SAMPLE_RATE,
        audio_path: Path | None = None,
        first_pool_seconds: float = 1.0,
        decode: str | None = None,
    ) -> None:
        self.vx = vx
        self.decode = decode

        self.pool_samples = max(1600, int(pool_seconds * sample_rate))

        self.first_pool_samples = max(
            1600, int(min(first_pool_seconds, pool_seconds) * sample_rate)
        )

        self._next_pool_samples = self.first_pool_samples

        self.input_format = input_format

        self.sample_rate = sample_rate

        self.audio_path = audio_path

        self.proc: asyncio.subprocess.Process | None = None

        self._pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        self._input_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=64)

        self._reader_task: asyncio.Task | None = None

        self._writer_task: asyncio.Task | None = None

        self._audio_file = None

        self._buffer = bytearray()

        self._closed = False

    async def start(self) -> None:
        if self.audio_path is not None:
            self.audio_path.parent.mkdir(parents=True, exist_ok=True)
            self._audio_file = self.audio_path.open("wb")

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

        if self.input_format != "auto":
            cmd += ["-f", self.input_format]
            if self.input_format in {"s16le", "f32le", "s24le", "s32le"}:
                cmd += ["-ar", str(self.sample_rate), "-ac", "1"]

        cmd += [
            "-i",
            "pipe:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "f32le",
            "-",
        ]

        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        # NOTE (yiakwy) : transcribing coroutine
        self._reader_task = asyncio.create_task(self._read_pcm())

        # NOTE (yiakwy) : reading audio coroutine
        self._writer_task = asyncio.create_task(self._write_stdin())

    async def write(self, data: bytes) -> None:
        if not self._writer_task or self._writer_task.done():
            raise RuntimeError("stream is not started")
        await self._input_queue.put(data)

    async def _write_stdin(self) -> None:
        if not self.proc or not self.proc.stdin:
            return
        stdin = self.proc.stdin
        try:
            while True:
                data = await self._input_queue.get()
                if data is None:
                    break
                if self._audio_file is not None:
                    self._audio_file.write(data)
                stdin.write(data)
                await stdin.drain()
        except Exception:
            pass
        finally:
            if stdin and not stdin.is_closing():
                stdin.close()

    async def finish(self) -> None:
        if self._audio_file is not None:
            self._audio_file.close()
            self._audio_file = None

        if self._writer_task is not None and not self._writer_task.done():
            try:
                await self._input_queue.put(None)
                await self._writer_task
            except Exception:
                pass

        if self.proc and self.proc.stdin and not self.proc.stdin.is_closing():
            self.proc.stdin.close()
            try:
                await asyncio.wait_for(self.proc.stdin.wait_closed(), 5.0)
            except Exception:
                pass

        try:
            await asyncio.wait_for(self.proc.wait(), 15.0)
        except Exception:
            await self.close()
        await self._reader_task

    async def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        if self._audio_file is not None:
            self._audio_file.close()
            self._audio_file = None

        if self._writer_task is not None and not self._writer_task.done():
            self._writer_task.cancel()
            try:
                await self._writer_task
            except Exception:
                pass

        if self.proc and self.proc.stdin and not self.proc.stdin.is_closing():
            self.proc.stdin.close()

        if self.proc and self.proc.returncode is None:
            self.proc.kill()
            await self.proc.wait()

        if self._reader_task is not None:
            await self._reader_task

    async def _read_pcm(self) -> None:
        assert self.proc and self.proc.stdout
        try:
            while True:
                chunk = await self.proc.stdout.read(65536)
                if not chunk:
                    break
                await self._pcm_queue.put(chunk)
        except Exception:
            pass
        finally:
            await self._pcm_queue.put(None)

    async def iter_results(self) -> AsyncIterator[tuple[TranscriptionResult, bool]]:
        """Yield a short first pool, then regular pools and one aggregate final."""
        results: list[TranscriptionResult] = []
        sample_offset = 0
        frame_offset = 0

        while True:
            chunk = await self._pcm_queue.get()

            if chunk is None:
                break

            self._buffer.extend(chunk)

            target_samples = self._next_pool_samples
            while len(self._buffer) >= target_samples * 4:
                raw = bytes(self._buffer[: target_samples * 4])

                del self._buffer[: target_samples * 4]

                pcm = np.frombuffer(raw, dtype=np.float32)

                result = await asyncio.to_thread(self.vx.transcribe_pcm, pcm, decode=self.decode)

                result.spans = _offset_spans(result.spans, frame_offset)
                results.append(result)
                sample_offset += pcm.size
                frame_offset = int(round(sample_offset / (self.sample_rate / 100)))
                self._next_pool_samples = self.pool_samples
                target_samples = self.pool_samples

                yield result, False

        if self._buffer:
            actual_samples = len(self._buffer) // 4
            pcm = np.frombuffer(bytes(self._buffer), dtype=np.float32)
            target_samples = self._next_pool_samples
            if pcm.size < target_samples:
                # Keep the final pool graph-shape stable with the warm pool.
                pcm = np.pad(pcm, (0, target_samples - pcm.size))

            result = await asyncio.to_thread(self.vx.transcribe_pcm, pcm, decode=self.decode)

            result.audio_seconds = actual_samples / self.sample_rate
            result.spans = _offset_spans(result.spans, frame_offset)
            results.append(result)
            sample_offset += actual_samples
            frame_offset = int(round(sample_offset / (self.sample_rate / 100)))
            self._next_pool_samples = self.pool_samples

            yield result, False

            self._buffer.clear()

        yield TranscriptionResult(
            text=_join_text([r.text for r in results]),
            ids=[t for r in results for t in r.ids],
            spans=[span for r in results for span in r.spans],
            audio_seconds=sum(r.audio_seconds for r in results),
            elapsed_seconds=sum(r.elapsed_seconds for r in results),
        ), bool(results)


# TODO (yiakwy) : move api and test server to different files
def _music_warmup_by_model(model, device):
    """Trigger cuDNN/allocator warmup for the music recognition AND prime the
    AR decode path (grammar FSM tables + weight pack + CUDA-graph capture).

    Without the decode warmup the first real request pays the one-time
    ``build_grammar_tables`` cost (~8 s) plus the graph capture."""
    import torch as _t

    dummy = _t.zeros(1, int(model.cfg.input_audio_length * 24000), device=device)
    with _t.inference_mode():
        mem = model.encode(dummy)
        try:
            from veloxvoice.models.sheetsage.v2 import api as sheetsage_api
            from veloxvoice.stream.grammar_ar_decode import FULL_TASK_PROMPTS

            prompts = model.tokenizer.normalize_prompts(FULL_TASK_PROMPTS)
            if sheetsage_api._decode_engine() == "pipeline" \
                    and getattr(model, "_graphed", None) is not None:
                from veloxvoice.models.sheetsage.v2.step_pipeline import (
                    transcribe_window_multi)

                transcribe_window_multi(model, dummy, prompts, 8, memory=mem)
            else:
                from veloxvoice.stream.grammar_ar_decode import (
                    build_grammar_tables)

                build_grammar_tables(model.tokenizer, prompts, str(device))
        except Exception:
            pass
    _t.cuda.synchronize()


def create_app(
    model_dir: str | Path,
    device: str = "cuda",
    precision: str = "bf16",
    enable_graphs: bool = True,
    pool_seconds: float = 20.0,
    upload_dir: str | Path = UPLOAD_DIR,
    music_model_dir: str | Path | None = None,
    base_model_dir: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(title="VeloxVoice ASR", version="0.1.0")
    app.state.model_name = Path(model_dir).name
    app.state.uploads = {}
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    vx = Velox.load(
        model_dir,
        backend="torch",
        device=device,
        native_kernels=True,
        kernel_precision=precision,
        enable_graphs=enable_graphs,
    )
    upload_dir = Path(upload_dir)
    is_music = getattr(vx, "kind", "") == "sheetsage2"

    # music recognition: load SheetSage2 + MERT-v2 in addition to ASR
    model_music = None
    if music_model_dir and base_model_dir:
        from veloxvoice.models.sheetsage.v2 import api as sheetsage_api

        model_music = sheetsage_api.load_model(
            str(music_model_dir), str(base_model_dir), "bf16", device)
        print(f"[VeloxVoice] music model loaded: {music_model_dir}")
    has_music = model_music is not None

    async def _do_music_transcribe(audio_path: str, out_dir: Path):
        from veloxvoice.models.sheetsage.v2 import api as sheetsage_api

        return await asyncio.to_thread(
            sheetsage_api.transcribe, model_music, audio_path,
            output_dir=str(out_dir), serving=True)

    @app.on_event("startup")
    async def _startup() -> None:
        await asyncio.to_thread(
            lambda: vx.warmup((0.5, 1.0, 5.0, 10.0, pool_seconds, 30.0))
        )
        if has_music:
            # schedule music warmup AFTER server starts accepting connections
            asyncio.get_event_loop().call_later(
                1.0,
                lambda: asyncio.ensure_future(
                    asyncio.to_thread(
                        lambda: _music_warmup_by_model(model_music, device)
                    )
                )
            )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "healthy", "running": True}

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "device": vx.platform.device,
            "precision": vx.precision,
            "graphs": vx.enable_graphs,
            "pool_seconds": pool_seconds,
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        model_name = app.state.model_name
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "veloxvoice",
                    "root": model_name,
                }
            ],
        }

    @app.post("/v1/audio/translations")
    async def translate_audio() -> None:
        raise HTTPException(
            status_code=400,
            detail="VeloxVoice does not support audio translation",
        )

    @app.post("/v1/audio/uploads")
    async def upload_audio(file: UploadFile = File(...)) -> JSONResponse:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
        audio_path = await asyncio.to_thread(
            _save_upload, data, file.filename, upload_dir
        )
        upload_id = uuid4().hex
        app.state.uploads[upload_id] = audio_path
        return JSONResponse(
            {
                "id": upload_id,
                "filename": file.filename,
                "audio_path": str(audio_path),
                "size": len(data),
            }
        )

    _has_asr = hasattr(vx, "warmup")  # VeloxMusic skips Velox.__init__

    if _has_asr:
      @app.post("/v1/audio/transcriptions")
      async def transcribe_file(
        file: UploadFile | None = File(default=None),
        audio_id: str | None = Form(default=None),
        model: str | None = Form(default=None),
        language: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        response_format: str = Form(default="json"),
        temperature: float | None = Form(default=None, ge=0.0, le=2.0),
        repetition_penalty: float | None = Form(default=None, gt=0.0),
        max_new_tokens: int | None = Form(default=None, ge=1),
        stream: bool = Form(default=False),
        rescore: bool | None = Form(default=None),
    ) -> Response:
        decode = None if rescore is None else ("att_rescore" if rescore else "greedy")
        response_format = response_format.strip().lower()
        if stream:
            if response_format not in STREAMING_RESPONSE_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "stream=true supports only response_format 'json' or "
                        f"'text', got {response_format!r}"
                    ),
                )
        elif response_format not in TRANSCRIPTION_RESPONSE_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported response_format for "
                    f"/v1/audio/transcriptions: {response_format!r}"
                ),
            )

        if file is not None:
            data = await file.read()
            if not data:
                raise HTTPException(
                    status_code=400, detail="Uploaded audio file is empty"
                )
            audio_path = await asyncio.to_thread(
                _save_upload, data, file.filename, upload_dir
            )
            upload_id = uuid4().hex
            app.state.uploads[upload_id] = audio_path
        elif audio_id is not None:
            audio_path = app.state.uploads.get(audio_id)
            if audio_path is None or not audio_path.exists():
                raise HTTPException(status_code=404, detail="Unknown audio_id")
            data = await asyncio.to_thread(audio_path.read_bytes)
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide multipart 'file' or a previously uploaded 'audio_id'",
            )

        def sse(payload: dict[str, Any]) -> str:
            return (
                "data: "
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                + "\n\n"
            )

        if stream:
            pcm = await asyncio.to_thread(_decode_audio_bytes, data)
            target = max(1600, int(pool_seconds * SAMPLE_RATE))
            total_seconds = pcm.size / SAMPLE_RATE
            request_id = f"transcription-{uuid4()}"

            async def generate() -> AsyncIterator[str]:
                buffer = np.empty(0, dtype=np.float32)
                results: list[TranscriptionResult] = []
                completed_seconds = 0.0
                try:
                    buffer = np.concatenate((buffer, pcm))
                    while buffer.size >= target:
                        result = await asyncio.to_thread(
                            vx.transcribe_pcm, buffer[:target], decode=decode
                        )
                        results.append(result)
                        completed_seconds += result.audio_seconds
                        if result.text:
                            yield sse(
                                {
                                    "type": "transcript.text.delta",
                                    "delta": result.text,
                                    "completed_seconds": completed_seconds,
                                    "total_seconds": total_seconds,
                                    "progress": (
                                        min(1.0, completed_seconds / total_seconds)
                                        if total_seconds
                                        else 1.0
                                    ),
                                }
                            )
                        buffer = buffer[target:]

                    if buffer.size:
                        result = await asyncio.to_thread(
                            vx.transcribe_pcm,
                            np.pad(buffer, (0, target - buffer.size)),
                            decode=decode,
                        )
                        result.audio_seconds = buffer.size / SAMPLE_RATE
                        results.append(result)
                        completed_seconds += result.audio_seconds
                        if result.text:
                            yield sse(
                                {
                                    "type": "transcript.text.delta",
                                    "delta": result.text,
                                    "completed_seconds": completed_seconds,
                                    "total_seconds": total_seconds,
                                    "progress": (
                                        min(1.0, completed_seconds / total_seconds)
                                        if total_seconds
                                        else 1.0
                                    ),
                                }
                            )

                    final = TranscriptionResult(
                        text=_join_text([r.text for r in results]),
                        ids=[token for r in results for token in r.ids],
                        spans=[span for r in results for span in r.spans],
                        audio_seconds=sum(r.audio_seconds for r in results),
                        elapsed_seconds=sum(r.elapsed_seconds for r in results),
                    )
                    await asyncio.to_thread(_write_result_metadata, audio_path, final)
                    payload: dict[str, Any] = {
                        "type": "transcript.text.done",
                        "text": final.text,
                        "audio_seconds": final.audio_seconds,
                        "elapsed_seconds": final.elapsed_seconds,
                        "rtf": final.rtf,
                    }
                    usage = _duration_usage(final.audio_seconds)
                    if usage is not None:
                        payload["usage"] = usage
                    yield sse(payload)
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    yield sse(
                        {
                            "type": "error",
                            "error": {"message": str(exc)},
                        }
                    )

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Request-Id": request_id,
                },
            )

        pcm = await asyncio.to_thread(_decode_audio_bytes, data)
        result = await asyncio.to_thread(vx.transcribe_pcm, pcm, decode=decode)
        result.text = result.text.strip()
        await asyncio.to_thread(_write_result_metadata, audio_path, result)
        return _transcription_response(result, response_format, language)

    @app.websocket("/v1/audio/ws")
    async def ws_stream(
        ws: WebSocket, pool_seconds: float = 20.0, input_format: str = "auto"
    ):
        await ws.accept()
        session: _AudioStreamSession | None = None
        results_task: asyncio.Task | None = None
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") and session is not None:
                    await session.write(msg["bytes"])
                elif msg.get("text"):
                    try:
                        obj = json.loads(msg["text"])
                    except Exception:
                        continue
                    if obj.get("type") == "start" and session is None:
                        filename = obj.get("filename")
                        day_dir = upload_dir / time.strftime("%Y%m%d")
                        day_dir.mkdir(parents=True, exist_ok=True)
                        audio_path = day_dir / (
                            f"{time.strftime('%H%M%S')}_{uuid4().hex[:8]}_"
                            f"{_safe_filename(filename)}"
                        )
                        session = _AudioStreamSession(
                            vx,
                            float(obj.get("pool_seconds", pool_seconds)),
                            str(obj.get("format", input_format)),
                            int(obj.get("sample_rate", SAMPLE_RATE)),
                            audio_path=audio_path,
                            decode=("att_rescore" if bool(obj.get("rescore")) else None),
                        )
                        await session.start()

                        async def drain_and_send() -> None:
                            try:
                                async for result, is_final in session.iter_results():
                                    payload = result.as_dict()
                                    payload["type"] = "final" if is_final else "partial"
                                    if is_final:
                                        payload["audio_path"] = str(audio_path)
                                        await asyncio.to_thread(
                                            _write_result_metadata,
                                            audio_path,
                                            result,
                                        )
                                    await ws.send_json(payload)
                            except Exception as exc:
                                await ws.send_json(
                                    {"type": "error", "detail": str(exc)}
                                )

                        results_task = asyncio.create_task(drain_and_send())
                    elif obj.get("type") == "stop":
                        break
        finally:
            if session is not None:
                await session.finish()
                if results_task is not None:
                    try:
                        await results_task
                    except Exception:
                        pass
                if session.proc is not None and session.proc.returncode:
                    try:
                        await ws.send_json(
                            {
                                "type": "error",
                                "detail": f"audio decoder exited with code {session.proc.returncode}",
                            }
                        )
                    except Exception:
                        pass
                await session.close()
            await ws.close()

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs", status_code=307)

    # ── music recognition endpoints ──────────────────────────────────────────────
    _music_output_dir = Path("/tmp/veloxvoice_music_output")
    _lyrics_jobs: set[str] = set()

    @app.post("/v1/audio/music/upload")
    async def music_upload(file: UploadFile = File(...)) -> JSONResponse:
        """Upload a song for music transcription."""
        data = await file.read()
        if not data:
            raise HTTPException(400, "Uploaded audio file is empty")
        upload_id = uuid4().hex[:12]
        music_dir = _music_output_dir / upload_id
        music_dir.mkdir(parents=True, exist_ok=True)
        audio_path = music_dir / (file.filename or "song.mp3")
        audio_path.write_bytes(data)
        return JSONResponse({
            "upload_id": upload_id,
            "filename": file.filename,
            "audio_path": str(audio_path),
            "size": len(data),
        })

    @app.post("/v1/audio/music/transcriptions")
    async def music_transcribe(upload_id: str = Form(...)):
        """Music audio to lead sheet (ABC + SVG score + MIDI)."""
        music_dir = _music_output_dir / upload_id
        if not music_dir.exists():
            raise HTTPException(404, f"Upload {upload_id} not found")

        audio_files = [f for f in music_dir.iterdir()
                       if f.suffix.lower() in (".mp3", ".wav", ".flac", ".ogg", ".m4a")]
        
        if not audio_files:
            raise HTTPException(404, "No audio file in upload")

        audio_path = audio_files[0]

        _t0 = time.monotonic()
        result = await _do_music_transcribe(str(audio_path), music_dir)
        _t1 = time.monotonic()
        print(f"[music-timing] model: {(_t1-_t0)*1000:.0f}ms")

        _t2 = time.monotonic()
        print(f"[music-timing] response build: {(_t2-_t1)*1000:.0f}ms")

        svg_pages = []
        for f in sorted(music_dir.glob("score_*.svg")):
            svg_pages.append({
                "page": int(f.stem.split("_")[1]),
                "url": f"/v1/audio/music/score/{music_dir.name}/{f.name}",
            })
        
        abc_text = None
        abc_file = music_dir / "score.abc"
        if abc_file.exists():
            abc_text = abc_file.read_text()

        elapsed = result.get("elapsed_seconds", 0)
        duration_s = result.get("duration_seconds", 0)

        _base = "/v1/audio/music/score/" + music_dir.name

        # see TEST_API.md
        return JSONResponse({
            "abc": abc_text,
            "abc_measures": result.get("abc_measures", 0),
            "abc_error": result.get("abc_error"),
            "svg_pages": svg_pages,
            "pages_pending": not svg_pages,
            "pages_list_url": _base + "/pages",
            "num_events": result.get("num_events", 0),
            "duration_seconds": duration_s,
            "output_dir": str(music_dir),
            "session": music_dir.name,
            "audio_url": _base + "/" + audio_path.name,
            "playback_url": _base + "/playback.json" if (music_dir / "playback.json").exists() else None,
            "pages_url": _base + "/score_pages.json" if (music_dir / "score_pages.json").exists() else None,
            "midi_files": {
                "full": _base + "/transcription.mid",
                "vocal": _base + "/melody_vocal.mid",
                "instrumental": _base + "/melody_instrumental.mid",
                "chords": _base + "/chords.mid",
            },
            "piano_wav": _base + "/piano/mix",
            "piano_wavs": {
                part: _base + "/piano/" + part for part in ("mix", "vocal", "instrumental")
            },
            "processing": {
                "elapsed_seconds": round(elapsed, 3),
                "audio_duration_seconds": round(duration_s, 3),
                "rtf": round(elapsed / duration_s, 5) if duration_s > 0 else None,
                "wall_time_seconds": round(elapsed, 3),
                "num_tokens": len(result.get("tokens", [])[0]) if result.get("tokens") else 0,
                "num_windows": len(result.get("tokens", [])),
            },
        })

    def _ensure_playback_bundle(music_dir: Path) -> None:
        """Rebuild the playback MIDI bundle from melody.mid + chord.lab when absent"""
        if (music_dir / "transcription.mid").exists():
            return
        if not (music_dir / "melody.mid").exists():
            raise HTTPException(404, "No transcription for this session")
        duration = None
        result_path = music_dir / "result.json"
        if result_path.exists():
            try:
                duration = json.loads(result_path.read_text(
                    encoding="utf-8")).get("duration_seconds")
            except Exception:
                duration = None
        from veloxvoice.audio.music_midi import export_playback

        export_playback(music_dir, None, duration)

    @app.get("/v1/audio/music/score/{session}/pages")
    async def music_score_pages(session: str):
        """Render all score pages on first call; return the page list JSON.

        Laziness keeps transcribe fast: the (multi-second) playwright render
        happens in the background while the client already shows ABC/MIDI.
        """
        music_dir = _music_output_dir / session
        if not music_dir.exists():
            raise HTTPException(404, f"Session {session} not found")

        def _render():
            from veloxvoice.audio.music_rendering import render_outputs

            render_assets = Path(__file__).resolve().parents[2] / "render_assets"
            render_outputs(input_dir=music_dir, output_dir=music_dir,
                           score="svg", audio=False,
                           assets_dir=str(render_assets) if render_assets.exists() else None)
        
        if not list(music_dir.glob("score_*.svg")):
            await asyncio.to_thread(_render)

        pages = []
        for f in sorted(music_dir.glob("score_*.svg")):
            pages.append({
                "page": int(f.stem.split("_")[1]),
                "url": f"/v1/audio/music/score/{session}/{f.name}",
            })
        return JSONResponse({"svg_pages": pages})

    @app.get("/v1/audio/music/score/{session}/piano/{part}")
    async def music_score_piano(session: str, part: str):
        """Piano-render one part on demand."""
        if part not in ("mix", "vocal", "instrumental"):
            raise HTTPException(404, "Unknown part")
        
        music_dir = _music_output_dir / session

        target = music_dir / f"piano_{part}.wav"
        if not target.exists():
            try:
                await asyncio.to_thread(_ensure_playback_bundle, music_dir)
            except HTTPException:
                raise
            midi = music_dir / "transcription.mid"
            if not midi.exists():
                raise HTTPException(404, "No transcription for this session")
            
            def _render():
                from veloxvoice.audio.music_rendering import render_outputs

                render_assets = Path(__file__).resolve().parents[2] / "render_assets"
                render_outputs(input_dir=music_dir, output_dir=music_dir,
                               audio=True, parts=(part,),
                               assets_dir=str(render_assets) if render_assets.exists() else None)
            
            await asyncio.to_thread(_render)
        
        if not target.exists():
            raise HTTPException(404, f"No {part} notes in this transcription")
        return FileResponse(target, media_type="audio/wav")

    @app.post("/v1/audio/music/lyrics")
    async def music_lyrics_start(session: str = Form(...)):
        """Start the background lyrics recognition:
        
        demucs separation + ASR on the vocal stem.
        """
        music_dir = _music_output_dir / session
        audio_files = ([f for f in music_dir.iterdir()
                        if f.suffix.lower() in (".mp3", ".wav", ".flac", ".ogg", ".m4a")]
                       if music_dir.exists() else [])
        if not audio_files:
            raise HTTPException(404, f"Session {session} not found")
        if session in _lyrics_jobs:
            return JSONResponse({"status": "running"})
        _lyrics_jobs.add(session)
        threading.Thread(target=_lyrics_job, args=(session, audio_files[0]),
                         daemon=True).start()
        return JSONResponse({"status": "started"})

    @app.get("/v1/audio/music/lyrics/{session}")
    async def music_lyrics_get(session: str):
        path = _music_output_dir / session / "lyrics.json"
        if not path.exists():
            return JSONResponse({"status": "idle"})
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    def _lyrics_job(session: str, audio_path: Path) -> None:
        out = _music_output_dir / session / "lyrics.json"

        def write(payload: dict) -> None:
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                           encoding="utf-8")

        t0 = time.monotonic()
        try:
            write({"status": "separating"})
            separate = _load_separate_fn()
            vocals, _instrumental = separate(audio_path, _music_output_dir / session,
                                             device="cuda")
            separation_s = time.monotonic() - t0
            write({"status": "transcribing",
                   "separation_seconds": round(separation_s, 1)})
            t1 = time.monotonic()
            text = _asr_vocals(vocals)
            asr_s = time.monotonic() - t1
            write({"status": "done", "lyrics": text,
                   "separation_seconds": round(separation_s, 1),
                   "asr_seconds": round(asr_s, 1),
                   "total_seconds": round(time.monotonic() - t0, 1)})
        except Exception as exc:  # noqa: BLE001 — surfaced to the client
            write({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            _lyrics_jobs.discard(session)

    def _load_separate_fn():
        import importlib.util

        tool = Path(__file__).resolve().parents[2] / "tools" / "music" / "separate_vocals.py"
        spec = importlib.util.spec_from_file_location("velox_separate_vocals", tool)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.separate

    def _asr_vocals(vocals_path: Path) -> str:
        """Transcribe the vocal stem with the ASR (16 kHz mono PCM)."""
        import soundfile as sf

        from veloxvoice.api import _resample_pcm16k

        pcm, sr = sf.read(str(vocals_path), dtype="float32", always_2d=True)
        pcm = np.asarray(pcm.mean(axis=1), dtype=np.float32).reshape(-1)
        if sr != 16000:
            pcm = _resample_pcm16k(pcm, sr)
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if pcm.size > 16000 * 60 * 12:
            pcm = pcm[:16000 * 60 * 12]  # cap at 12 minutes for the demo
        return vx.transcribe_pcm(pcm).text

    @app.get("/v1/audio/music/score/{session}/render/{page_num}")
    async def music_score_render(session: str, page_num: int):
        """Render ONE page of sheet music as SVG (on-demand, lazy)."""
        music_dir = _music_output_dir / session
        svg_file = music_dir / f"score_{page_num:03d}.svg"
        if svg_file.exists():
            return FileResponse(svg_file, media_type="image/svg+xml")

        # render this page (playwright)
        render_assets = Path(__file__).resolve().parents[4] / "render_assets"
        if not render_assets.exists():
            render_assets = Path(__file__).parent / "render_assets"

        def _render():
            from veloxvoice.audio.music_rendering import render_outputs
            render_outputs(
                input_dir=music_dir, output_dir=music_dir,
                score="svg", audio=False,
                assets_dir=str(render_assets))

        await asyncio.to_thread(_render)

        if svg_file.exists():
            return FileResponse(svg_file, media_type="image/svg+xml")
        raise HTTPException(404, f"Page {page_num} render failed")

    @app.get("/v1/audio/music/score/{session}/{filename}")
    async def music_score_file(session: str, filename: str):
        """Serve SVG/PNG/MIDI/WAV files from the music output directory."""
        safe = _safe_filename(filename)
        path = _music_output_dir / session / safe
        if not path.exists() and safe in ("transcription.mid", "chords.mid",
                                          "playback.json"):
            await asyncio.to_thread(_ensure_playback_bundle,
                                    _music_output_dir / session)
        if not path.exists():
            raise HTTPException(404, "File not found")
        media = {
            ".svg": "image/svg+xml", ".png": "image/png",
            ".mid": "audio/midi", ".wav": "audio/wav",
            ".abc": "text/plain", ".json": "application/json",
            ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
            ".flac": "audio/flac", ".ogg": "audio/ogg",
        }
        ext = path.suffix.lower()
        return FileResponse(path, media_type=media.get(ext, "application/octet-stream"))

    @app.get("/client", include_in_schema=False, response_class=HTMLResponse)
    async def client() -> HTMLResponse:
        return HTMLResponse(_CLIENT_HTML)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        # keep browser consoles clean: the tab icon is served inline
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
               "<text y='13' font-size='13'>🎵</text></svg>")
        return Response(svg, media_type="image/svg+xml")

    @app.get("/sitemap.xml")
    async def sitemap(request: Request) -> Response:
        base = str(request.base_url).rstrip("/")
        urls = ["/docs", "/api", "/client"]
        if has_music:
            urls += ["/v1/audio/music/transcriptions", "/v1/audio/music/upload"]
        items = "".join(
            f"<url><loc>{base}{path}</loc><changefreq>monthly</changefreq></url>"
            for path in urls
        )
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            + items
            + "</urlset>"
        )
        return Response(content=xml, media_type="application/xml")

    @app.get("/robots.txt")
    async def robots(request: Request) -> Response:
        base = str(request.base_url).rstrip("/")
        content = f"User-agent: *\nAllow: /\nSitemap: {base}/sitemap.xml\n"
        return Response(content=content, media_type="text/plain")

    @app.get("/api")
    async def api_index() -> JSONResponse:
        routes = {
            "health": "/health",
            "models": "/v1/models",
            "client": "/client",
            "transcriptions": "/v1/audio/transcriptions",
        }
        if has_music:
            routes["music_transcriptions"] = "/v1/audio/music/transcriptions"
        return JSONResponse(
            {
                "name": "VeloxVoice",
                "model_type": "asr+music" if has_music else "asr",
                "routes": routes,
                "docs": "/docs",
                "openapi": "/openapi.json",
            }
        )

    return app


_CLIENT_HTML_PATH = Path(__file__).parent / "templates" / "music_client.html"
_CLIENT_HTML = _CLIENT_HTML_PATH.read_text(encoding="utf-8") if _CLIENT_HTML_PATH.exists() else "<html><body>music_client.html not found</body></html>"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run VeloxVoice SGLang-Omni ASR server")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--base-model-dir", default="",
                    help="MERT-v2-FullSong encoder dir (music recognition)")
    ap.add_argument("--music-model-dir", default="",
                    help="SheetSage2 checkpoint dir (music recognition); "
                         "requires --base-model-dir")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--precision",
        default="bf16",
        choices=("fp16", "bf16", "fp32"),
        help="model precision; env VELOXVOICE_PRECISION overrides it "
             "(fp16=fp32 weights+autocast, fp32=high-precision, bf16=default)",
    )
    ap.add_argument("--pool-seconds", type=float, default=20.0)
    ap.add_argument("--no-graphs", action="store_true")
    args = ap.parse_args()

    import uvicorn

    app = create_app(
        model_dir=args.model_dir,
        music_model_dir=args.music_model_dir or None,
        base_model_dir=args.base_model_dir or None,
        device=args.device,
        precision=args.precision,
        enable_graphs=not args.no_graphs,
        pool_seconds=args.pool_seconds,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ws_per_message_deflate=False,
        ws_max_size=64 * 1024 * 1024,
        ws_max_queue=256,
    )


if __name__ == "__main__":
    main()
