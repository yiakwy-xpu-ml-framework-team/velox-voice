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
import asyncio
import json
import math
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
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
    ) -> None:
        self.vx = vx

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

                result = await asyncio.to_thread(self.vx.transcribe_pcm, pcm)

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

            result = await asyncio.to_thread(self.vx.transcribe_pcm, pcm)

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
def create_app(
    model_dir: str | Path,
    device: str = "cuda",
    precision: str = "bf16",
    enable_graphs: bool = True,
    pool_seconds: float = 20.0,
    upload_dir: str | Path = UPLOAD_DIR,
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

    @app.on_event("startup")
    async def _startup() -> None:
        await asyncio.to_thread(
            lambda: vx.warmup((0.5, 1.0, 5.0, 10.0, pool_seconds, 30.0))
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
    ) -> Response:
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
                            vx.transcribe_pcm, buffer[:target]
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
        result = await asyncio.to_thread(vx.transcribe_pcm, pcm)
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

    @app.get("/client", include_in_schema=False, response_class=HTMLResponse)
    async def client() -> HTMLResponse:
        return HTMLResponse(_CLIENT_HTML)

    @app.get("/sitemap.xml")
    async def sitemap(request: Request) -> Response:
        base = str(request.base_url).rstrip("/")
        urls = ["/docs", "/api"]
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
        return JSONResponse(
            {
                "name": "VeloxVoice ASR",
                "docs": "/docs",
                "openapi": "/openapi.json",
                "routes": {
                    "health": "/health",
                    "models": "/v1/models",
                    "client": "/client",
                    "sitemap": "/sitemap.xml",
                    "transcriptions": "/v1/audio/transcriptions",
                    "uploads": "/v1/audio/uploads",
                    "websocket": "/v1/audio/ws",
                },
                "quickstart": {
                    "method": "POST",
                    "url": "/v1/audio/transcriptions",
                    "content_type": "multipart/form-data",
                    "required_field": "file",
                    "streaming_field": "stream",
                },
            }
        )

    return app


_CLIENT_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>VeloxVoice ASR Client</title>
  <style>
    :root { color-scheme: light dark; }
    body { font: 15px/1.5 system-ui,sans-serif; max-width: 900px; margin: 30px auto; }
    h1 { font-size: 22px; margin-bottom: 14px; }
    .card { background: #f7f7f8; border-radius: 12px; padding: 18px; }
    .drop { border: 2px dashed #bbb; border-radius: 10px; padding: 22px; text-align: center; cursor: pointer; }
    .drop.drag { border-color: #10a37f; background: #e8f7f2; }
    button { font-size: 15px; padding: 8px 14px; border-radius: 8px; border: 0; background: #10a37f; color: white; cursor: pointer; }
    button.live { background: #7c3aed; }
    button:disabled { opacity: .5; cursor: not-allowed; }
    input[type=file] { display: none; }
    select { padding: 6px; border-radius: 6px; }
    .actions { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
    .progress-label { display: flex; justify-content: space-between; gap: 10px; color: #555; font-size: 13px; margin-top: 6px; }
    .progress { height: 6px; background: #ddd; border-radius: 3px; overflow: hidden; margin-top: 12px; }
    .progress > div { height: 100%; width: 0%; background: #10a37f; transition: width .2s; }
    #asrBar { background: #7c3aed; }
    pre { white-space: pre-wrap; background: #0f172a; color: #d1d5db; padding: 14px; border-radius: 8px; min-height: 150px; }
    .meta { color: #666; font-size: 13px; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>VeloxVoice ASR Client</h1>
  <div class="card">
    <div class="drop" id="drop">
      <b>Select audio</b><br>
      <small>MP3, WAV, FLAC, M4A, OGG, MP4…</small>
    </div>
    <input type="file" id="file" accept="audio/*,video/*" />
    <div class="actions">
      <select id="pool">
        <option value="5" selected>5s pool</option>
        <option value="10">10s pool</option>
        <option value="20">20s pool</option>
        <option value="30">30s pool</option>
      </select>
      <button id="uploadBtn" disabled>Upload</button>
      <button id="transcribeBtn" disabled>Transcribe (after uploading)</button>
      <button id="webrtcBtn" class="live" disabled>Transcribing (WebRTC)</button>
    </div>

    <div class="progress"><div id="uploadBar"></div></div>
    <div class="progress-label"><span id="uploadLabel">Upload</span><span id="uploadStatus">Idle</span></div>

    <div class="progress"><div id="asrBar"></div></div>
    <div class="progress-label"><span id="asrLabel">Transcribing</span><span id="asrStatus">Idle</span></div>

    <pre id="out">Waiting for audio…</pre>
    <div class="meta">
      API docs: <code>/docs</code>. Live streaming: <code>/v1/audio/ws</code>.
      Live mode emits a short 1s first partial, then uses the selected pool cadence.
    </div>
  </div>
<script>
const fileInput = document.querySelector('#file');
const drop = document.querySelector('#drop');
const poolSel = document.querySelector('#pool');
const uploadBtn = document.querySelector('#uploadBtn');
const transcribeBtn = document.querySelector('#transcribeBtn');
const webrtcBtn = document.querySelector('#webrtcBtn');
const out = document.querySelector('#out');
const uploadBar = document.querySelector('#uploadBar');
const asrBar = document.querySelector('#asrBar');
const uploadLabel = document.querySelector('#uploadLabel');
const asrLabel = document.querySelector('#asrLabel');
const uploadStatus = document.querySelector('#uploadStatus');
const asrStatus = document.querySelector('#asrStatus');
let selectedFile = null;
let uploadedId = null;
let audioDuration = 0;
let busy = false;
let ws = null;

function logState(message) {
  console.log(`[VeloxVoice] ${message}`);
}

function seconds(value) {
  return `${Number(value).toFixed(2)} s`;
}

function setProgress(bar, percent) {
  const value = Math.max(0, Math.min(100, Number(percent) || 0));
  bar.style.width = `${value}%`;
}

function setButtons() {
  uploadBtn.disabled = busy || !selectedFile;
  transcribeBtn.disabled = busy || !uploadedId;
  webrtcBtn.disabled = busy || !selectedFile;
}

function resetForFile(file) {
  selectedFile = file;
  uploadedId = null;
  audioDuration = 0;
  out.textContent = file ? 'Audio selected. Choose Upload or live WebRTC transcription.' : 'Waiting for audio…';
  setProgress(uploadBar, 0);
  uploadLabel.textContent = 'Upload';
  uploadStatus.textContent = file ? 'Selected' : 'Idle';
  setProgress(asrBar, 0);
  asrLabel.textContent = 'Transcribing';
  asrStatus.textContent = 'Idle';
  setButtons();
}

function loadDuration(file) {
  const media = document.createElement('audio');
  const url = URL.createObjectURL(file);
  media.preload = 'metadata';
  media.onloadedmetadata = () => {
    audioDuration = Number.isFinite(media.duration) ? media.duration : 0;
    URL.revokeObjectURL(url);
    logState(`Audio duration detected: ${seconds(audioDuration)}`);
  };
  media.onerror = () => {
    URL.revokeObjectURL(url);
    logState('Audio duration detection failed; live progress will use received audio seconds');
  };
  media.src = url;
}

function appendTranscript(text) {
  if (!text) return;
  if (out.textContent.startsWith('Waiting for audio') || out.textContent.startsWith('Audio selected')) {
    out.textContent = '';
  }
  out.appendChild(document.createTextNode(text));
  out.appendChild(document.createElement('br'));
}

function markTranscribed(wallSeconds, audioSeconds, processingSeconds, rtf) {
  setProgress(asrBar, 100);
  asrLabel.textContent = 'Transcribed';
  const processing = processingSeconds !== undefined
    ? `Server ${seconds(processingSeconds)}`
    : null;
  const rtfText = (rtf !== undefined && Number.isFinite(Number(rtf)))
    ? `RTF ${Number(rtf).toFixed(5)}`
    : null;
  asrStatus.textContent = `Completed in ${seconds(wallSeconds)}`;
  if (audioSeconds !== undefined) {
    asrStatus.textContent += ` / Audio ${seconds(audioSeconds)}`;
  }
  if (processing) {
    asrStatus.textContent += ` / ${processing}`;
  }
  if (rtfText) {
    asrStatus.textContent += ` / ${rtfText}`;
  }
  logState(`Transcription complete: wall=${seconds(wallSeconds)}, audio=${seconds(audioSeconds || 0)}${processing ? `, server=${processing}` : ''}${rtfText ? `, ${rtfText}` : ''}`);
}

fileInput.onchange = () => {
  const file = fileInput.files[0];
  resetForFile(file);
  if (file) {
    loadDuration(file);
    logState(`Audio selected: ${file.name} (${file.size} bytes)`);
  } else {
    logState('Audio selection cleared');
  }
};

drop.onclick = () => fileInput.click();
drop.ondragover = ev => {
  ev.preventDefault();
  drop.classList.add('drag');
};
drop.ondragleave = () => drop.classList.remove('drag');
drop.ondrop = ev => {
  ev.preventDefault();
  drop.classList.remove('drag');
  const file = ev.dataTransfer.files[0];
  resetForFile(file);
  if (file) {
    loadDuration(file);
    logState(`Audio selected: ${file.name} (${file.size} bytes)`);
  }
};

function inputFormat(file) {
  const type = (file.type || '').toLowerCase();
  const name = (file.name || '').toLowerCase();
  if (type === 'audio/mpeg' || name.endsWith('.mp3')) return 'mp3';
  if (type === 'audio/wav' || type === 'audio/x-wav' || name.endsWith('.wav')) return 'wav';
  if (type === 'audio/flac' || name.endsWith('.flac')) return 'flac';
  if (type === 'audio/ogg' || name.endsWith('.ogg') || name.endsWith('.opus')) return 'ogg';
  return 'auto';
}

function uploadWithProgress() {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append('file', selectedFile);
    const xhr = new XMLHttpRequest();
    const startedAt = performance.now();
    xhr.open('POST', '/v1/audio/uploads');
    xhr.responseType = 'json';
    xhr.upload.onprogress = ev => {
      if (!ev.lengthComputable) return;
      const percent = 100 * ev.loaded / ev.total;
      setProgress(uploadBar, percent);
      uploadStatus.textContent = `Uploading ${Math.round(percent)}%`;
      logState(`Upload progress: ${Math.round(percent)}%`);
    };
    xhr.onload = () => {
      const elapsed = (performance.now() - startedAt) / 1000;
      if (xhr.status >= 200 && xhr.status < 300 && xhr.response && xhr.response.id) {
        setProgress(uploadBar, 100);
        uploadLabel.textContent = 'Upload';
        uploadStatus.textContent = `Uploaded in ${seconds(elapsed)}`;
        logState(`Upload complete in ${seconds(elapsed)}: id=${xhr.response.id}, path=${xhr.response.audio_path}`);
        resolve(xhr.response);
      } else {
        reject(new Error(xhr.response && xhr.response.detail ? xhr.response.detail : `Upload failed (HTTP ${xhr.status})`));
      }
    };
    xhr.onerror = () => reject(new Error('Upload network error'));
    xhr.send(form);
  });
}

uploadBtn.onclick = async () => {
  if (!selectedFile || busy) return;
  busy = true;
  setButtons();
  logState('Upload started');
  try {
    const result = await uploadWithProgress();
    uploadedId = result.id;
  } catch (err) {
    setProgress(uploadBar, 0);
    uploadStatus.textContent = 'Upload failed';
    console.error(`[VeloxVoice] Upload error: ${err.message}`);
    alert(err.message);
  } finally {
    busy = false;
    setButtons();
  }
};

async function readSse(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop();
    for (const event of events) {
      const dataLine = event.split(/\r?\n/).find(line => line.startsWith('data:'));
      if (dataLine) await onEvent(dataLine.slice(5).trim());
    }
  }
}

transcribeBtn.onclick = async () => {
  if (!uploadedId || busy) return;
  busy = true;
  setButtons();
  out.textContent = '';
  setProgress(asrBar, 0);
  asrLabel.textContent = 'Transcribing';
  asrStatus.textContent = 'Starting';
  logState('Post-upload transcription started');
  const startedAt = performance.now();
  const form = new FormData();
  form.append('audio_id', uploadedId);
  form.append('response_format', 'json');
  form.append('stream', 'true');
  try {
    const response = await fetch('/v1/audio/transcriptions', { method: 'POST', body: form });
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new Error(detail && detail.detail ? detail.detail : `Transcription failed (HTTP ${response.status})`);
    }
    await readSse(response, async data => {
      if (data === '[DONE]') return;
      const obj = JSON.parse(data);
      if (obj.type === 'transcript.text.delta') {
        appendTranscript(obj.delta);
        const percent = obj.progress !== undefined ? obj.progress * 100 : 0;
        setProgress(asrBar, percent);
        asrStatus.textContent = `Transcribing ${Math.round(percent)}%`;
        logState(`Transcription progress: ${Math.round(percent)}%`);
      } else if (obj.type === 'transcript.text.done') {
        out.textContent = obj.text;
        markTranscribed(
          (performance.now() - startedAt) / 1000,
          obj.audio_seconds,
          obj.elapsed_seconds,
          obj.rtf,
        );
      } else if (obj.type === 'error') {
        throw new Error(obj.error && obj.error.message ? obj.error.message : 'Transcription failed');
      }
    });
  } catch (err) {
    setProgress(asrBar, 0);
    asrLabel.textContent = 'Transcribing';
    asrStatus.textContent = 'Failed';
    out.textContent = `Error: ${err.message}`;
    console.error(`[VeloxVoice] Transcription error: ${err.message}`);
  } finally {
    busy = false;
    setButtons();
  }
};

webrtcBtn.onclick = async () => {
  if (!selectedFile || busy) return;
  busy = true;
  setButtons();
  out.textContent = '';
  setProgress(uploadBar, 0);
  uploadLabel.textContent = 'Upload';
  uploadStatus.textContent = 'Connecting';
  setProgress(asrBar, 0);
  asrLabel.textContent = 'Transcribing';
  asrStatus.textContent = 'Starting';
  const liveStart = performance.now();
  let uploadStart = performance.now();
  logState('Live WebRTC transcription started');
  const wsUrl = `${location.origin.replace(/^http/, 'ws')}/v1/audio/ws`;
  ws = new WebSocket(wsUrl);
  try {
    await new Promise((resolve, reject) => {
      ws.onopen = () => {
        logState('Live streaming connection opened');
        uploadStart = performance.now();
        uploadStatus.textContent = 'Uploading';
        ws.send(JSON.stringify({
          type: 'start',
          pool_seconds: Number(poolSel.value),
          format: inputFormat(selectedFile),
          filename: selectedFile.name,
        }));
        resolve();
      };
      ws.onerror = () => {
        uploadStatus.textContent = 'Failed';
        asrStatus.textContent = 'Failed';
        console.error('[VeloxVoice] Live streaming connection error');
        reject(new Error('Live streaming connection failed'));
      };
    });
  } catch (err) {
    busy = false;
    setButtons();
    return;
  }

  ws.onmessage = ev => {
    const obj = JSON.parse(ev.data);
    if (obj.type === 'partial') {
      appendTranscript(obj.text);
      if (audioDuration > 0) {
        const percent = 100 * obj.audio_seconds / audioDuration;
        setProgress(asrBar, percent);
        asrStatus.textContent = `Transcribing ${Math.round(percent)}%`;
        logState(`Live transcription progress: ${Math.round(percent)}%`);
      } else {
        asrStatus.textContent = `Transcribed ${seconds(obj.audio_seconds)}`;
        logState(`Live transcription progress: ${seconds(obj.audio_seconds)} received`);
      }
    } else if (obj.type === 'final') {
      out.textContent = obj.text;
      markTranscribed(
        (performance.now() - liveStart) / 1000,
        obj.audio_seconds,
        obj.elapsed_seconds,
        obj.rtf,
      );
    } else if (obj.type === 'error') {
      out.textContent = `Error: ${obj.detail}`;
      setProgress(asrBar, 0);
      asrLabel.textContent = 'Transcribing';
      asrStatus.textContent = 'Failed';
      console.error(`[VeloxVoice] Live transcription error: ${obj.detail}`);
    }
  };
  ws.onclose = () => {
    logState('Live streaming connection closed');
    busy = false;
    setButtons();
  };
  ws.onerror = () => {
    uploadStatus.textContent = 'Failed';
    asrStatus.textContent = 'Failed';
    console.error('[VeloxVoice] Live streaming connection error');
  };

  const chunkSize = 1024 * 1024 * 4;
  let offset = 0;
  try {
    while (offset < selectedFile.size && ws.readyState === 1) {
      const buf = await selectedFile.slice(offset, offset + chunkSize).arrayBuffer();
      ws.send(buf);
      offset += buf.byteLength;
      const percent = 100 * offset / selectedFile.size;
      setProgress(uploadBar, percent);
      uploadStatus.textContent = `Uploading ${Math.round(percent)}%`;
      logState(`Live upload progress: ${Math.round(percent)}%`);
    }
    if (ws.readyState === 1) {
      ws.send(JSON.stringify({ type: 'stop' }));
      setProgress(uploadBar, 100);
      uploadStatus.textContent = `Uploaded in ${seconds((performance.now() - uploadStart) / 1000)}`;
      logState(`Live upload complete in ${seconds((performance.now() - uploadStart) / 1000)}; waiting for final transcription`);
    }
  } catch (err) {
    console.error(`[VeloxVoice] Live upload error: ${err.message}`);
  }
};
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Run VeloxVoice SGLang-Omni ASR server")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--pool-seconds", type=float, default=20.0)
    ap.add_argument("--no-graphs", action="store_true")
    args = ap.parse_args()

    import uvicorn

    app = create_app(
        model_dir=args.model_dir,
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
