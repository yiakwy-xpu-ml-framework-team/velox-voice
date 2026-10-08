#!/usr/bin/env python3
"""Round-robin load balancer in front of dynamically scaled VeloxVoice replicas.

One entry port (default 8000) serves BOTH the HTTP API and the test webpage
(`/client`, `/docs`, `/openapi.json`), plus the `WS /v1/audio/ws` streaming lane.

Session affinity — the ASR service is *stateful* for the two-step flow the
webpage uses:

    POST /v1/audio/uploads        -> {id}  (kept in that process's memory)
    POST /v1/audio/transcriptions -> audio_id=<id>  (must reach the SAME replica)

A plain round-robin breaks that second call with a 404, which is exactly what
the browser shows as "transcribing failed". This balancer therefore:

  * sets a sticky `velox_lb` cookie on first response and pins a browser to a
    replica for subsequent requests;
  * records `audio_id -> replica` from each upload response and routes the
    matching transcription back to it (works for cookie-less API clients too).

The backend is a Docker Compose service name (e.g. `asr:8000`); Compose's
embedded DNS resolves it to all replica IPs, re-resolved every `--ttl` seconds,
so the balancer follows `docker compose up --scale asr=N` automatically.

    python asr_lb.py --listen 0.0.0.0:8000 --backend asr:8000 --ttl 5
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import re
import socket
import time
from collections import OrderedDict

import aiohttp
from aiohttp import web

COOKIE = "velox_lb"
UPLOAD_JSON_CAP = 256 * 1024          # only buffer small upload JSON replies
AUDIO_ID_RE = re.compile(rb'name="audio_id"\r\n\r\n(?P<id>[^\r\n]+)')
_LOGGER = logging.getLogger("velox_lb")


class BackendPool:
    """Resolves a service name to its replica IPs."""

    def __init__(self, backend: str, ttl: float = 5.0):
        host, _, port = backend.rpartition(":")
        self.host = host or "127.0.0.1"
        self.port = int(port or 8000)
        self.ttl = float(ttl)
        self._ips: list[str] = []
        self._resolved_at = 0.0
        self._lock = asyncio.Lock()

    async def ips(self) -> list[str]:
        now = time.monotonic()
        if self._ips and now - self._resolved_at < self.ttl:
            return self._ips
        else:
            pass
        async with self._lock:
            now = time.monotonic()
            if self._ips and now - self._resolved_at < self.ttl:
                return self._ips
            else:
                pass
            infos = await asyncio.get_running_loop().getaddrinfo(
                self.host, self.port, type=socket.SOCK_STREAM)
            self._ips = sorted({info[4][0] for info in infos})
            self._resolved_at = time.monotonic()
            return self._ips


class Router:
    """Replica selection with cookie + audio_id affinity."""

    def __init__(self, pool: BackendPool, affinity_ttl: float = 600.0,
                 affinity_max: int = 10000):
        self.pool = pool
        self.affinity_ttl = affinity_ttl
        self.affinity_max = affinity_max
        self._affinity: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def remember(self, audio_id: str, ip: str) -> None:
        self._affinity[audio_id] = (ip, time.monotonic())
        self._affinity.move_to_end(audio_id)
        while len(self._affinity) > self.affinity_max:
            self._affinity.popitem(last=False)
        _LOGGER.info("remember upload id=%s -> replica %s", audio_id, ip)

    def recall(self, audio_id: str, ips: list[str]) -> str | None:
        entry = self._affinity.get(audio_id)
        if entry is None:
            return None
        ip, when = entry
        if ip in ips and time.monotonic() - when < self.affinity_ttl:
            _LOGGER.info("pin transcript id=%s -> replica %s", audio_id, ip)
            return ip
        self._affinity.pop(audio_id, None)
        return None

    async def choose(self, request: web.Request, counter,
                     audio_id: str | None, exclude: frozenset = frozenset()) -> str:
        all_ips = await self.pool.ips()
        if not all_ips:
            raise RuntimeError(f"no replicas resolved for {self.pool.host}")
        ips = [ip for ip in all_ips if ip not in exclude] or all_ips
        if audio_id:
            pinned = self.recall(audio_id, ips)
            if pinned:
                return pinned
        cookie = request.cookies.get(COOKIE)
        if cookie in ips:
            return cookie
        return ips[next(counter) % len(ips)]


def _forward_headers(request: web.Request) -> dict[str, str]:
    return {k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "connection", "keep-alive",
                                 "transfer-encoding", "content-length")}


def _audio_id_from_body(body: bytes | None) -> str | None:
    if not body:
        return None
    match = AUDIO_ID_RE.search(body)
    return match.group("id").decode("utf-8", "replace") if match else None


def _set_sticky_cookie(response: web.StreamResponse, request: web.Request,
                       ip: str) -> None:
    if request.cookies.get(COOKIE) != ip:
        response.set_cookie(COOKIE, ip, path="/", max_age=3600, httponly=False,
                            samesite="Lax")


async def _proxy_http(request: web.Request, pool: BackendPool, router: Router,
                      counter) -> web.StreamResponse:
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=3600)
    body = await request.read() if request.can_read_body else None
    is_upload = request.path == "/v1/audio/uploads"
    audio_id = None
    if request.path == "/v1/audio/transcriptions":
        audio_id = _audio_id_from_body(body)
    last_error: Exception | None = None
    tried: set[str] = set()
    for _ in range(max(1, len(await pool.ips()))):
        ip = await router.choose(request, counter, audio_id, frozenset(tried))
        url = f"http://{ip}:{pool.port}{request.rel_url.path_qs}"
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(request.method, url, data=body,
                                           headers=_forward_headers(request)) as upstream:
                    response = web.StreamResponse(
                        status=upstream.status,
                        headers={k: v for k, v in upstream.headers.items()
                                 if k.lower() not in ("content-encoding", "content-length",
                                                      "transfer-encoding", "connection")})
                    _set_sticky_cookie(response, request, ip)
                    if is_upload and upstream.status == 200:
                        # Small JSON reply: buffer it to learn the audio_id so
                        # the follow-up transcription can be pinned to this
                        # replica (keeps the webpage's two-step flow correct).
                        payload = await upstream.read()
                        if len(payload) <= UPLOAD_JSON_CAP:
                            try:
                                upload_id = json.loads(payload).get("id")
                            except Exception:
                                upload_id = None
                            if upload_id:
                                router.remember(str(upload_id), ip)
                        await response.prepare(request)
                        await response.write(payload)
                        await response.write_eof()
                        return response
                    await response.prepare(request)
                    async for chunk in upstream.content.iter_any():
                        await response.write(chunk)
                    await response.write_eof()
                    return response
        except Exception as exc:  # replica down/not-ready: try the next one
            last_error = exc
            tried.add(ip)
            _LOGGER.warning("replica %s failed for %s: %s", ip, request.path, exc)
    return web.json_response({"error": str(last_error)}, status=502)


async def _proxy_ws(request: web.Request, pool: BackendPool, router: Router,
                    counter) -> web.WebSocketResponse:
    ip = await router.choose(request, counter, None)
    url = f"http://{ip}:{pool.port}{request.rel_url.path_qs}"
    downstream = web.WebSocketResponse()
    await downstream.prepare(request)
    try:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(url) as upstream:
                async def pump(source, sink) -> None:
                    async for msg in source:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await sink.send_str(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await sink.send_bytes(msg.data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING,
                                          aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                        else:
                            pass
                await asyncio.gather(pump(upstream, downstream),
                                     pump(downstream, upstream))
    except Exception:
        _LOGGER.exception("ws proxy to %s failed", ip)
    return downstream


async def _health(request: web.Request, pool: BackendPool) -> web.Response:
    """Aggregate readiness: reachable replica count + their /health."""
    ips = await pool.ips()
    healthy = 0
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for ip in ips:
            try:
                async with session.get(f"http://{ip}:{pool.port}/health") as resp:
                    if resp.status == 200 and (await resp.json()).get("running", True):
                        healthy += 1
            except Exception:
                pass
    return web.json_response({"status": "healthy" if healthy else "degraded",
                              "running": healthy > 0,
                              "replicas": len(ips), "healthy": healthy})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="0.0.0.0:8000")
    parser.add_argument("--backend", default="asr:8000",
                        help="compose service name:port of the replicas")
    parser.add_argument("--ttl", type=float, default=5.0,
                        help="DNS re-resolution interval (seconds)")
    args = parser.parse_args()

    lhost, _, lport = args.listen.rpartition(":")
    pool = BackendPool(args.backend, ttl=args.ttl)
    router = Router(pool)
    counter = itertools.count()

    async def handle(request: web.Request) -> web.StreamResponse:
        if request.path == "/health":
            return await _health(request, pool)
        elif request.path == "/v1/audio/ws":
            return await _proxy_ws(request, pool, router, counter)
        else:
            return await _proxy_http(request, pool, router, counter)

    app = web.Application(client_max_size=1024 ** 3)
    app.router.add_route("*", "/{path:.*}", handle)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [lb] %(levelname)s %(message)s",
        datefmt="%H:%M:%S")
    print(f"[lb] listening on {lhost or '0.0.0.0'}:{lport or 8000} "
          f"-> {args.backend} (ttl {args.ttl}s, cookie={COOKIE})", flush=True)
    web.run_app(app, host=lhost or "0.0.0.0", port=int(lport or 8000),
                print=None, access_log=logging.getLogger("aiohttp.access"))


if __name__ == "__main__":
    main()
