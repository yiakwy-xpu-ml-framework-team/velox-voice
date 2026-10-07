#!/usr/bin/env python3
"""Minimal round-robin load balancer for VeloxVoice ASR workers.

Proxies HTTP (multipart uploads included) and WebSocket connections from a
single listen port to a pool of worker ports on the same host, vLLM-style:
workers own the GPU; the balancer only distributes.

    python benchmark/asr_lb.py --port 8010 --workers 8000 8001 8002
"""

from __future__ import annotations

import argparse
import asyncio
import itertools

import aiohttp
from aiohttp import web


async def proxy_handler(
    request: web.Request, pool: list[str], counter
) -> web.StreamResponse:
    target = pool[next(counter) % len(pool)]
    url = f"{target}{request.rel_url.path_qs}"
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=3600)
    body = await request.read() if request.can_read_body else None
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower()
        not in (
            "host",
            "connection",
            "keep-alive",
            "transfer-encoding",
            "content-length",
        )
    }
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                request.method, url, data=body, headers=headers
            ) as upstream:
                response = web.StreamResponse(
                    status=upstream.status,
                    headers={
                        k: v
                        for k, v in upstream.headers.items()
                        if k.lower()
                        not in (
                            "content-encoding",
                            "content-length",
                            "transfer-encoding",
                            "connection",
                        )
                    },
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)


async def ws_handler(
    request: web.Request, pool: list[str], counter
) -> web.WebSocketResponse:
    target = pool[next(counter) % len(pool)]
    url = f"{target}{request.rel_url.path_qs}"
    downstream = web.WebSocketResponse()
    await downstream.prepare(request)
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
        ) as session:
            async with session.ws_connect(url) as upstream:

                async def pump(source, sink):
                    async for msg in source:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await sink.send_str(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await sink.send_bytes(msg.data)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break

                await asyncio.gather(
                    pump(upstream, downstream), pump(downstream, upstream)
                )
    except Exception:
        import traceback

        traceback.print_exc()
    return downstream


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--workers", type=int, nargs="+", default=[8000, 8001, 8002])
    args = parser.parse_args()

    pool = [f"http://127.0.0.1:{port}" for port in args.workers]
    counter = itertools.count()

    async def handle(request: web.Request) -> web.StreamResponse:
        if request.path == "/v1/audio/ws":
            return await ws_handler(request, pool, counter)
        else:
            return await proxy_handler(request, pool, counter)

    app = web.Application(client_max_size=1024**3)
    app.router.add_route("*", "/{path:.*}", handle)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
