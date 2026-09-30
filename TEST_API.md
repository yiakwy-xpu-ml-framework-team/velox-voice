# VeloxVoice API test commands

Base URL:

```bash
BASE=http://127.0.0.1:8000
AUDIO=audio.mp3
```

## 1. Upload only

```bash
curl -sS -X POST "$BASE/v1/audio/uploads" \
  -F "file=@$AUDIO;type=audio/mpeg"
```

Example response:

```json
{
  "id": "b6ffa05fc67c4383a5df2ca1fd1d5b51",
  "filename": "audio.mp3",
  "audio_path": "logs/user_data/20260928/130947_67772438_audio.mp3",
  "size": 57441
}
```

Save the returned `id` for deferred transcription:

```bash
AUDIO_ID=your_upload_id
```

Or extract it automatically:

```bash
AUDIO_ID=$(curl -sS -X POST "$BASE/v1/audio/uploads" \
  -F "file=@$AUDIO;type=audio/mpeg" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
```

## 2. Transcribe after uploading

Non-streaming:

```bash
curl -sS -X POST "$BASE/v1/audio/transcriptions" \
  -F "audio_id=$AUDIO_ID" \
  -F "response_format=json"
```

Streaming SSE:

```bash
curl -N -sS -X POST "$BASE/v1/audio/transcriptions" \
  -F "audio_id=$AUDIO_ID" \
  -F "response_format=json" \
  -F "stream=true"
```

Optional fields:

```bash
curl -N -sS -X POST "$BASE/v1/audio/transcriptions" \
  -F "audio_id=$AUDIO_ID" \
  -F "model=asr_model" \
  -F "language=zh" \
  -F "prompt=meeting" \
  -F "response_format=json" \
  -F "stream=true"
```

## 3. Live upload and transcription

`/v1/audio/ws` is a WebSocket endpoint, not a REST endpoint, so `curl` cannot
drive the full upload/stream/test lifecycle. Use Python with `websockets`:

```bash
pip install websockets
```

```bash
python3 - <<'PY'
import asyncio
import json
import pathlib
import websockets

BASE = "127.0.0.1:8000"
AUDIO = pathlib.Path("audio.mp3")
POOL_SECONDS = 5


async def main():
    uri = f"ws://{BASE}/v1/audio/ws?pool_seconds={POOL_SECONDS}"
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "start",
            "pool_seconds": POOL_SECONDS,
            "format": "auto",
            "filename": AUDIO.name,
        }))

        data = AUDIO.read_bytes()
        chunk_size = 1024 * 1024
        for offset in range(0, len(data), chunk_size):
            await ws.send(data[offset:offset + chunk_size])

        await ws.send(json.dumps({"type": "stop"}))

        while True:
            obj = json.loads(await ws.recv())
            print(obj)
            if obj.get("type") in ("final", "error"):
                break


asyncio.run(main())
PY
```

## 4. Direct upload and transcribe in one REST request

```bash
curl -sS -X POST "$BASE/v1/audio/transcriptions" \
  -F "file=@$AUDIO;type=audio/mpeg" \
  -F "response_format=json"
```

Streaming variant:

```bash
curl -N -sS -X POST "$BASE/v1/audio/transcriptions" \
  -F "file=@$AUDIO;type=audio/mpeg" \
  -F "response_format=json" \
  -F "stream=true"
```

## 5. Discovery and health

```bash
curl -sS "$BASE/health"
curl -sS "$BASE/v1/models"
curl -sS "$BASE/api"
curl -sS "$BASE/openapi.json" | python3 -m json.tool
curl -sS "$BASE/sitemap.xml"
```
