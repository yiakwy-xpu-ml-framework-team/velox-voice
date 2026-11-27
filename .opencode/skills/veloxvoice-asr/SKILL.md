---
name: veloxvoice-asr
description: Operate and extend VeloxVoice's ASR lane: optional WeNet model loading, music-only mode, ASR-tab confirmation behavior, REST/SSE/WebSocket transcription, JIT/model placement, and the handoff from Demucs vocal stems to timed lyrics. Use for VeloxVoice ASR server work, 语音识别, streaming transcription, missing-ASR demos, or lyrics ASR debugging.
---

# VeloxVoice ASR lane

VeloxVoice is ASR-first but the demo can run music-only. ASR and music share
`create_app`; `MODEL_DIR` selects the WeNet ASR checkpoint, while
`MUSIC_MODEL_DIR` + `BASE_MODEL_DIR` select SheetSage2/MERT-v2.

## Launch modes

Both lanes:

```bash
MODEL_DIR=/path/to/asr_checkpoint \
MUSIC_MODEL_DIR=/home/yiakwy/workspace/Github/models/SheetSage2 \
BASE_MODEL_DIR=/home/yiakwy/workspace/Github/models/MERT-v2-FullSong \
/home/yiakwy/venvs/sglang-h3/bin/python -m uvicorn \
  launch_music_server:app --host 0.0.0.0 --port 8000
```

Music-only (current demo): omit `MODEL_DIR` (or set it empty).

`/healthz` is the source of truth:

```json
{"ok":true,"asr":false,"music":true}
```

`vx is None` when ASR is not loaded. Never call `transcribe_pcm` / `_asr_vocals`
without checking this. The web ASR tab must show the confirmation modal and
must not auto-switch to the music lane; the user has to explicitly confirm
starting ASR.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/audio/uploads` | stage an upload, returns `id` |
| `POST /v1/audio/transcriptions` | multipart or `audio_id`; JSON/text/SRT/VTT |
| `POST /v1/audio/transcriptions` + `stream=true` | SSE partials/final |
| `WS /v1/audio/ws` | live upload + streaming transcription |
| `GET /v1/models` | ASR model name |
| `GET /healthz` | `asr`/`music` capabilities |

Non-streaming smoke:

```bash
AUDIO_ID=$(curl -sS -X POST http://127.0.0.1:8000/v1/audio/uploads \
  -F "file=@test.wav" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
curl -sS -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "audio_id=$AUDIO_ID" -F "response_format=json"
```

Streaming and the Python WebSocket client are in `TEST_API.md`; use that file
instead of inventing another protocol.

## Missing-ASR behavior

- `/client` ASR tab: confirmation modal (`#asrModal`), checkbox enables the
  confirm button. Cancelling stays on the current tab.
- Music lyrics job: still separates vocals with Demucs; returns
  `status: done`, `stems_only: true`, `vocals_url`, `instrumental_url`.
  The UI shows the stems and tells the user to upload an LRC for timed lyrics.
- Do not fake ASR output or silently substitute a different model.

## Model placement and kernels

- WeNet model **code** lives in `veloxvoice/models/wenet/` and the runtime in
  `veloxvoice/api.py` (`Velox.load`, `transcribe_pcm`,
  `transcribe_pcm_iter`).
- ASR checkpoints are external and must not be committed. Point `MODEL_DIR`
  at the checkpoint directory; record the revision/hash in the manifest.
- JIT/CUDA kernels for DGX Spark GB10 belong in
  `veloxvoice/kernels/csrc/dgx/`; the Python model code must expose the
  optimized op and a fallback path.
- Do not add third-party demo weights to `veloxvoice/models` just to run a
  demo. External tools belong in `tools/` + optional requirements. See
  `.opencode/skills/veloxvoice-music/references/model-placement.md`.

## Vocal stem → timed lyrics

Current music-only pipeline: Demucs separates `vocals.wav`; no ASR/timestamps.
The timed lyrics in the demo (`cdb74bb4867c`) are **LRCLIB**, not ASR; the
`a617ffe6070d` timeline is an uploaded `.lrc`. The UI and
`GET /v1/audio/music/sessions` expose `lyrics_source`; never label those as
ASR output. For automatic timed lyrics the recommended next step is:

1. Separate vocals with Demucs `htdemucs_ft` (or BS-RoFormer for SOTA).
2. Run `faster-whisper` `large-v3` on the 16 kHz mono vocal stem.
3. Use segment timestamps as `lines: [{start, end, text}]` and write
   `lyrics.json`; the existing staff/lyrics UI already consumes that format.
4. Preserve `vocals_url` / `instrumental_url` when replacing `lyrics.json`.

Do not claim timed lyrics from the plain WeNet text path; it has no segment
timestamps. Keep LRC as a deterministic fallback.

## Validation

```bash
curl -s http://127.0.0.1:8000/healthz
curl -s http://127.0.0.1:8000/v1/models
# with ASR loaded:
curl -sS -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@test.wav" -F "response_format=json"
# music-only:
curl -s -X POST -F "session=<id>" http://127.0.0.1:8000/v1/audio/music/lyrics
curl -s http://127.0.0.1:8000/v1/audio/music/lyrics/<id>
```

Also verify the ASR-tab modal in Playwright: `#asrModal` display `flex`,
`#asrOk` disabled until `#asrConfirm`, and no auto tab switch.

## Recursive self-improvement

1. After an ASR change, update this skill and `TEST_API.md` if the wire format
   changed.
2. Record the `healthz` capability and the exact smoke command/result.
3. Add a regression test for the no-ASR modal and the stems-only fallback.
4. Never commit checkpoints or generated audio.
5. If faster-whisper / BS-RoFormer becomes part of the pipeline, update the
   model-placement table and the music skill instead of duplicating paths.
