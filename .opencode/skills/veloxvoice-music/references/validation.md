# Music lane validation

Use a fresh session when possible. The server is on the DGX host; do not use
`localhost` from a remote browser.

## 1. Health and reachability

```bash
curl -s http://127.0.0.1:8000/healthz
# {"ok":true,"device":"cuda","asr":false,"music":true}

curl -s -o /dev/null -w '%{http_code}\n' \
  http://192.168.18.85:8000/client?tab=music
```

If the second call works on the host but not from a laptop, use the LAN/Tailscale
IP or an SSH tunnel (see the main skill).

## 2. Transcribe

```bash
UP=$(curl -sS -F "file=@test_data/music/Through the Shifting Light.mp3" \
  http://127.0.0.1:8000/v1/audio/music/upload)
ID=$(printf '%s' "$UP" | python3 -c 'import json,sys;print(json.load(sys.stdin)["upload_id"])')

curl -sS -X POST -F "upload_id=$ID" -F "engine=sheetsage" \
  http://127.0.0.1:8000/v1/audio/music/transcriptions | tee /tmp/sheet.json
curl -sS -X POST -F "upload_id=$ID" -F "engine=piano" \
  http://127.0.0.1:8000/v1/audio/music/transcriptions | tee /tmp/piano.json
```

Expected:
- `staff_url` present, `staff_mode` `sheet` or `a`.
- `processing.wall_time_seconds` real; SheetSage2 ~1 s, Piano A/B ~6-17 s.
- `abc` starts `X:1` with `V: RH`, `V: LH`, `V: Bass` (no `V: Vocal`).

## 3. Staff payload

```bash
curl -sS "http://127.0.0.1:8000/piano-staff/data?session=$ID&mode=a" \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["bars"],len(d["measures"]),[(p["name"],len(p["events"])) for p in d["parts"]])'
```

- `parts` names: `Vocal`, `RH`, `LH`, `Bass` (empty voices are omitted;
  `Vocal` only appears when the recognized vocal track has notes).
- `measures` follows `playback.json` (pickup-aware).
- `bass_details` is non-empty for modes with a Bass part; each row has
  `time`, `bar`, `note`, `chord`, `role`, `source`.
- `duration` is the real audio duration, not 60 hardcoded.

## 4. Browser checks (Playwright)

Load `/piano-staff?session=<id>&mode=sheet&embed=1` and assert:
- cdb: 4 staves, voice names `Vocal / RH / LH / Bass`, ~64 Bass detail rows
- 5d5: 3 staves, no `Vocal` voice (instrumental)
- fresh vocal upload `6331453c7f01`: 4 parts; fresh instrumental
  `a14efbef6910`: 3 parts (dynamic detection)
- at load, noteheads are black (`rgb(0,0,0)`) and no `.playing`/`.past`
  classes exist
- while **playing** around 30 s: only notes within 14 px of the playhead
  cursor are `.playing` (lane colors/glow); all noteheads left of the cursor
  are `.past` gray-black `rgb(55,65,81)`, future notes stay black
- pause: `.playing` count is 0; `.past` remains for notes left of the cursor;
  the note at the cursor is black
- stop/replay clears both classes and returns all notes to black
- click-to-audition: clicking a `.abcjs-note` (or its notehead/stem) adds
  exactly one `.selected` class, uses that lane's `--play-color` (Vocal green,
  RH sky/green melody, LH violet, Bass amber), fills `#noteInfo` with
  pitches/time/lane, and invokes WebAudio
  (`AudioContext.prototype.createOscillator` count > 0). Repeat in
  `mode=b` and `mode=sheet`; every `.abcjs-note` group must have a
  `window.__veloxStaff.groupMeta` entry, including tie continuations.
- background click or Escape clears `.selected` and `#noteInfo`.
- i18n: `?lang=en` / `?lang=zh` and the top-right `EN/中文` switch translate
  tabs, engine/session labels, recent sessions, staff buttons, legends,
  Bass headers and note-info text. The iframe `src` must carry `lang=`.
  `localStorage.velox_lang` must persist the choice.
- `.abcjs-staff-wrapper` count == 1 (one long row)
- playhead line center == arrow center
- `window.__veloxStaff.xAtTime(note.q_start)` ≈ notehead x (p90/p99 0 or < a few px)
- `audio.readyState == 4`, `audio.duration` matches payload
- stop: `track.style.transform == translateX(0px)`, `audio.paused == true`,
  `audio.currentTime == 0`; replay starts playing.

For the client (`/client?tab=music&session=<id>`), assert:
- `#musicEngine` defaults to `sheetsage` and its label is exactly `SheetSage2`
  (the option text must not include `三轨长行五线谱`)
- `#staffFrame` exists and its `src` includes `embed=1`
- iframe has 3 staves and one line
- clicking a note inside the iframe selects/auditions it (same assertions as
  above via `frame.contentDocument`)
- lyrics timeline active line updates after iframe seek (postMessage)
- `#stemRow` shows `vocals.wav` / `instrumental.wav` when separation is done

## 5. Demucs stems

```bash
curl -s -X POST -F "session=$ID" http://127.0.0.1:8000/v1/audio/music/lyrics
curl -s http://127.0.0.1:8000/v1/audio/music/lyrics/$ID
```

Expected on a music-only server:

```json
{"status":"done","stems_only":true,"model":"htdemucs_ft",
 "vocals_url":"/v1/audio/music/score/<id>/vocals.wav",
 "instrumental_url":"/v1/audio/music/score/<id>/instrumental.wav",
 "separation_seconds":14.3,"lines":[...preserved if any...]}
```

Play both URLs; a 120 s track is ~31 MB per 24-bit stem and should reach
`readyState 4`. Instrumental source tracks may produce a near-silent vocal
stem — verify SheetSage2's `Vocal` note count first.

## 6. Persistence, IDs, dedup

```bash
curl -s http://127.0.0.1:8000/v1/audio/music/sessions | python3 -m json.tool
# root should be output/music_sessions (or VELOX_MUSIC_OUTPUT_DIR)
```

- Restart the server, then reopen the same `/client?tab=music&session=<id>`:
  the staff, stems, and lyrics must still load.
- Upload `test_data/music/Through the Shifting Light.mp3` again; the response
  must be `reused: true` and `session` `5d5c35abdd63` (the canonical 60 s
  instrumental A/B session), not a new id.
- In the UI, the Session row shows/copies the id, the recent list filters by
  title/id, and clicking an item updates the URL session parameter.
- `GET /v1/audio/music/sessions` must report `lyrics_source` (`lrclib`,
  `lrc`, ...) and `has_stems`; it is the fastest way to confirm the lyrics are
  not being misrepresented as ASR.

## 7. Regression sessions

- `5d5c35abdd63`: 60 s instrumental, piano A/B, 3 staves/73 bars, no vocals
  in the source (canonical pure-music session; Demucs vocal stem is
  legitimately near-silent).
- `cdb74bb4867c`: 120 s Bohemian, SheetSage2 **Vocal + RH/LH/Bass**,
  Bass details, **LRCLIB** timed lyrics, Demucs stems (reference for
  vocal + accompaniment + lyrics + stems together).
- `a617ffe6070d`: 120 s test upload with stems + a manually attached `.lrc`
  (source `lrc`, not ASR); also a vocal song.
- `6331453c7f01`: 30 s fresh vocal upload, 4 parts (Vocal/RH/LH/Bass).
- `a14efbef6910`: 30 s fresh instrumental upload, 3 parts (no Vocal).
- `c4f5fc9cc530`: full 343 s Bohemian; no timed lyrics/stems.
