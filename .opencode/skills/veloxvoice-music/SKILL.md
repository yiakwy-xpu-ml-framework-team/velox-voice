---
name: veloxvoice-music
description: Operate and extend VeloxVoice's music lane: SheetSage2/MERT-v2 transcription, Piano A/B, multi-lane scrolling staff, Demucs vocal separation, timed lyrics/LRC, model/tool placement, and DGX Spark demo URLs. Use for VeloxVoice music server work, 五线谱/A-B 对比, 人声分轨, 歌词时间轴, or music demo debugging.
---

# VeloxVoice music lane

This repo is a JIT-kernel ASR/music server, not a general model zoo. Keep
third-party demo tools outside the JIT tree unless we have optimized them.
See [model placement](references/model-placement.md) before adding a model.

## Start and reach the server

Preferred launcher:

```bash
VENV_PY=/home/yiakwy/venvs/sglang-h3/bin/python \
MUSIC_MODEL_DIR=/home/yiakwy/workspace/Github/models/SheetSage2 \
BASE_MODEL_DIR=/home/yiakwy/workspace/Github/models/MERT-v2-FullSong \
VELOX_DEMUCS_MODEL=htdemucs_ft \
bash scripts/run_music_server.sh
```

Manual equivalent (what the demo usually runs):

```bash
cd /home/yiakwy/workspace/Github/velox-voice
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
MUSIC_MODEL_DIR=/home/yiakwy/workspace/Github/models/SheetSage2 \
BASE_MODEL_DIR=/home/yiakwy/workspace/Github/models/MERT-v2-FullSong \
VELOX_DEMUCS_MODEL=htdemucs_ft \
/home/yiakwy/venvs/sglang-h3/bin/python -m uvicorn \
  launch_music_server:app --host 0.0.0.0 --port 8000
```

`MODEL_DIR` is optional. Setting it loads WeNet ASR as well; leaving it empty
is music-only and the ASR tab must show a confirmation modal, not silently
switch tabs.

**Reachability:** `localhost:8000` only works on the host that runs the server.
From another machine use the host LAN IP (currently `192.168.18.85`) or the
Tailscale IP (`100.111.227.95`), or forward the port:

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@192.168.18.85
# then http://localhost:8000/client works on the client
```

Check: `curl -s http://127.0.0.1:8000/healthz` -> `"music": true`,
`"asr": false` on a music-only server.

## Demo flow

1. `POST /v1/audio/music/upload` (multipart `file`) -> `upload_id`.
2. `POST /v1/audio/music/transcriptions` with `upload_id` and
   `engine=sheetsage|piano` (default `sheetsage`; the web `<select>` also
   defaults to **SheetSage2** and its label is just `SheetSage2`).
3. The response contains `staff_url`, `staff_mode`, `session`, audio and
   artifact URLs. The web client embeds the staff **in place** in the score
   card; do not redirect to a new page.
4. `GET /piano-staff/data?session=...&mode=a|b|sheet` returns:
   `duration`, `tempo`, `meter`, `key`, `bars`, `bar_seconds`, `measures`,
   `parts` (each with `events`), `abc`, `audio_url`, `metrics`.
5. The staff page renders **one long SVG row** with up to four staves
   (`Vocal`, `RH`, `LH`, `Bass`). `Vocal` is included only when the recognized
   vocal track has notes, so an instrumental track stays at three staves.
   For vocal songs, RH = recognized Ins melody + generated high accompaniment;
   LH = generated mid-register chords; Bass = generated root/fifth. A
   **Bass line details** table below the staff lists time/bar/note/chord/role
   (`bass_details` in the data payload). Notation is **black by default**;
   only while the audio is actually playing do currently sounding notes use
   the bright track color + glow (melody = Vocal if present, otherwise RH =
   green; RH light blue, LH light violet, Bass amber). Pausing removes the
   bright/glow state. Notes already performed turn **gray-black** (`#374151`),
   not a dark lane hue. Click any printed note/chord to select it and audition
   its pitches through WebAudio; the selected note uses the **same lane color
   and glow** as playback (pulsing), and the selected pitch/time/lane appears
   in the `noteInfo` chip. Once the cursor passes a selected note it still
   obeys the played-gray rule (`.past` wins). Selection is available in
   A/B/sheet, including the generated LH/Bass notes that are not present in
   the mix. Playback highlight is **x-based at the cursor**: only notes within
   14 px of the playhead glow, everything left of it is immediately `.past`
   gray, and a paused cursor has no `.playing` notes. Do not go back to raw
   event start/end windows; the quantized printed score and performance-time
   events drift apart.
   A lane legend shows the playing palette and the
   "默认黑色 · 播放过灰黑" rule. The playhead is a full-height
   vertical line plus a centered top arrow. It is pinned at half the viewport;
   the staff scrolls left after that. Cursor anchors use real
   `playback.json` measure bounds and notehead positions, not a flat index.
6. A/B: `mode=a` is real piano recognition (RH/LH/Bass split by register,
   original audio); `mode=b` is SheetSage2 Ins melody + generated LH/Bass
   (`piano_B.wav`). `mode=sheet` is Vocal (when present) + RH/LH/Bass against
   the original audio. The generated RH/LH/Bass are an arrangement; do not
   describe them as full accompaniment recognition.
7. Reopen sessions with `/client?tab=music&session=<id>`.

## Session persistence and IDs

Sessions live in `output/music_sessions` by default (env
`VELOX_MUSIC_OUTPUT_DIR` can point at another persistent volume). Never
default to `/tmp`; that is what lost sessions across reboots in the first
place. The upload/transcribe response returns both `session` and `upload_id`
(they are the same 12-hex id), and the web client shows it in the Session row
with Copy/Bookmark plus a searchable "Recent music sessions" panel.

- List persisted sessions: `GET /v1/audio/music/sessions?limit=30`.
- Identical audio is deduplicated by SHA-256: uploading it again returns
  `reused: true` and the existing session instead of creating a duplicate.
- `session_meta.json` stores the original filename/hash/created_at.
- Legacy `/tmp/veloxvoice_music_output` sessions are migrated with
  `scripts/migrate_music_sessions.py` (copies dirs, writes session_meta,
  builds `_index.json`).
- Reopening a session sets `musicUploadedId` so the user can re-run
  transcription with another engine without re-uploading.
- **Dynamic vocal detection:** `build_sheet_staff` reads `playback.json` and
  includes `Vocal` only when the recognized Vocal track has notes. A vocal
  song becomes 4 staves; an instrumental stays RH/LH/Bass. Do not hardcode the
  lane count. Fresh-upload regression: `6331453c7f01` (30 s vocal) -> 4 parts,
  `a14efbef6910` (30 s instrumental) -> 3 parts.
- Canonical regression sessions:
  - `5d5c35abdd63` — 60 s instrumental "Through the Shifting Light",
    Piano A/B, RH/LH/Bass, no vocals in the source (Demucs stems may exist;
    the vocal stem is near-silent).
  - `cdb74bb4867c` — 120 s Bohemian, SheetSage2 **Vocal + RH/LH/Bass**
    (165/163/288/64 events), Bass details, timed lyrics from LRCLIB, audible
    Demucs stems. This is the reference for a vocal song.
  - `c4f5fc9cc530` — full 343 s Bohemian; no timed lyrics/stems (older job).

**Lyrics provenance:** the cdb timeline is LRCLIB, not our ASR. The a617
timeline is a test `.lrc`. The music-only server has `asr: false`, so the
Demucs job returns `stems_only` and no ASR text. The UI prints
"Timed lyrics source: LRCLIB (not ASR)" / "uploaded .lrc (not ASR)" below the
timeline. Do not claim ASR-generated lyrics until faster-whisper (segment
timestamps) or a timestamped ASR path is actually wired.

### Key endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/audio/music/upload` | save uploaded song |
| `POST /v1/audio/music/transcriptions` | SheetSage2 or Piano A/B bundle |
| `GET /v1/audio/music/session/{id}` | reopen artifacts + `staff_url` |
| `GET /v1/audio/music/sessions` | persisted session list (title/id/stems/lyrics source) |
| `GET /piano-staff?session=&mode=&embed=1` | scrolling staff page |
| `GET /piano-staff/data?session=&mode=` | ABC/events/measures/metrics |
| `GET /i18n.js` | shared language loader (`?lang=en|zh`, localStorage) |
| `GET /i18n/en.md`, `/i18n/cn.md` | UI dictionaries (markdown tables) |
| `GET /piano-view/audio/{mode}?session=` | original (a/sheet) or `piano_B.wav` |
| `POST /v1/audio/music/lyrics` | start Demucs (+ ASR if loaded) |
| `GET /v1/audio/music/lyrics/{id}` | poll stems/lyrics/timed lines |
| `POST /v1/audio/music/lyrics/{id}/lrc` | attach user timed lyrics |

## Vocal separation and lyrics

- Adapter: `tools/music/separate_vocals.py`; default `htdemucs_ft`; outputs
  `vocals.wav`, `instrumental.wav`, `vocals.model` in the session directory.
- The lyrics lane runs automatically after transcription and on session
  reopen. On a music-only server it returns `status: done`, `stems_only: true`,
  and `vocals_url`/`instrumental_url`; the UI shows players/downloads.
- If timed lyrics already exist, separation still runs when stems are missing;
  the job preserves `lines`/`source` in every status payload.
- LRC upload preserves existing stem URLs.
- The 120 s Bohemian session `cdb74bb4867c` is the reference for
  stems + timed lyrics. The instrumental test track
  `Through the Shifting Light.mp3` has no Vocal notes, so its vocal stem is
  legitimately near-silent.
- Do not treat `vocals.wav` / `instrumental.wav` / `piano_*.wav` as the
  original song. `_original_audio_files` and `piano_engine.is_original_audio`
  exclude derived stems; keep that invariant.

## Internationalization

The client and staff pages share `templates/i18n.js` and the markdown
dictionaries `templates/i18n/en.md` / `cn.md`. The top-right `EN/中文` switch
sets `?lang=` and persists the choice in `localStorage`; the client propagates
the language to the embedded staff iframe. Any new user-visible string must be
added to both dictionaries, preferably via `data-i18n`/`data-i18n-placeholder`
or `VeloxI18n.t(key, vars)`. Keep product names (`SheetSage2`, `Piano A/B`),
note names, BPM/RTF/DTW and lane abbreviations (Vocal/RH/LH/Bass) unchanged.
Default language is Chinese; the engine selector defaults to `SheetSage2`.

## Model placement policy

- **External / demo / not yet optimized:** `tools/music/`, optional
  requirements (`requirements/vocal-separation.txt`), weights in the external
  model cache. This is where Demucs/BS-RoFormer belong today.
- **First-party optimized:** `veloxvoice/models/<name>/` for model code and
  `veloxvoice/kernels/csrc/dgx/` for GB10 kernels. Only move a model here after
  the JIT/optimized implementation and quality/speed gates exist.
- **Weights:** never commit checkpoints or stems. Record revision/hash in a
  manifest; use an external models root.
- Full policy and promotion checklist:
  [references/model-placement.md](references/model-placement.md).

## Validate before claiming success

Use [references/validation.md](references/validation.md). Minimum:

```bash
curl -s http://127.0.0.1:8000/healthz
curl -sS -F "file=@test_data/music/Through the Shifting Light.mp3" \
  http://127.0.0.1:8000/v1/audio/music/upload
```

Then transcribe, fetch `/piano-staff/data` for `a`, `b`, `sheet`, and check:
three staves, one SVG line, no `Vocal`, `playhead` line/arrow same center,
`xAtTime(q_start)` matches notehead x, audio readyState 4, stop resets to
start, replay starts from 0. For lyrics, run the Demucs job and confirm the
stems play and the LRC/active line stays in sync via the iframe postMessage.

## Known pitfalls

- `localhost` on a laptop is not the DGX host; use LAN/Tailscale/SSH tunnel.
- Demucs 4.1.x needs `sphn` (aarch64 build can hang); pin 4.0.1.
- Recent torchaudio `save()` needs torchcodec; `separate_vocals.py` patches
  it with soundfile. Do not remove that patch.
- The music-only server has `vx is None`; do not call ASR unconditionally.
- SheetSage2 can hit a CUDA device-side assert on long audio; 60/120 s are
  validated, the full 343 s case is not.
- The embedded staff is an iframe that owns its audio; lyrics sync via
  `postMessage` (`velox-staff-state`). Do not reintroduce a second playing
  audio element.
- Do not exclude original audio by a broad `piano_` / `vocals` prefix:
  test uploads are legitimately named `piano_theme_melody.wav`. Exclude the
  exact generated names (`vocals.wav`, `instrumental.wav`,
  `piano_A_synth.wav`, `piano_B.wav`, `piano_mix/vocal/instrumental.wav`,
  `no_vocals.wav`).
- Do not describe LRCLIB/LRC lines as ASR output. Check `lyrics_source` in
  `GET /v1/audio/music/sessions` and the UI source label.
- Do not drop the Vocal staff for a song that actually has vocal notes
  (cdb has 165). Conversely, do not render an empty Vocal staff for an
  instrumental track (5d5 has 0).
- For vocal songs, do not leave RH empty or push every generated chord into
  LH. `with_rh=True` adds the high C4-C5 chord layer; LH is mid-register,
  Bass is root/fifth. The Bass details table must stay in sync.
- Do not permanently color the notation. Default noteheads are black; the
  bright per-track color + glow is only for `.playing` while `audio.paused`
  is false, and `.past` must be gray-black (`#374151`). Pausing must remove
  `.playing` (no glow at a paused cursor); `updateHighlights` must toggle both
  classes, and `clearHighlights` must remove both on stop/replay/mode switch.
- Playback highlight is derived from the cursor x (`xAtTime`) and a 14 px
  window, not event start/end. This is intentional: quantized bar positions
  and performance-time note events can be seconds apart. The moment a note is
  left of the cursor it must be `.past`; never keep it `.playing`.
- Click-to-audition must work in every staff mode. Keep `groupMeta` populated
  for *all* `.abcjs-note` groups (including tie continuations), fall back to
  backend `event.pitches` because ABCJS leaves `midiPitches` empty in this
  render configuration, and never require the generated LH/Bass to exist in
  the audio mix. Selected notes use the lane `--play-color`/`--play-glow`.
- UI strings: add new keys to both `templates/i18n/en.md` and `cn.md`; the
  `EN/中文` switch must preserve `session` and `lang` parameters. Do not
  hardcode new Chinese/English UI text in the templates.
- `bass_details` is a UI explanation, not a new audio stem. Regenerating the
  staff (version 7) is required after changing the generation rules.

## Recursive self-improvement

After any music-lane change:

1. Update this skill or its references with the new command/endpoint/pitfall.
2. Add or update a regression check in `tests/` or the validation reference.
3. Record the actual command output (timings, session ids) in the change note.
4. Keep third-party weights and generated stems out of git.
5. If a model becomes genuinely optimized, update the placement table and
   promotion checklist; do not leave stale "tool" paths in the skill.

Related global skill for song generation/covers (different repo):
`yue2-music`. This repo owns the recognition/score/stem demo, not YuE2
generation.
