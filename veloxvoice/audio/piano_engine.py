"""Piano A/B and three-track staff engine for the VeloxVoice music lane.

A: ByteDance high-resolution piano transcription (polyphonic note events).
B: SheetSage2 melody (RH) + generated LH chords/bass from chord.lab.
S: SheetSage2 staff view (same melody + generated LH/bass, original audio).

The staff is deliberately rendered as *one long horizontal row* with up to
three voices (RH / LH / Bass), matching a Reaper-style scrolling timeline.
Artifacts are written into the session directory so the web UI only needs a
few static JSON/ABC files.
"""
from __future__ import annotations

import bisect
import functools
import json
import math
import re
from pathlib import Path

import librosa
import mir_eval.chord
import numpy as np
import soundfile as sf

SR = 48000
REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO_ROOT / "render_assets" / "soundfonts" / "acoustic_grand_piano-mp3"
CHECKPOINT = Path("/home/yiakwy/workspace/Github/models/piano_transcription/note_pedal.pth")
SAMPLE_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Order matters: it is also the top-to-bottom staff order in the score.
# Vocal is omitted automatically when the source has no vocal notes, so an
# instrumental track never shows an empty lane.
VOICE_ORDER = (
    ("Vocal", "treble", "Vocal"),
    ("RH", "treble", "R.H."),
    ("LH", "bass", "L.H."),
    ("Bass", "bass", "Bass"),
)

DEFAULT_CONTEXT = {
    "meter": "4/4",
    "tempo": 72,
    "q_num": 1,
    "q_den": 4,
    "key": "C",
    "units_per_bar": 16,
    "unit_seconds": 60.0 / 72.0 / 4.0,
    "bar_seconds": 60.0 / 72.0 * 4.0,
}

AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
# Exact generated filenames only: a user is allowed to upload e.g.
# piano_theme_melody.wav without it being mistaken for a piano render.
DERIVED_AUDIO_NAMES = {
    "vocals.wav", "instrumental.wav", "no_vocals.wav",
    "piano_A_synth.wav", "piano_B.wav",
    "piano_mix.wav", "piano_vocal.wav", "piano_instrumental.wav",
}


def is_original_audio(path: Path) -> bool:
    return (path.is_file() and
            path.suffix.lower() in AUDIO_SUFFIXES and
            path.name not in DERIVED_AUDIO_NAMES)


# ---------------------------------------------------------------------------
# ABC / staff
# ---------------------------------------------------------------------------
def abc_pitch(midi: int) -> str:
    """MIDI pitch -> ABC pitch token (flats preferred, as SheetSage2 emits)."""
    names = ["C", "_D", "D", "_E", "E", "F",
             "_G", "G", "_A", "A", "_B", "B"]
    name = names[int(midi) % 12]
    octave = int(midi) // 12 - 1
    if octave >= 5:
        return name.lower() + "'" * (octave - 5)
    if octave == 4:
        return name
    return name + "," * (4 - octave)


def _dur(n: int) -> str:
    return "" if n == 1 else str(n)


def _parse_meter(text: str) -> str:
    match = re.search(r"^M:\s*(\S+)", text, re.M)
    if not match:
        return "4/4"
    raw = match.group(1)
    if raw in ("C", "c"):
        return "4/4"
    if raw in ("C|", "c|"):
        return "2/2"
    parsed = re.match(r"(\d+)\s*/\s*(\d+)", raw)
    if parsed:
        return f"{int(parsed.group(1))}/{int(parsed.group(2))}"
    return "4/4"


def _parse_tempo(text: str) -> tuple[int, int, int]:
    """Return (bpm, beat-numerator, beat-denominator)."""
    match = re.search(
        r"^Q:\s*(\d+)\s*/\s*(\d+)\s*(?:=|[ \t])\s*(\d+)", text, re.M)
    if match:
        return int(match.group(3)), int(match.group(1)), int(match.group(2))
    match = re.search(r"^Q:\s*(\d+)", text, re.M)
    if match:
        return int(match.group(1)), 1, 4
    return 72, 1, 4


def score_context(session_dir: Path) -> dict:
    """Read the musical context (meter/tempo/key) of the recognized score."""
    session_dir = Path(session_dir)
    text = ""
    score_abc = session_dir / "score.abc"
    if score_abc.exists():
        try:
            text = score_abc.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""

    meter = _parse_meter(text)
    try:
        num, den = (int(x) for x in meter.split("/"))
    except Exception:
        meter, num, den = "4/4", 4, 4
    tempo, q_num, q_den = _parse_tempo(text)
    key_match = re.search(r"^K:\s*(\S+)", text, re.M)
    key = key_match.group(1) if key_match else "C"

    units_per_bar = max(1, int(round(16.0 * num / den)))
    unit_seconds = 60.0 * q_den / (tempo * 16.0 * q_num)
    return {
        "meter": meter,
        "tempo": tempo,
        "q_num": q_num,
        "q_den": q_den,
        "key": key,
        "units_per_bar": units_per_bar,
        "unit_seconds": unit_seconds,
        "bar_seconds": unit_seconds * units_per_bar,
    }


def session_duration(session_dir: Path, default: float = 60.0) -> float:
    """Best-effort audio duration for a session (result -> playback -> audio)."""
    session_dir = Path(session_dir)
    for name, key in (("result.json", "duration_seconds"),
                      ("playback.json", "duration")):
        path = session_dir / name
        if not path.exists():
            continue
        try:
            value = float(json.loads(
                path.read_text(encoding="utf-8")).get(key) or 0.0)
            if value > 0:
                return value
        except Exception:
            pass
    try:
        audio = next(p for p in session_dir.iterdir()
                     if is_original_audio(p))
        value = float(librosa.get_duration(path=str(audio)))
        if value > 0:
            return value
    except Exception:
        pass
    return float(default)


def normalize_note(note: dict) -> dict:
    """Normalize both our JSON format and raw ByteDance result records."""
    if "pitch" in note:
        pitch = int(note["pitch"])
        start = float(note["start"])
        end = float(note["end"])
        velocity = int(note.get("velocity", 70))
    else:
        pitch = int(note["midi_note"])
        start = float(note["onset_time"])
        end = float(note["offset_time"])
        velocity = int(note.get("velocity", 70))
    return {"pitch": pitch, "start": start, "end": max(start + 0.02, end),
            "velocity": max(1, min(127, velocity))}


def split_piano_notes(notes) -> tuple[list, list, list]:
    """Split recognized piano notes into RH / LH / Bass lanes."""
    rh, lh, bass = [], [], []
    for raw in notes:
        note = normalize_note(raw)
        pitch = note["pitch"]
        if pitch >= 60:
            rh.append(note)
        elif pitch >= 48:
            lh.append(note)
        else:
            bass.append(note)
    for group in (rh, lh, bass):
        group.sort(key=lambda n: n["start"])
    return rh, lh, bass


def load_measure_bounds(session_dir: Path):
    """Actual measure start/end times from playback.json (pickup-aware)."""
    path = Path(session_dir) / "playback.json"
    if not path.exists():
        return None
    try:
        playback = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    bounds = []
    for measure in playback.get("measures") or []:
        try:
            start = max(0.0, float(measure.get("start")))
            end = float(measure.get("end"))
        except (TypeError, ValueError):
            continue
        if end > start + 1e-4:
            bounds.append((start, end))
    if not bounds:
        return None
    bounds.sort(key=lambda item: item[0])
    return bounds


def build_staff_abc(part_notes: dict, context: dict, duration: float,
                    *, title: str = "Piano", measure_bounds=None):
    """Build a one-row ABC score plus per-voice events for playback sync.

    ``part_notes`` maps voice name -> note list.  Empty voices are omitted so
    the score never shows an empty "Vocal Melody" lane.  When
    ``measure_bounds`` is supplied, notes are quantized proportionally inside
    the *actual* measures (so pickup bars and tempo drift keep their real
    barline times) instead of a uniform nominal grid.
    """
    upb = int(context["units_per_bar"])
    bar_seconds = float(context["bar_seconds"])

    bounds = []
    if measure_bounds:
        for start, end in measure_bounds:
            try:
                start, end = max(0.0, float(start)), float(end)
            except (TypeError, ValueError):
                continue
            if end > start + 1e-4:
                bounds.append((start, end))
    if not bounds:
        n_bars = max(1, int(math.ceil(float(duration) / bar_seconds - 1e-9)))
        bounds = [(index * bar_seconds, (index + 1) * bar_seconds)
                  for index in range(n_bars)]
    n_bars = len(bounds)
    measure_starts = [start for start, _ in bounds]

    def locate(time):
        index = bisect.bisect_right(measure_starts, time) - 1
        return max(0, min(n_bars - 1, index))

    def position_in_bar(time, measure):
        start, end = bounds[measure]
        span = max(1e-6, end - start)
        fraction = min(max((time - start) / span, 0.0), 1.0 - 1e-9)
        return int(round(fraction * upb))

    def time_at(measure, units):
        start, end = bounds[measure]
        return start + (units / float(upb)) * (end - start)

    def quantize(notes, voice):
        onsets = [dict() for _ in range(n_bars)]
        for raw in notes:
            note = normalize_note(raw)
            measure = locate(note["start"])
            pos = min(upb - 1, position_in_bar(note["start"], measure))
            group = onsets[measure].setdefault(pos, {
                "start": note["start"], "end": note["end"],
                "pitches": set(),
            })
            group["start"] = min(group["start"], note["start"])
            group["end"] = max(group["end"], note["end"])
            group["pitches"].add(note["pitch"])

        # Expand notes that sustain across barlines into tied segments.  Every
        # segment keeps the original start/end so playback highlighting lights
        # the whole tied note; q_start/q_end describe the printed segment.
        slots = [dict() for _ in range(n_bars)]
        for measure in range(n_bars):
            onset_positions = sorted(onsets[measure])
            for pos in onset_positions:
                group = onsets[measure][pos]
                pitches = set(group["pitches"])
                current_measure, current_pos = measure, pos
                while True:
                    positions_here = sorted(onsets[current_measure])
                    next_pos = next(
                        (p for p in positions_here if p > current_pos), upb)
                    _, measure_end = bounds[current_measure]
                    is_onset = (current_measure == measure and
                                current_pos == pos)
                    end_time = min(group["end"], measure_end)
                    end_pos = position_in_bar(end_time, current_measure)
                    if end_pos <= current_pos:
                        end_pos = current_pos + 1
                    segment_end = min(upb, next_pos, end_pos)
                    if segment_end <= current_pos:
                        segment_end = current_pos + 1
                    length = segment_end - current_pos
                    continues = (group["end"] > measure_end + 1e-4 and
                                 segment_end == upb)
                    slot = slots[current_measure].get(current_pos)
                    if slot is None:
                        slot = {
                            "start": group["start"], "end": group["end"],
                            "pitches": set(pitches), "length": length,
                            "tie": continues, "onset": is_onset,
                        }
                        slots[current_measure][current_pos] = slot
                    else:
                        # A new onset and a tied continuation can land on the
                        # same beat; show them as one chord.
                        slot["start"] = min(slot["start"], group["start"])
                        slot["end"] = max(slot["end"], group["end"])
                        slot["pitches"] |= pitches
                        slot["length"] = max(slot["length"], length)
                        slot["tie"] = slot["tie"] or continues
                        slot["onset"] = slot.get("onset", False) or is_onset
                    if not continues or current_measure + 1 >= n_bars:
                        break
                    current_measure += 1
                    current_pos = 0

        # A tie is only valid if the next bar starts with the same pitch set.
        for measure in range(n_bars - 1):
            for pos, slot in slots[measure].items():
                if not slot["tie"]:
                    continue
                following = slots[measure + 1].get(0)
                if not following or following["pitches"] != slot["pitches"]:
                    slot["tie"] = False

        lines = []
        events = []
        for measure in range(n_bars):
            pieces, cursor = [], 0
            for pos, slot in sorted(slots[measure].items()):
                length = max(1, min(slot["length"], upb - pos))
                pitches = sorted(slot["pitches"])
                token = ("[" + "".join(abc_pitch(p) for p in pitches) + "]"
                         if len(pitches) > 1 else abc_pitch(pitches[0]))
                if pos > cursor:
                    pieces.append("z" + _dur(pos - cursor))
                # ABC ties are written after the duration: d4-|d4
                pieces.append(token + _dur(length) +
                              ("-" if slot["tie"] else ""))
                cursor = pos + length
                if not slot.get("onset"):
                    continue
                events.append({
                    "voice": voice,
                    "start": round(slot["start"], 4),
                    "end": round(slot["end"], 4),
                    "q_start": round(time_at(measure, pos), 4),
                    "q_end": round(time_at(
                        measure, min(pos + length, upb)), 4),
                    "pitches": pitches,
                    "bar": measure,
                    "pos": pos,
                    "units": length,
                    "tie": bool(slot["tie"]),
                })
            if cursor < upb:
                pieces.append("z" + _dur(upb - cursor))
            lines.append("".join(pieces) + "|")
        return "".join(lines), events

    headers, bodies, parts_out = [], [], []
    for name, clef, snm in VOICE_ORDER:
        notes = part_notes.get(name) or []
        if not notes:
            continue
        body, events = quantize(notes, name)
        headers.append(
            f'V: {name} clef={clef} name="{name}" snm="{snm}"')
        bodies.extend([f"V: {name}", body])
        parts_out.append({
            "name": name, "clef": clef, "snm": snm,
            "events": events,
        })

    if not parts_out:
        raise ValueError("No notes to render on the staff")

    abc = "\n".join([
        "X:1",
        f"M:{context['meter']}",
        "L:1/16",
        f"Q:{context['q_num']}/{context['q_den']}={context['tempo']}",
        *headers,
        f"K:{context['key']}",
        *bodies,
    ]) + "\n"
    meta = {
        "version": 7,
        "duration": round(float(duration), 4),
        "tempo": int(context["tempo"]),
        "meter": context["meter"],
        "key": context["key"],
        "bars": n_bars,
        "bar_seconds": round(bar_seconds, 6),
        "measures": [
            {"bar": index, "start": round(start, 4), "end": round(end, 4)}
            for index, (start, end) in enumerate(bounds)
        ],
        "parts": parts_out,
    }
    return abc, meta


# ---------------------------------------------------------------------------
# piano sample rendering
# ---------------------------------------------------------------------------
def _sample_midi(path: Path) -> int:
    match = re.match(r"([A-G])([#sb]?)(-?\d+)", path.stem)
    if not match:
        raise ValueError(path)
    semi = SAMPLE_NAMES[match.group(1)]
    if match.group(2) in ("#", "s"):
        semi += 1
    elif match.group(2) == "b":
        semi -= 1
    return (int(match.group(3)) + 1) * 12 + semi


@functools.lru_cache(maxsize=1)
def _samples():
    return sorted((_sample_midi(p), p) for p in SAMPLE_DIR.glob("*.mp3"))


@functools.lru_cache(maxsize=256)
def _shifted(sample_m: int, target: int):
    path = next(p for m, p in _samples() if m == sample_m)
    y, _ = librosa.load(path, sr=SR, mono=True)
    ratio = 2.0 ** ((target - sample_m) / 12.0)
    if abs(ratio - 1.0) > 1e-4:
        y = librosa.resample(y, orig_sr=SR, target_sr=SR / ratio)
    return y.astype(np.float32)


def _nearest(target: int):
    return min(_samples(), key=lambda sm: (abs(sm[0] - target), sm[0] < target))


def _add_note(buf, midi, start, dur, vel, hand):
    sample_m, _ = _nearest(midi)
    y = _shifted(sample_m, midi)
    tail = min(len(y), int((min(dur, 2.5) + 1.6) * SR))
    if tail <= 0:
        return
    y = y[:tail].copy()
    release = int(dur * SR)
    if release < len(y):
        y[release:] *= np.linspace(1.0, 0.0, len(y) - release) ** 1.4
    y *= (max(1, int(vel)) / 127.0) ** 1.7 * 0.55
    pan = 0.10 if hand == "R" else -0.10
    left = y * math.sqrt((1.0 - pan) / 2.0)
    right = y * math.sqrt((1.0 + pan) / 2.0)
    start_index = int(start * SR)
    end_index = min(len(buf), start_index + len(y))
    if start_index < 0 or end_index <= start_index:
        return
    buf[start_index:end_index, 0] += left[:end_index - start_index]
    buf[start_index:end_index, 1] += right[:end_index - start_index]


def render_notes(rh, lh, out_path: Path, duration=60.0):
    buf = np.zeros((int(duration * SR), 2), dtype=np.float32)
    for note in rh:
        _add_note(buf, int(note["pitch"]), float(note["start"]),
                  float(note["end"]) - float(note["start"]), 78, "R")
    for note in lh:
        _add_note(buf, int(note["pitch"]), float(note["start"]),
                  float(note["end"]) - float(note["start"]),
                  int(note.get("velocity", 52)), "L")
    fade = int(0.6 * SR)
    buf[:fade] *= np.linspace(0, 1, fade)[:, None]
    buf[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 1.2
    peak = float(np.max(np.abs(buf))) or 1.0
    sf.write(out_path, buf / peak * 0.89, SR, subtype="PCM_24")
    return out_path


# ---------------------------------------------------------------------------
# A: ByteDance piano transcription
# ---------------------------------------------------------------------------
def transcribe_a(audio_path: str):
    from piano_transcription_inference import PianoTranscription

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    transcription = PianoTranscription(checkpoint_path=str(CHECKPOINT),
                                       device="cuda")
    result = transcription.transcribe(audio, None)
    notes = []
    for note in result["est_note_events"]:
        notes.append({
            "pitch": int(note["midi_note"]),
            "start": round(float(note["onset_time"]), 4),
            "end": round(float(note["offset_time"]), 4),
            "velocity": int(note["velocity"]),
        })
    del transcription
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return notes


# ---------------------------------------------------------------------------
# B: SheetSage2 melody + generated chords/bass
# ---------------------------------------------------------------------------
def load_chords_with_labels(session_dir: Path):
    """Return (start, end, label, root, intervals) from chord.lab."""
    path = Path(session_dir) / "chord.lab"
    if not path.exists():
        return []
    chords = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            start, end = float(parts[0]), float(parts[1])
            label = parts[2].strip()
        except Exception:
            continue
        try:
            root, bitmap, _ = mir_eval.chord.encode(
                label, reduce_extended_chords=True)
            intervals = [int(x) for x in np.flatnonzero(bitmap)]
        except Exception:
            root = 0
            intervals = [0, 3, 7] if "min" in label else [0, 4, 7]
        chords.append((start, end, label, int(root), intervals))
    return chords


def load_chords(session_dir: Path):
    return [(start, end, root, intervals)
            for start, end, _label, root, intervals
            in load_chords_with_labels(session_dir)]


def _fold_register(pitch: int, low: int, high: int) -> int:
    """Fold a pitch by octaves into [low, high]."""
    while pitch > high:
        pitch -= 12
    while pitch < low:
        pitch += 12
    return int(pitch)


def generate_accompaniment(chords, duration: float, context: dict,
                           *, with_rh: bool = False):
    """Create RH/LH arpeggios and a bass line from chord.lab.

    This is an arrangement, not recognition.  The old implementation put every
    generated chord tone in LH, which left the RH staff empty for songs whose
    recognized Ins melody is sparse.  ``with_rh`` adds a high chord-tone
    layer (C4-C5) so the right-hand piano staff carries accompaniment while
    the recognized Vocal lane remains separate.
    """
    rh, lh, bass = [], [], []
    if not chords:
        return rh, lh, bass
    step = max(0.25, min(0.6, float(context["unit_seconds"]) * 2.0))
    bar_seconds = float(context.get("bar_seconds") or (60.0 / 72.0))
    for raw_start, raw_end, root, intervals in chords:
        start = max(0.0, float(raw_start))
        end = min(float(duration), float(raw_end))
        if end <= start + 0.05:
            continue
        base = 36 + int(root) % 12
        third = intervals[1] if len(intervals) > 1 else 4
        fifth = intervals[2] if len(intervals) > 2 else 7

        # LH: C3-C4 chord arpeggio.
        root3 = _fold_register(base + 12, 45, 60)
        third3 = _fold_register(root3 + third, 45, 60)
        fifth3 = _fold_register(root3 + fifth, 45, 60)
        oct3 = _fold_register(root3 + 12, 45, 60)
        pattern_lh = [root3, fifth3, oct3, third3, fifth3, oct3]

        # RH: C4-C5 high chord-tone layer (only for vocal songs, where the
        # Ins melody is sparse and otherwise leaves RH visually empty).
        root4 = _fold_register(base + 24, 60, 76)
        third4 = _fold_register(root4 + third, 60, 76)
        fifth4 = _fold_register(root4 + fifth, 60, 76)
        oct4 = _fold_register(root4 + 12, 60, 76)
        pattern_rh = [root4, third4, fifth4, oct4, fifth4, third4]

        cursor, index = start, 0
        while cursor < end - 0.04:
            pitch = pattern_lh[index % len(pattern_lh)]
            lh.append({
                "pitch": int(pitch),
                "start": round(cursor, 4),
                "end": round(min(end, cursor + step * 0.92), 4),
                "velocity": 46 + (index % 3) * 3,
            })
            if with_rh and index % 2 == 0:
                rh_pitch = pattern_rh[(index // 2) % len(pattern_rh)]
                rh.append({
                    "pitch": int(rh_pitch),
                    "start": round(cursor, 4),
                    "end": round(min(end, cursor + step * 1.7), 4),
                    "velocity": 44 + (index % 3) * 2,
                })
            cursor += step
            index += 1

        # Bass: root / fifth pulses, about every two bars.
        segment_seconds = max(1.0, bar_seconds * 2.0)
        segments = max(1, int(round((end - start) / segment_seconds)))
        segment = (end - start) / segments
        for index in range(segments):
            note_start = start + index * segment
            pitch = base if index % 2 == 0 else base + fifth
            bass.append({
                "pitch": int(pitch),
                "start": round(note_start, 4),
                "end": round(min(end, note_start + min(segment * 0.9, 2.2)), 4),
                "velocity": 44 + (index % 2) * 4,
            })
    return rh, lh, bass


def _playback_notes(playback: dict, prefix: str):
    notes = []
    for track in playback.get("tracks", []):
        if track.get("name", "").lower().startswith(prefix):
            notes.extend(normalize_note(n) for n in track.get("notes", []))
    notes.sort(key=lambda n: n["start"])
    return notes


def build_b(session_dir: Path, duration: float | None = None,
            context: dict | None = None):
    session_dir = Path(session_dir)
    playback = json.loads((session_dir / "playback.json").read_text(
        encoding="utf-8"))
    rh = _playback_notes(playback, "ins")
    if context is None:
        context = score_context(session_dir)
    if duration is None:
        duration = float(playback.get("duration") or
                         session_duration(session_dir))
    chords = load_chords(session_dir)
    _, lh, bass = generate_accompaniment(chords, duration, context)
    return rh, lh, bass


# ---------------------------------------------------------------------------
# staff artifact helpers
# ---------------------------------------------------------------------------
MIDI_NAMES = ("C", "D♭", "D", "E♭", "E", "F",
              "G♭", "G", "A♭", "A", "B♭", "B")


def midi_name(midi: int) -> str:
    midi = int(midi)
    return f"{MIDI_NAMES[midi % 12]}{midi // 12 - 1}"


def _chord_at(chords, time: float):
    """Find the chord.lab row covering ``time``; returns row or None."""
    if not chords:
        return None
    starts = [row[0] for row in chords]
    index = bisect.bisect_right(starts, time) - 1
    index = max(0, min(len(chords) - 1, index))
    row = chords[index]
    if time < row[0] - 0.05 or time > row[1] + 0.05:
        return None
    return row


def build_bass_details(session_dir: Path, meta: dict, source: str):
    """Per-note bass explanation for the UI table below the staff.

    ``source='generated'`` marks root/fifth roles from chord.lab;
    ``source='recognized'`` is a real transcribed bass note.
    """
    bass = next((part for part in meta.get("parts", [])
                 if part.get("name") == "Bass"), None)
    if not bass:
        return []
    chords = load_chords_with_labels(session_dir)
    details = []
    for event in bass.get("events", []):
        chord = _chord_at(chords, float(event.get("start", 0)))
        label = chord[2] if chord else ""
        root = chord[3] if chord else 0
        intervals = chord[4] if chord else [0, 4, 7]
        base = 36 + int(root) % 12
        fifth = intervals[2] if len(intervals) > 2 else 7
        for pitch in event.get("pitches", []):
            if source == "generated":
                if int(pitch) == base:
                    role = "root"
                elif int(pitch) == base + fifth:
                    role = "fifth"
                elif int(pitch) == base + 12:
                    role = "octave"
                else:
                    role = "chord tone"
            else:
                role = "recognized"
            details.append({
                "time": round(float(event.get("start", 0)), 3),
                "bar": event.get("bar"),
                "pitch": int(pitch),
                "note": midi_name(pitch),
                "chord": label,
                "role": role,
                "source": source,
            })
    details.sort(key=lambda row: (row["time"], row["pitch"]))
    return details


def write_staff_artifacts(session_dir: Path, tag: str, part_notes: dict,
                          context: dict, duration: float, *, title: str,
                          measure_bounds=None, bass_source: str | None = None):
    session_dir = Path(session_dir)
    abc, meta = build_staff_abc(
        part_notes, context, duration, title=title,
        measure_bounds=measure_bounds)
    if bass_source:
        meta["bass_details"] = build_bass_details(
            session_dir, meta, bass_source)
    (session_dir / f"{tag}_staff.abc").write_text(abc, encoding="utf-8")
    (session_dir / f"{tag}_staff_events.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return abc, meta


def _load_cached_staff(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return meta if int(meta.get("version") or 0) >= 7 else None


def rebuild_piano_staff(session_dir: Path, tag: str) -> dict:
    """Rebuild a three-track staff for an older A/B session."""
    session_dir = Path(session_dir)
    context = score_context(session_dir)
    duration = session_duration(session_dir)
    bounds = load_measure_bounds(session_dir)
    if tag.upper() == "A":
        raw = json.loads((session_dir / "piano_A_notes.json").read_text(
            encoding="utf-8"))
        rh, lh, bass = split_piano_notes(raw)
        title = "Piano A - transcription (RH/LH/Bass)"
        tag_name = "piano_A"
    else:
        rh, lh, bass = build_b(session_dir, duration, context)
        title = "Piano B - melody + generated LH/Bass"
        tag_name = "piano_B"
        (session_dir / "piano_B_notes.json").write_text(json.dumps({
            "version": 2,
            "tracks": [{"name": "RH", "notes": rh},
                       {"name": "LH", "notes": lh},
                       {"name": "Bass", "notes": bass}],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    _, meta = write_staff_artifacts(
        session_dir, tag_name,
        {"RH": rh, "LH": lh, "Bass": bass}, context, duration, title=title,
        measure_bounds=bounds,
        bass_source="recognized" if tag.upper() == "A" else "generated")

    # Keep the A/B comparison card in sync with the new three-track split.
    metrics_path = session_dir / "piano_metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            key = "A" if tag.upper() == "A" else "B"
            notes = rh + lh + bass
            metrics.setdefault(key, {})["notes"] = len(notes)
            metrics[key]["range"] = _pitch_range(notes)
            if key == "A":
                metrics[key]["label"] = "A · 钢琴转谱（识别）"
                metrics[key]["description"] = (
                    "ByteDance note+pedal CRNN; RH/LH/Bass split by register")
            else:
                metrics[key]["label"] = (
                    "B · SheetSage2 旋律 + 生成左手/低音（编曲）")
                metrics[key]["description"] = (
                    "RH from SheetSage2 melody + generated LH chords and "
                    "bass from chord.lab")
            metrics["coverage"] = (
                "A is a real transcription; B LH/Bass are generated")
            metrics_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=1),
                encoding="utf-8")
        except Exception:
            pass
    return meta


def build_sheet_staff(session_dir: Path) -> dict:
    """Build/cache the SheetSage2 multi-track staff.

    Vocal is included only when the recognized vocal track has notes, so an
    instrumental song still shows exactly RH/LH/Bass.  For vocal songs, RH =
    recognized Ins melody + generated high accompaniment, LH = generated
    mid-register chords, Bass = generated root/fifth; this prevents the old
    "empty RH, everything in LH" layout.
    """
    session_dir = Path(session_dir)
    meta_path = session_dir / "sheet_staff_events.json"
    cached = _load_cached_staff(meta_path)
    if cached is not None:
        return cached
    context = score_context(session_dir)
    playback = json.loads((session_dir / "playback.json").read_text(
        encoding="utf-8"))
    duration = float(playback.get("duration") or
                     session_duration(session_dir))
    vocal = _playback_notes(playback, "vocal")
    ins = _playback_notes(playback, "ins")
    chords = load_chords(session_dir)
    with_rh = bool(vocal)
    rh_acc, lh, bass = generate_accompaniment(
        chords, duration, context, with_rh=with_rh)
    rh = sorted(ins + rh_acc, key=lambda n: n["start"]) if with_rh else ins
    _, meta = write_staff_artifacts(
        session_dir, "sheet",
        {"Vocal": vocal, "RH": rh, "LH": lh, "Bass": bass},
        context, duration,
        title="SheetSage2 - Vocal + RH/LH/Bass",
        measure_bounds=load_measure_bounds(session_dir),
        bass_source="generated")
    return meta


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _chroma(path):
    y, _ = librosa.load(str(path), sr=22050, mono=True)
    c = librosa.feature.chroma_cqt(y=y, sr=22050, hop_length=512)
    return c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)


def _metrics(original, a_audio, b_audio):
    from librosa.sequence import dtw
    co = _chroma(original)
    out = {}
    for key, path in (("A", a_audio), ("B", b_audio)):
        c = _chroma(path)
        D, wp = dtw(co, c, subseq=True, metric="cosine")
        end = int(np.argmin(D[-1]))
        h1, h2 = co.mean(axis=1), c.mean(axis=1)
        out[key] = {
            "chroma_dtw": round(float(D[-1, end] / len(wp)), 4),
            "chroma_hist": round(float(np.dot(h1, h2) /
                (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-9)), 4),
        }
    return out


def _pitch_range(notes) -> str:
    pitches = [int(n["pitch"]) for n in notes]
    return f"{min(pitches)}-{max(pitches)}" if pitches else "—"


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------
def run_bundle(audio_path: str, session_dir: Path):
    session_dir = Path(session_dir)
    audio = next(p for p in session_dir.iterdir() if is_original_audio(p))
    duration = session_duration(session_dir)
    context = score_context(session_dir)
    bounds = load_measure_bounds(session_dir)

    a_notes = transcribe_a(str(audio))
    rh_a, lh_a, bass_a = split_piano_notes(a_notes)

    rh_b, lh_b, bass_b = build_b(session_dir, duration, context)
    a_audio = render_notes(rh_a, lh_a + bass_a,
                           session_dir / "piano_A_synth.wav", duration)
    b_audio = render_notes(rh_b, lh_b + bass_b,
                           session_dir / "piano_B.wav", duration)

    write_staff_artifacts(
        session_dir, "piano_A",
        {"RH": rh_a, "LH": lh_a, "Bass": bass_a}, context, duration,
        title="Piano A - transcription (RH/LH/Bass)",
        measure_bounds=bounds, bass_source="recognized")
    write_staff_artifacts(
        session_dir, "piano_B",
        {"RH": rh_b, "LH": lh_b, "Bass": bass_b}, context, duration,
        title="Piano B - melody + generated LH/Bass",
        measure_bounds=bounds, bass_source="generated")

    (session_dir / "piano_A_notes.json").write_text(
        json.dumps(a_notes, indent=1), encoding="utf-8")
    (session_dir / "piano_B_notes.json").write_text(json.dumps({
        "version": 2,
        "tracks": [{"name": "RH", "notes": rh_b},
                   {"name": "LH", "notes": lh_b},
                   {"name": "Bass", "notes": bass_b}],
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    try:
        m = _metrics(str(audio), str(a_audio), str(b_audio))
    except Exception:
        m = {"A": {"chroma_dtw": None, "chroma_hist": None},
             "B": {"chroma_dtw": None, "chroma_hist": None}}
    metrics = {
        "A": {"label": "A · 钢琴转谱（识别）",
              "description": "ByteDance note+pedal CRNN; RH/LH/Bass split by register",
              "notes": len(a_notes),
              "range": _pitch_range(a_notes),
              **m["A"]},
        "B": {"label": "B · SheetSage2 旋律 + 生成左手/低音（编曲）",
              "description": "RH from SheetSage2 melody + generated LH chords and bass from chord.lab",
              "notes": len(rh_b) + len(lh_b) + len(bass_b),
              "range": _pitch_range(rh_b + lh_b + bass_b),
              **m["B"]},
        "coverage": "A is a real transcription; B LH/Bass are generated",
    }
    (session_dir / "piano_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    return metrics
