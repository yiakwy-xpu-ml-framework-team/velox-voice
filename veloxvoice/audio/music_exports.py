# Vendored from m-a-p/SheetSage2 (CC-BY-NC-4.0)
"""Raw decoded events, key-aware chord LAB, MIDI and validated ABC."""
from pathlib import Path
import json
from fractions import Fraction

import numpy as np
import pretty_midi

import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path


def _fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_text(path, text):
    """Write text through a temp file + rename so readers never see a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, os.stat(tmp).st_mode | stat.S_IWUSR)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_directory(path.parent)


from .music_abc import generate_abc_from_data
def field_text(event, tokenizer):
    # (inlined from the generation module — pure event formatting)
    parts = []
    for field in tokenizer.event_field_order:
        value = event["values"].get(field)
        if value is None:
            continue
        if field == "melody":
            note_parts = []
            for note in value:
                note_parts.append(
                    f"pitch={note['pitch']}:track={note['track']}"
                    f":dur_bin={note['duration_bin']}:dur_steps={note['duration_steps']}"
                )
            parts.append("melody=[" + ",".join(note_parts) + "]")
        elif isinstance(value, dict):
            parts.append(field + "=" + ",".join(f"{k}:{v}" for k, v in value.items()))
        else:
            parts.append(f"{field}={value}")
    return " ".join(parts)
from .music_schema import correct_chord_rows, normalize_key_name
from .music_midi import build_playback, midi_bytes


def rows_text(rows):
    return "".join("\t".join(str(x) for x in row) + "\n" for row in rows)


def write_rows(path, rows):
    atomic_write_text(path, rows_text(rows))


def interval_rows(events, field, duration):
    rows = [[e["time"], 0, e["values"][field]] for e in events if field in e["values"]]
    for i, row in enumerate(rows):
        row[1] = rows[i + 1][0] if i + 1 < len(rows) else duration
    return [r for r in rows if r[1] > r[0]]


def rhythm_rows(events):
    rows, meter = [], None
    for event in events:
        rhythm = event["values"].get("rhythm", {})
        meter = rhythm.get("meter", meter)
        eighth = rhythm.get("eighth_position")
        if eighth is not None and meter is not None:
            # The model stores eighth-note position, not denominator-beat index.
            position = Fraction(int(eighth) * int(meter[1]), 8)
            if position.denominator != 1:
                raise ValueError(f"Eighth position {eighth} is off the {meter[0]}/{meter[1]} beat grid")
            if not 0 <= position < meter[0]:
                raise ValueError(f"Eighth position {eighth} is outside meter {meter}")
            rows.append([float(event["time"]), int(position) + 1, int(meter[0]), int(meter[1])])
    # Repair beat IDs: enforce consecutive 1,2,...,meter[0] cycling.
    # The AR model may predict slightly off-by-one beat IDs due to bf16
    # near-tie divergence.  Re-number based on sequence index to ensure
    # the beat grid is continuous (required by the ABC notation generator).
    numerator = rows[0][2] if rows else 4
    for i, row in enumerate(rows):
        expected_id = (i % numerator) + 1
        if row[1] != expected_id:
            row[1] = expected_id
    return rows


def _midi(notes, paper=False):
    midi = pretty_midi.PrettyMIDI(resolution=220 if paper else 960)
    names = ((0, "vocal_melody"), (1, "instrumental_melody")) if paper else ((0, "Vocal"), (1, "Ins"))
    for track, name in names:
        instrument = pretty_midi.Instrument(program=0, name=name)
        for start, end, pitch, source in notes:
            if track == source and end > start:
                instrument.notes.append(pretty_midi.Note(100, pitch, start, end))
        midi.instruments.append(instrument)
    return midi


def notation_notes(notes):
    """A monophonic notation view, retaining the exact raw prediction separately."""
    result, diagnostics = [], []
    for track in (0, 1):
        ordered = sorted((list(n) for n in notes if n[3] == track), key=lambda n: (n[0], n[2], n[1]))
        for i, note in enumerate(ordered):
            if i + 1 < len(ordered) and note[1] > ordered[i + 1][0] + 1e-6:
                note[1] = ordered[i + 1][0]
                diagnostics.append(f"notation only: clipped track {track} note at {note[0]:.3f} to next onset")
            if note[1] > note[0] + 1e-6:
                result.append(note)
    return sorted(result), diagnostics


def export_result(decoded, tokenizer, output_dir=None, duration=None, paper=False, *,
                  melody_only=False, playback=True, debug=True):
    """Build ABC, MIDI and LAB in memory, optionally writing the same bytes.

    Statistics retain their file-interface names. ``payload`` contains the
    in-memory ABC text, MIDI bytes, decoded event list, LAB texts and playback.
    melody_only removes chords from notation and playback without changing
    decoded events or the key-corrected LAB annotations.

    ``playback=False`` skips the (multi-step) transcription/playback MIDI
    bundle — the serving lane rebuilds it lazily on first client fetch via
    :func:`export_playback`. ``debug=False`` skips the diagnostics-only
    artifacts (events.json/tsv, .lab files, notation/*) that no consumer
    reads during serving.
    """
    if not isinstance(melody_only, bool):
        raise ValueError("melody_only must be True or False")
    if duration is None:
        raise TypeError("duration is required")
    texts, midis, notation_midis, labs = {}, {}, {}, {}

    def keep_rows(name, rows):
        text = rows_text(rows)
        texts[name] = text
        if name.endswith(".lab"):
            labs[name[:-4]] = text

    events = decoded["events"]
    notes = []
    for event in events:
        start = float(event["time"])
        for note in event["values"].get("melody", ()):
            end = max(start + 0.04, float(note["end_time"])) if paper else min(duration, float(note["end_time"]))
            if end > start:
                notes.append([start, end, int(note["pitch"]), int(note["track"])])
    if not paper:
        notes.sort()
    if debug:
        texts["events.json"] = json.dumps(decoded, indent=2)
        keep_rows("events.tsv", [["time", "global_subbeat", "fields"]] +
                   [[e["time"], e["global_subbeat"], field_text(e, tokenizer)] for e in events])
    midis["melody"] = midi_bytes(_midi(notes, paper=paper))
    lab_notes = [[f"{a:.6f}", f"{b:.6f}", p, t] for a, b, p, t in notes] if paper else notes
    if debug:
        keep_rows("melody_full.lab", lab_notes)
    for track, name in ((0, "vocal"), (1, "instrumental")):
        selected = [n for n in notes if n[3] == track]
        if debug:
            keep_rows(f"melody_{name}.lab", [n[:3] for n in lab_notes if n[3] == track])
        midis[f"melody_{name}"] = midi_bytes(_midi(selected, paper=paper))
    intervals = {}
    for field in ("chord", "key", "structure"):
        rows = interval_rows(events, field, duration)
        if paper:
            rows = [[float(e["time"]), None, e["values"][field]] for e in events if field in e["values"]]
            for i, row in enumerate(rows):
                row[1] = max(row[0], rows[i + 1][0]) if i + 1 < len(rows) else float(duration)
        intervals[field] = rows
    intervals["key"] = [[start, end, normalize_key_name(key)]
                        for start, end, key in intervals["key"]]
    intervals["chord"] = correct_chord_rows(intervals["chord"], intervals["key"])
    # chord.lab is required by the lazy playback-bundle rebuild — always write
    texts["chord.lab"] = rows_text(intervals["chord"])
    labs["chord"] = texts["chord.lab"]
    if debug:
        for field, rows in intervals.items():
            if field != "chord":
                keep_rows(f"{field}.lab", rows)
    if paper:
        rows = []
        for event in events:
            rhythm, stamp = event["values"].get("rhythm"), event["values"].get("timestamp")
            if rhythm is None and stamp is None:
                continue
            meter = rhythm.get("meter") if isinstance(rhythm, dict) else None
            eighth = rhythm.get("eighth_position") if isinstance(rhythm, dict) else None
            rows.append([f"{event['time']:.6f}", "" if stamp is None else f"{float(stamp):.6f}",
                         "" if eighth is None else int(eighth), "" if meter is None else f"{meter[0]}/{meter[1]}"])
        keep_rows("beat_meter.lab", rows)
    if debug:
        raw_rhythm = [[e["time"], json.dumps(e["values"].get("rhythm", {}))]
                      for e in events if "rhythm" in e["values"] or "timestamp" in e["values"]]
        keep_rows("rhythm_events.lab", raw_rhythm)
    diagnostics, abc_error, score, text = [], None, None, None
    try:
        beats = rhythm_rows(events)
        if debug:
            keep_rows("beat.lab", beats)
            keep_rows("downbeat.lab", [[r[0]] for r in beats if r[1] == 1])
        if len(beats) < 2:
            raise ValueError("At least two decoded beats are required for ABC")
        # canonical notation's last beat is a boundary, so append it explicitly. Continue the
        # final tempo only as far as the audio/last note and let it pad the bar.
        abc_beats = [list(b) for b in beats]
        period = float(np.median(np.diff([b[0] for b in beats[-9:]])))
        if period <= 0:
            raise ValueError("Decoded beats must increase in time")
        end = max(duration, max((n[1] for n in notes), default=0))
        while abc_beats[-1][0] < end - 1e-6:
            prev = abc_beats[-1]
            abc_beats.append([prev[0] + period, prev[1] % prev[2] + 1, prev[2], prev[3]])
        if debug:
            keep_rows("notation/song_beats.txt", abc_beats)
        notation_intervals = {}
        for field in ("chord", "key", "structure"):
            # reuse the already-corrected/normalized rows from the LAB pass
            # (a second correct_chord_rows call on identical inputs was pure
            # duplicate work)
            rows = intervals[field]
            # Clip to the actual beat domain: events beyond it have no ABC bin.
            rows = [[max(abc_beats[0][0], a), min(abc_beats[-1][0], b), v]
                    for a, b, v in rows if b > abc_beats[0][0] and a < abc_beats[-1][0]]
            notation_intervals[field] = rows
            if debug:
                keep_rows(f"notation/song_{field}s.txt", rows)
        if not intervals["key"]:
            raise ValueError("No key was decoded; cannot construct a keyed ABC score")
        clean, adjustments = notation_notes(notes)
        diagnostics.extend(adjustments)
        notation_midi = _midi(clean)
        text, score = generate_abc_from_data(
            notation_midi, abc_beats,
            notation_intervals["chord"], notation_intervals["key"],
            notation_intervals["structure"],
            melody_only=melody_only,
        )
        diagnostics.extend(score.diagnostics)
        texts["score.abc"] = text
        abc_measures = len(score.measures)
    except (ValueError, FileNotFoundError) as exc:
        abc_error = str(exc)
        abc_measures = 0
    if playback:
        playback, playback_midis = build_playback(
            midis["melody"], [] if melody_only else intervals["chord"], score, duration)
        midis.update(playback_midis)
        texts["playback.json"] = json.dumps(playback, indent=2)
        diagnostics.extend(playback["warnings"])
    else:
        playback = {}
        # light sync data for the client's score-following: notes + measures
        # straight from memory (no MIDI serialize/parse round-trip). MUST
        # carry the measure map — the lazy rebuild path has no score object
        # and cannot reconstruct it.
        from .music_midi import measure_map

        tracks = [dict(name=name, notes=[dict(pitch=p, start=a, end=b, velocity=100)
                                         for a, b, p, t in notes if t == track])
                  for track, name in ((0, "Vocal"), (1, "Ins"))]
        texts["playback.json"] = json.dumps(dict(
            version=1, duration=duration, tracks=tracks,
            measures=measure_map(score, duration) if score is not None else [],
            warnings=[]), indent=1)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, content in texts.items():
            atomic_write_text(output_dir / name, content)
        for name, content in midis.items():
            path = output_dir / f"{name}.mid"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        if debug:
            for name, content in notation_midis.items():
                path = output_dir / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        if abc_error is not None:
            (output_dir / "score.abc").unlink(missing_ok=True)
        else:
            (output_dir / "score.browser.abc").unlink(missing_ok=True)
    payload = dict(abc=text, midi=midis.get("transcription"), midis=midis,
                   events=events, labs=labs, playback=playback)
    return dict(melody_notes=len(notes), vocal_notes=sum(n[3] == 0 for n in notes),
                instrumental_notes=sum(n[3] == 1 for n in notes), events=len(events),
                abc_measures=abc_measures, abc_error=abc_error, diagnostics=diagnostics, payload=payload)
