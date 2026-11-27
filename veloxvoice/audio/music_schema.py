# Vendored from m-a-p/SheetSage2 (CC-BY-NC-4.0)
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mir_eval.chord


@dataclass(frozen=True)
class PromptTaskSpec:
    name: str
    sampling_group: str
    output_field: str


@dataclass(frozen=True)
class TokenBlockSpec:
    name: str
    labels: tuple[str, ...]
    output_field: str | None = None


@dataclass(frozen=True)
class PromptMultitaskSchema:
    version: str
    tasks: tuple[PromptTaskSpec, ...]
    event_field_order: tuple[str, ...]
    appended_token_blocks: tuple[TokenBlockSpec, ...] = ()


V1_TASKS = (
    PromptTaskSpec("timestamp", "timestamp", "timestamp"),
    PromptTaskSpec("downbeat_meter", "rhythm", "rhythm"),
    PromptTaskSpec("structure", "structure", "structure"),
    PromptTaskSpec("key", "key", "key"),
    PromptTaskSpec("chord_majmin", "chord", "chord"),
    PromptTaskSpec("chord_full", "chord", "chord"),
    PromptTaskSpec("melody_vocal", "melody", "melody"),
    PromptTaskSpec("melody_full", "melody", "melody"),
)


PROMPT_MULTITASK_SCHEMAS = {
    "v1": PromptMultitaskSchema(
        version="v1",
        tasks=V1_TASKS,
        event_field_order=(
            "timestamp",
            "rhythm",
            "structure",
            "key",
            "chord",
            "melody",
        ),
    ),
}


def _validate_schema(schema):
    task_names = tuple(task.name for task in schema.tasks)
    if len(task_names) != len(set(task_names)):
        raise ValueError(f"Duplicate task name in schema {schema.version!r}")
    field_names = tuple(schema.event_field_order)
    if len(field_names) != len(set(field_names)):
        raise ValueError(f"Duplicate output field in schema {schema.version!r}")
    missing_fields = {
        task.output_field for task in schema.tasks if task.output_field not in field_names
    }
    if missing_fields:
        raise ValueError(
            f"Tasks in schema {schema.version!r} use unknown output fields: "
            f"{tuple(sorted(missing_fields))}"
        )
    block_names = tuple(block.name for block in schema.appended_token_blocks)
    if len(block_names) != len(set(block_names)):
        raise ValueError(f"Duplicate token block in schema {schema.version!r}")
    for block in schema.appended_token_blocks:
        if not block.labels or len(block.labels) != len(set(block.labels)):
            raise ValueError(
                f"Token block {block.name!r} must contain unique labels"
            )


def register_prompt_multitask_schema(schema):
    """Register a frozen schema version without permitting redefinition."""
    if not isinstance(schema, PromptMultitaskSchema):
        raise TypeError("schema must be a PromptMultitaskSchema")
    _validate_schema(schema)
    existing = PROMPT_MULTITASK_SCHEMAS.get(schema.version)
    if existing is not None and existing != schema:
        raise ValueError(f"Schema {schema.version!r} is already registered")
    PROMPT_MULTITASK_SCHEMAS[schema.version] = schema
    return schema


def extend_prompt_multitask_schema(
    base_version,
    version,
    tasks=(),
    event_fields=(),
    token_blocks=(),
):
    """Append tasks and vocabulary blocks while preserving every base token id."""
    base = get_prompt_multitask_schema(base_version)
    schema = PromptMultitaskSchema(
        version=str(version),
        tasks=base.tasks + tuple(tasks),
        event_field_order=base.event_field_order + tuple(event_fields),
        appended_token_blocks=base.appended_token_blocks + tuple(token_blocks),
    )
    return register_prompt_multitask_schema(schema)


def get_prompt_multitask_schema(version):
    version = str(version)
    try:
        return PROMPT_MULTITASK_SCHEMAS[version]
    except KeyError as exc:
        raise ValueError(
            f"Unknown prompt multitask schema {version!r}; "
            f"available={tuple(PROMPT_MULTITASK_SCHEMAS)}"
        ) from exc


_validate_schema(PROMPT_MULTITASK_SCHEMAS["v1"])


# ---- durations (vendored) ----


DURATION_TEMPLATES = np.array(
    [
        1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
        96, 128, 192, 256, 384, 512, 768, 1024,
        1536, 2048, 3072, 4096,
    ],
    dtype=np.int32,
)
duration_boundaries = (DURATION_TEMPLATES[:-1] + DURATION_TEMPLATES[1:]) / 2



# ---- labels (vendored) ----



STRUCTURE_LABELS = ['silence', 'intro', 'outro', 'verse', 'chorus', 'bridge', 'pre-chorus', 'post-chorus',
                                'interlude', 'fade-out', 'loop', 'rap', 'preshot', 'irregular']


STRUCTURE_LABELS = ['silence', 'intro', 'outro', 'verse', 'chorus', 'bridge', 'pre-chorus', 'post-chorus',
                               'interlude', 'fade-out', 'loop', 'rap', 'preshot', 'irregular', 'instrumental',
                               'intro and verse', 'pre-chorus and chorus', 'verse and pre-chorus', 'solo', 'theme', 'development', 'variation', 'pre-outro']


MIREX_STRUCTURE_LABEL_REDUCTION = {
    'silence': 'silence',
    'intro': 'intro',
    'outro': 'outro',
    'verse': 'verse',
    'chorus': 'chorus',
    'bridge': 'bridge',
    'pre-chorus': 'verse',
    'post-chorus': 'verse',
    'interlude': 'inst',
    'inst': 'inst',
    'fade-out': 'outro',
    'loop': 'chorus',
    'rap': 'verse',
    'preshot': 'inst',
    'irregular': 'verse',
}

# ---- chord spelling (vendored) ----
"""Local-key chord spelling, ported from SheetSage2 LM."""

KEY_MAP = [
    ['C:major', 'Db:major', 'D:major', 'Eb:major', 'E:major', 'F:major', 'F#:major', 'G:major', 'Ab:major', 'A:major', 'Bb:major', 'B:major'],
    ['D:dorian', 'Eb:dorian', 'E:dorian', 'F:dorian', 'F#:dorian', 'G:dorian', 'G#:dorian', 'A:dorian', 'Bb:dorian', 'B:dorian', 'C:dorian', 'C#:dorian'],
    ['E:phrygian', 'F:phrygian', 'F#:phrygian', 'G:phrygian', 'G#:phrygian', 'A:phrygian', 'A#:phrygian', 'B:phrygian', 'C:phrygian', 'C#:phrygian', 'D:phrygian', 'D#:phrygian'],
    ['F:lydian', 'Gb:lydian', 'G:lydian', 'Ab:lydian', 'A:lydian', 'Bb:lydian', 'B:lydian', 'C:lydian', 'Db:lydian', 'D:lydian', 'Eb:lydian', 'E:lydian'],
    ['G:mixolydian', 'Ab:mixolydian', 'A:mixolydian', 'Bb:mixolydian', 'B:mixolydian', 'C:mixolydian', 'C#:mixolydian', 'D:mixolydian', 'Eb:mixolydian', 'E:mixolydian', 'F:mixolydian', 'F#:mixolydian'],
    ['A:minor', 'Bb:minor', 'B:minor', 'C:minor', 'C#:minor', 'D:minor', 'D#:minor', 'E:minor', 'F:minor', 'F#:minor', 'G:minor', 'G#:minor'],
    ['B:locrian', 'C:locrian', 'C#:locrian', 'D:locrian', 'D#:locrian', 'E:locrian', 'E#:locrian', 'F#:locrian', 'G:locrian', 'G#:locrian', 'A:locrian', 'A#:locrian']
]

MODE_NAMES = ['major', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'minor', 'locrian']
MODE_STARTS = [0, 2, 4, 5, 7, 9, 11]


QUALITIES = {
    #           1     2     3     4  5     6     7
    'maj':     [2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':     [2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'aug':     [2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'dim':     [2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4':    [2, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus4(b7)':[2, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'sus4(b7,9)':[2, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'sus2':    [2, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':       [2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7':    [2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7':    [2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'minmaj7': [2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'maj6':    [2, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6':    [2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    '9':       [2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj9':    [2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min9':    [2, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '7(b9)':   [2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '7(#9)':   [2, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj6(9)': [2, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6(9)': [2, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    'maj(9)':  [2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min(9)':  [2, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'maj(11)': [2, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'min(11)': [2, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    '11':      [2, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    'maj9(11)':[2, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'min11':   [2, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    '13':      [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    'maj13':   [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    'min13':   [2, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    'dim7':    [2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'hdim7':   [2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    #'5':       [2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    }
DEFAULT_SPELLING = ['1', 'b2', '2', 'b3', '3', '4', 'b5', '5', '#5', '6', 'b7', '7']
CHORD_SPELLING_TABLE = None
CIRCLE_OF_FIFTH = np.array([3, 0, 5, 1, 6, 2, 7])
CIRCLE_OF_FIFTH_INV = np.array([1, 3, 5, 0, 2, 4, 6])
NOTE_NAMES = 'CDEFGAB'

def scale_degree_to_tuple(scale_degree):
    offset = 0
    if scale_degree.startswith("#"):
        offset = scale_degree.count("#")
        scale_degree = scale_degree.strip("#")
    elif scale_degree.startswith("b"):
        offset = -1 * scale_degree.count("b")
        scale_degree = scale_degree.strip("b")
    return np.array([int(scale_degree) - 1, offset], dtype=int)

def note_name_to_tuple(note_name):
    scale_degree = NOTE_NAMES.index(note_name[0])
    offset = 0
    if note_name.endswith("#"):
        offset = note_name.count("#")
    elif note_name.endswith("b"):
        offset = -1 * note_name.count("b")
    return np.array([scale_degree, offset], dtype=int)

def scale_degrees_to_tuple(scale_degrees):
    return np.stack([scale_degree_to_tuple(sd) for sd in scale_degrees], axis=0)


def get_chord_spelling_table():
    global CHORD_SPELLING_TABLE
    if CHORD_SPELLING_TABLE is not None:
        return CHORD_SPELLING_TABLE
    chord_spelling_table = {}
    for quality in QUALITIES:
        quality_spelling = []
        chroma = QUALITIES[quality]
        for i in range(12):
            if chroma[i] > 0:
                if '#9' in quality and i == DEFAULT_SPELLING.index('b3'):
                    quality_spelling.append('#2')
                elif 'dim7' in quality and i == DEFAULT_SPELLING.index('6'):
                    quality_spelling.append('bb7')
                else:
                    quality_spelling.append(DEFAULT_SPELLING[i])
        chord_spelling_table[quality] = scale_degrees_to_tuple(quality_spelling)
    CHORD_SPELLING_TABLE = chord_spelling_table
    return chord_spelling_table

# Circle-of-fifths offsets of major-scale degrees 1-7 from the tonic.
MAJOR_SCALE_FIFTHS = np.array([0, 2, 4, -1, 1, 3, 5])

def spell_chord_tones(root_spelling, quality):
    '''
    Spell every chord tone from a root spelling.
    Template intervals are relative to the root's own major scale, so adding
    them on the circle of fifths keeps each tone's pitch class; e.g., E:maj
    gives E, G#, B, and Db:hdim7 gives Db, Fb, Abb, Cb.
    :param root_spelling: Numpy array of shape (2,) with note letter id and accidental offset.
    :param quality: Chord quality in QUALITIES.
    :return: Numpy array of shape (N, 2); the first row is the root.
    '''
    intervals = get_chord_spelling_table()[quality]
    letters = (root_spelling[0] + intervals[:, 0]) % 7
    root_pos = CIRCLE_OF_FIFTH_INV[root_spelling[0]] + root_spelling[1] * 7
    tone_pos = root_pos + MAJOR_SCALE_FIFTHS[intervals[:, 0]] + intervals[:, 1] * 7
    return np.stack([letters, (tone_pos - CIRCLE_OF_FIFTH_INV[letters]) // 7], axis=1)

def score_spelling_under_key(spelling, key):
    '''
    Score the chord spelling under a given key.
    :param spelling: Numpy array of shape (N, 2)
    :param key: Numpy array of shape (2,) representing the degree of 1 (DO)
    :return: Score of the chord spelling under the key.
    '''
    key_pos = CIRCLE_OF_FIFTH_INV[key[0]] + key[1] * 7
    spelling_pos = CIRCLE_OF_FIFTH_INV[spelling[:, 0] % 7] + spelling[:, 1] * 7
    relative_pos = np.clip(np.abs((spelling_pos - (key_pos + 2))) - 3, 0, None)
    return np.sum(relative_pos) + np.sum(relative_pos[:1]) * 1.0  # root matters more

def normalize_key_name(key_name):
    """Use the same canonical tonal spelling as SheetSage2 LM/NoLM."""
    tonic, mode = key_name.split(':')
    mode_id = MODE_NAMES.index(mode)
    scale = (mir_eval.chord.pitch_class_to_semitone(tonic) - MODE_STARTS[mode_id]) % 12
    return KEY_MAP[mode_id][scale]


def correct_chord_spelling(label, key_name):
    if label == 'N' or label == 'X':
        return label
    key_tonal, key_mode = key_name.split(':')
    key_mode = MODE_NAMES.index(key_mode)
    scale_semitone = (mir_eval.chord.pitch_class_to_semitone(key_tonal) - MODE_STARTS[key_mode]) % 12
    # Tokenizer keys use sharps; map enharmonic aliases to the LM key signature.
    key_spelling = note_name_to_tuple(KEY_MAP[0][scale_semitone].split(':')[0])
    # Ignore all inversions
    inversion = ''
    if '/' in label:
        label, inversion = label.split('/')
    # Separate quality and root
    root, quality = label.split(':')
    root_semitone = mir_eval.chord.pitch_class_to_semitone(root)
    scores = []
    for possible_root_id in range(7):
        possible_root = NOTE_NAMES[possible_root_id]
        root_spelling = np.array([
            possible_root_id,
            (root_semitone - mir_eval.chord.pitch_class_to_semitone(possible_root) + 6) % 12 - 6
        ])
        chord_spelling = spell_chord_tones(root_spelling, quality)
        scores.append(score_spelling_under_key(chord_spelling, key_spelling))
    best_root_id = np.argmin(scores)
    best_root = NOTE_NAMES[best_root_id]
    best_root_spelling = np.array([
        best_root_id,
        (root_semitone - mir_eval.chord.pitch_class_to_semitone(best_root) + 6) % 12 - 6
    ])
    if best_root_spelling[1] < 0:
        best_root = best_root + 'b' * abs(best_root_spelling[1])
    else:
        best_root = best_root + '#' * best_root_spelling[1]
    if inversion:
        return f"{best_root}:{quality}/{inversion}"
    else:
        return f"{best_root}:{quality}"


def correct_chord_rows(chords, keys):
    """Use the local key at each chord midpoint, as in SheetSage2 LM.

    Match LM's left-boundary tie rule. Extend the first/last decoded key to
    uncovered edges. Without a decoded key, preserve the original spelling.
    Return new rows without changing event values, timing, quality or inversion.
    """
    if not keys:
        return [list(row) for row in chords]
    boundaries = np.array([row[1] for row in keys[:-1]])
    return [[start, end, correct_chord_spelling(
        label, keys[np.searchsorted(boundaries, (start + end) / 2.0)][2])]
        for start, end, label in chords]