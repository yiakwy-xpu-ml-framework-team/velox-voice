"""Grammar-constrained AR decode

Just as ``rescore.py`` implements the DECODE phase of ASR, 
the two modules however use fundamentally different decoding paradigms:

  ASR (rescore.py)             :   CTC beam search → N candidates → decoder SCORES
                                   them in ONE batched forward

  Music (grammar_ar_decode.py) :   Decoder GENERATES tokens one at a time
                                   following abc melody structures

This module is the MODEL-LEVEL AR forward that connects ``sheetsage.py``
(the encoder + decoder weights) and ``mert.py`` with the shared conformer
kernels. 

The exports (ABC/MIDI, ~13% of e2e) live in ``veloxvoice/audio/music_exports.py``.
"""

from __future__ import annotations

import copy
import functools
from contextlib import nullcontext

import numpy as np
import torch

from ..models.sheetsage.v2.config import SheetSage2Config


FULL_TASK_PROMPTS = ("timestamp", "downbeat_meter", "structure", "key",
                     "chord_full", "melody_full")


def inference_autocast(device, dtype=torch.bfloat16):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


class PromptGrammarState:
    """Incremental allowed-token sets over the multitask event schema."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generated_events = 0
        self.in_shift = True
        self.shift_run = 0
        self.payload_count = 0
        self.last_field_index = -1
        self.incomplete = None
        self._field_index = {
            name: i for i, name in enumerate(tokenizer.event_field_order)}

    def _allow_field_starts(self, allowed):
        tok = self.tokenizer
        fi = self._field_index

        if self.last_field_index < fi["timestamp"]:
            allowed[tok.time_token_start:tok.time_token_end] = True

        if self.last_field_index < fi["rhythm"]:
            allowed[tok.meter_token_start:tok.meter_token_end] = True
            allowed[tok.eighth_position_token_start:
                    tok.eighth_position_token_end] = True

        if self.last_field_index < fi["structure"]:
            allowed[tok.structure_token_start:tok.structure_token_end] = True

        if self.last_field_index < fi["key"]:
            allowed[tok.key_token_start:tok.key_token_end] = True

        if self.last_field_index < fi["chord"]:
            allowed[tok.full_chord_token_start:tok.full_chord_token_end] = True

        if self.last_field_index <= fi["melody"]:
            allowed[tok.pitch_token_start:tok.pitch_token_end] = True

    def allowed(self, device):
        tok = self.tokenizer
        allowed = torch.zeros(tok.n_tokens, dtype=torch.bool, device=device)
        can_end = self.payload_count > 0
        if can_end:
            allowed[tok.eos_token] = True

        if self.payload_count > 0 or self.in_shift:
            if self.shift_run < 4:
                allowed[tok.subbeat_shift_token_start:
                        tok.subbeat_shift_token_end] = True

        if self.incomplete == "rhythm_after_meter":
            allowed[tok.eighth_position_token_start:
                    tok.eighth_position_token_end] = True
            return allowed

        if self.incomplete == "melody_after_pitch":
            allowed[tok.duration_token_start:tok.duration_token_end] = True
            allowed[tok.pitch_token_start:tok.pitch_token_end] = True
            return allowed

        self._allow_field_starts(allowed)
        return allowed

    def update(self, token):
        tok = self.tokenizer
        token = int(token)
        token_type = tok.token_type(token)

        if token == tok.eos_token:
            return True

        if token_type == "subbeat_shift":
            if not self.in_shift and self.payload_count > 0:
                self.generated_events += 1
                self.payload_count = 0
                self.last_field_index = -1
                self.incomplete = None
            self.in_shift = True
            self.shift_run += 1
            return False
        
        self.in_shift = False
        self.shift_run = 0
        self.payload_count += 1
        index = self._field_index

        if token_type == "time":
            self.last_field_index = index["timestamp"]
            self.incomplete = None

        elif token_type == "meter":
            self.last_field_index = index["rhythm"]
            self.incomplete = "rhythm_after_meter"

        elif token_type == "eighth_position":
            self.last_field_index = index["rhythm"]
            self.incomplete = None

        elif token_type == "structure":
            self.last_field_index = index["structure"]
            self.incomplete = None

        elif token_type == "key":
            self.last_field_index = index["key"]
            self.incomplete = None

        elif token_type == "chord_full":
            self.last_field_index = index["chord"]
            self.incomplete = None

        elif token_type == "pitch":
            self.last_field_index = index["melody"]
            self.incomplete = "melody_after_pitch"

        elif token_type == "duration":
            self.last_field_index = index["melody"]
            self.incomplete = None
        else:
            raise RuntimeError(f"Unexpected prompt token type {token_type!r}")
        return False


# constrained Audio Symbol Recognition (ASR) generation
@torch.inference_mode()
def constrained_prompt_generate(model, audio, prompts, max_sequence_length,
                                prefix_tokens=None, autocast_dtype=torch.bfloat16,
                                stop_time_seconds=None, memory=None,
                                progress_callback=None):
    """Greedy grammar-constrained decode for one [1, samples] waveform."""
    tokenizer = model.tokenizer
    prefix = list(prefix_tokens) if prefix_tokens is not None \
        else tokenizer.prompt_prefix(prompts)

    if not prefix or prefix[0] != tokenizer.sos_token:
        raise ValueError("generation prefix must begin with <|sos|>")

    if prefix[-1] == tokenizer.eos_token:
        prefix = prefix[:-1]

    if audio.ndim != 2 or audio.shape[0] < 1:
        raise ValueError("audio must have shape [batch, samples]")

    state = PromptGrammarState(tokenizer)

    out_index = prefix.index(tokenizer.out_token)
    for token in prefix[out_index + 1:]:
        state.update(token)

    output_tokens = list(prefix)
    if memory is None:
        memory = model.encode(audio)

    graphed = getattr(model, "_graphed", None)
    device = audio.device
    decoder_input = torch.tensor([prefix], dtype=torch.long, device=device)

    if graphed is None:
        # fallback to hf implementation
        past = None
        current_length = len(prefix)
        while current_length < int(max_sequence_length):
            with inference_autocast(device, autocast_dtype):
                logits, past = model.decode(memory, decoder_input,
                                            use_cache=True, past_key_values=past)
        
            next_logits = logits[:, -1].float()
            allowed = state.allowed(device)
            masked = next_logits.masked_fill(~allowed, float("-inf"))
            token = int(masked.argmax(dim=-1))
            output_tokens.append(token)
            current_length += 1

            if state.update(token):
                break
            
            decoder_input = torch.tensor([[token]], dtype=torch.long,
                                         device=device)

        if output_tokens[-1] != tokenizer.eos_token:
            output_tokens.append(tokenizer.eos_token)
        return torch.tensor(output_tokens, dtype=torch.long)

    # fast path
    logits, past = model.decode(memory, decoder_input, 
                                use_cache=True,
                                past_key_values=None)

    next_logits = logits[:, -1].float()

    allowed = state.allowed(device)
    token = int(next_logits.masked_fill(~allowed, float("-inf")).argmax())
    current_length = len(prefix)
    while current_length < int(max_sequence_length):
        output_tokens.append(token)

        if state.update(token):
            break

        stop = (
            stop_time_seconds is not None
            and tokenizer.time_token_start <= token < tokenizer.time_token_end
            and tokenizer.token_to_time_id(token) / tokenizer.time_hz
            >= stop_time_seconds)
        
        if stop:
            output_tokens.append(tokenizer.eos_token)
            break

        current_length += 1
        if progress_callback is not None and current_length % 64 == 0:
            progress_callback(current_length)
        
        token = graphed.constrained_step(memory, token, state)
    
    if output_tokens[-1] != tokenizer.eos_token:
        output_tokens.append(tokenizer.eos_token)
    return torch.tensor(output_tokens, dtype=torch.long)


def sliding_window_plan(duration, window_seconds=300.0, overlap_seconds=200.0,
                        lookahead_seconds=100.0):
    if not duration > 0 or not window_seconds > 0:
        raise ValueError("Duration and window length must be positive")

    if not 0 <= lookahead_seconds <= overlap_seconds < window_seconds:
        raise ValueError("Require 0 <= lookahead <= overlap < window length")

    hop = window_seconds - overlap_seconds
    start, accepted = 0.0, 0.0
    result = []

    while True:
        last = start + window_seconds >= duration - 1e-6
        accept_end = duration if last else start + window_seconds - lookahead_seconds
        result.append(dict(start=start, end=min(duration, start + window_seconds),
                           accept_start=accepted, accept_end=accept_end,
                           prefix_end=accepted,
                           generation_stop=None if last
                           else window_seconds - lookahead_seconds))
        if last:
            return result
        accepted = accept_end
        start = min(start + hop, duration - window_seconds)


@torch.inference_mode()
def batched_constrained_generate(model, memories, prompts,
                                 max_sequence_length,
                                 autocast_dtype=torch.bfloat16,
                                 stop_time_seconds=None):
    """Pooled grammar-constrained ASR decode"""
    tokenizer = model.tokenizer
    graphed = getattr(model, "_graphed", None)

    if graphed is None:
        raise ValueError("batched decode requires the graphed decoder")

    if memories[0].dim() == 2:
        memories = [m[None] for m in memories]

    N = len(memories)
    bucket = next((b for b in graphed.batches if b >= N), graphed.batches[-1])
    B = bucket
    if B > N:
        pad = memories[0].expand(B - N, -1, -1)
        memories = list(memories) + [pad[i:i + 1].contiguous()
                                     for i in range(B - N)]
    
    device = memories[0].device
    
    prefix = tokenizer.prompt_prefix(prompts)
    if prefix[-1] == tokenizer.eos_token:
        prefix = prefix[:-1]

    decoder_input = torch.tensor([prefix] * B, dtype=torch.long, device=device)
    batch_memory = torch.cat(memories, dim=0).contiguous()

    logits, _ = graphed.decode(batch_memory, decoder_input,
                               use_cache=True, past_key_values=None)
    states = [PromptGrammarState(tokenizer) for _ in range(B)]
    out_index = prefix.index(tokenizer.out_token)
    for state in states:
        for token in prefix[out_index + 1:]:
            state.update(token)

    outputs = [list(prefix) for _ in range(B)]

    def first_token(b, state):
        allowed = state.allowed(device)
        masked = logits[b, -1].float().masked_fill(~allowed, float("-inf"))
        return int(masked.argmax(dim=-1))

    tokens_in, active = [], []
    for b in range(N):
        t0 = first_token(b, states[b])
        outputs[b].append(t0)
        if states[b].update(t0):  # t0 was eos — sequence complete
            active.append(False)
        else:
            tokens_in.append(t0)
            active.append(True)
    
    # pad slot bookkeeping to the bucket length
    tokens_in += [tokenizer.eos_token] * (B - N)
    active += [False] * (B - N)
    lengths = [len(o) for o in outputs]

    while any(active):
        nxt = graphed.constrained_step_batch(
            batch_memory, tokens_in, states, active=active)
        for b in range(N):
            if not active[b]:
                continue
            outputs[b].append(nxt[b])
            lengths[b] += 1
            slot_stop = stop_time_seconds[b] if stop_time_seconds is not None else None
            stop = (
                slot_stop is not None
                and tokenizer.time_token_start <= nxt[b] < tokenizer.time_token_end
                and tokenizer.token_to_time_id(nxt[b]) / tokenizer.time_hz
                >= slot_stop)
            done = states[b].update(nxt[b])
            if done or lengths[b] >= int(max_sequence_length) or stop:
                if not done:  # stopped early — pad the eos terminator
                    outputs[b].append(tokenizer.eos_token)
                active[b] = False
            else:
                tokens_in[b] = nxt[b]
    return outputs[:N]


# device-resident grammar tables for persistent AR kernel)
@functools.lru_cache(maxsize=8)
def build_grammar_tables(tokenizer, prompts, device="cuda"):
    import copy as _copy

    prefix = tokenizer.prompt_prefix(prompts)
    init = PromptGrammarState(tokenizer)
    out_ix = prefix.index(tokenizer.out_token)
    for t in prefix[out_ix + 1:]:
        init.update(t)

    def key(s):
        return (min(s.payload_count, 1), bool(s.in_shift),
                min(s.shift_run, 4), int(s.last_field_index),
                s.incomplete)

    states, ids = [init], {key(init): 0}
    masks, maps = [init.allowed("cpu")], []
    i = 0
    while i < len(states):
        state, i = states[i], i + 1
        mask = masks[i - 1]
        nxt = torch.full((tokenizer.n_tokens,), ids[key(state)],
                         dtype=torch.int32)
        for t in mask.nonzero(as_tuple=True)[0].tolist():
            if t == tokenizer.eos_token:
                continue  # the kernel stops on eos
            
            child = _copy.copy(state)
            child.update(t)
            k = key(child)
            if k in ids:
                # sufficiency invariant: same key -> same mask
                assert torch.equal(masks[ids[k]], child.allowed("cpu")), \
                    f"grammar state key {k} is not sufficient"
            else:
                if len(states) >= 256:
                    raise RuntimeError("grammar state space unexpectedly large")
                ids[k] = len(states)
                states.append(child)
                masks.append(child.allowed("cpu"))
            nxt[t] = ids[k]
        maps.append(nxt)
    allowed_table = torch.stack(masks).to(torch.uint8).to(device)
    state_map = torch.stack(maps).to(device)
    return allowed_table, state_map, len(states)


def decode_generated_tokens(tokenizer, tokens, song_id, window_index=None):
    try:
        return tokenizer.decode_sequence(tokens, strict=True), None
    except ValueError as exc:
        recoverable = ("empty event at subbeat", "belongs to inactive output field")
        if not any(message in str(exc) for message in recoverable):
            raise
        decoded = tokenizer.decode_sequence(tokens, strict=False)
        warning = {"song_id": song_id,
                   "warning": "strict decode failed; recovered in non-strict decode",
                   "error": str(exc)}
        if window_index is not None:
            warning["window_index"] = int(window_index)
        return decoded, warning


def event_time_map(decoded, target_seconds):
    anchors = []
    for event in decoded["events"]:
        value = event["values"].get("timestamp")
        if value is not None:
            anchors.append((int(event["subbeat"]), float(value)))
    if not anchors:
        return lambda step: min(float(target_seconds),
                                max(0.0, float(step) * 0.125))
    anchors = sorted(dict(anchors).items())
    steps = np.asarray([a[0] for a in anchors], dtype=np.float64)
    times = np.asarray([a[1] for a in anchors], dtype=np.float64)
    if len(anchors) >= 2:
        step_seconds = float(np.median(np.diff(times)
                                       / np.maximum(np.diff(steps), 1)))
        if not np.isfinite(step_seconds) or step_seconds <= 0:
            step_seconds = 0.125
    else:
        step_seconds = 0.125

    def lookup(step):
        step = float(step)
        if step <= steps[0]:
            return float(np.clip(times[0] + (step - steps[0]) * step_seconds,
                                 0, target_seconds))
        if step >= steps[-1]:
            return float(np.clip(times[-1] + (step - steps[-1]) * step_seconds,
                                 0, target_seconds))
        return float(np.interp(step, steps, times))

    return lookup


def clone_decoded_event(event):
    return {"subbeat": int(event["subbeat"]),
            "tokens_by_field": {f: [int(t) for t in toks]
                                for f, toks in event["tokens_by_field"].items()},
            "values": copy.deepcopy(event["values"])}


def refresh_event_values(event, tokenizer, prompts):
    event["values"] = {
        field: tokenizer._decode_field(field, tokens, prompts)
        for field, tokens in event["tokens_by_field"].items() if tokens}


def set_event_local_timestamp(event, tokenizer, local_time):
    time_id = int(round(float(local_time) * tokenizer.time_hz))
    time_id = max(0, min(time_id, tokenizer.n_time_tokens - 1))
    event["tokens_by_field"]["timestamp"] = [tokenizer.time_id_to_token(time_id)]


def active_context_before(events, tokenizer, time_abs):
    state = {}
    for event in events:
        event_time = event.get("time")
        if event_time is None or float(event_time) > float(time_abs) + 1e-6:
            continue
        
        for field in ("structure", "key", "chord"):
            tokens = event["tokens_by_field"].get(field)
            if tokens:
                state[field] = [int(t) for t in tokens]
        
        rhythm = event["tokens_by_field"].get("rhythm", ())
        meter = [int(t) for t in rhythm
                 if tokenizer.token_type(t) == "meter"]
        
        if meter:
            state["meter"] = meter[:1]
    return state


def apply_prefix_context(event, context, tokenizer, prompts):
    for field in ("structure", "key", "chord"):
        if field not in event["tokens_by_field"] and field in context:
            event["tokens_by_field"][field] = list(context[field])
    
    rhythm = list(event["tokens_by_field"].get("rhythm", ()))
    has_meter = any(tokenizer.token_type(t) == "meter" for t in rhythm)
    has_eighth = any(tokenizer.token_type(t) == "eighth_position"
                     for t in rhythm)
    
    if has_eighth and not has_meter and "meter" in context:
        event["tokens_by_field"]["rhythm"] = list(context["meter"]) + rhythm
    refresh_event_values(event, tokenizer, prompts)


def build_overlap_prefix_tokens(stitched_events, tokenizer, prompts,
                                window_start, prefix_end):
    """Cached overlap prefix: the events inside [window_start, prefix_end),
    re-timed to the new window, with the active key/chord/meter context."""
    eps = 1e-4
    source_events = [
        e for e in stitched_events
        if float(window_start) - eps <= float(e.get("time", -1.0))
        < float(prefix_end) - eps]
    source_events.sort(key=lambda e: (
        int(e.get("global_subbeat", e.get("source_subbeat", e["subbeat"]))),
        float(e.get("time", 0.0))))
    first = next((i for i, e in enumerate(source_events)
                  if "timestamp" in e["values"] or "rhythm" in e["values"]), None)
    if first is None:
        return None, None, None
    source_events = source_events[first:]
    base_subbeat = int(source_events[0].get(
        "global_subbeat", source_events[0].get("source_subbeat",
                                               source_events[0]["subbeat"])))
    context = active_context_before(stitched_events, tokenizer,
                                    source_events[0]["time"])
    prefix_events = []
    for source in source_events:
        event = clone_decoded_event(source)
        subbeat = int(source.get("global_subbeat",
                                 source.get("source_subbeat",
                                            source["subbeat"])))
        event["subbeat"] = max(0, subbeat - base_subbeat)
        if "timestamp" in event["tokens_by_field"]:
            set_event_local_timestamp(event, tokenizer,
                                      float(source["time"]) - window_start)
        refresh_event_values(event, tokenizer, prompts)
        prefix_events.append(event)
    apply_prefix_context(prefix_events[0], context, tokenizer, prompts)
    prefix_decoded = {"schema_version": tokenizer.schema_version,
                      "prompts": prompts, "events": prefix_events,
                      "has_eos": False}
    return (prefix_decoded,
            tokenizer.encode_decoded_sequence(prefix_decoded),
            base_subbeat)


def stitched_window_events(decoded, time_lookup, window_start, accept_start,
                           accept_end, song_duration, window_index,
                           global_subbeat_base=0):
    accepted = []
    eps = 1e-4
    for event in decoded["events"]:
        local_time = float(time_lookup(event["subbeat"]))
        abs_time = float(window_start) + local_time

        if abs_time < float(accept_start) - eps:
            continue

        if abs_time >= float(accept_end) - eps \
                or abs_time >= float(song_duration) - eps:
            continue

        output = clone_decoded_event(event)
        output["time"] = float(np.clip(abs_time, 0.0, song_duration))
        output["window_index"] = int(window_index)
        output["window_start"] = float(window_start)
        output["source_subbeat"] = int(event["subbeat"])
        output["global_subbeat"] = int(global_subbeat_base) + int(event["subbeat"])

        if "timestamp" in output["values"]:
            output["values"]["timestamp"] = output["time"]
        notes = output["values"].get("melody")
        if notes is not None:
            fixed = []
            for note in notes:
                note = copy.deepcopy(note)
                duration_steps = int(note["duration_steps"])
                local_end = float(time_lookup(int(event["subbeat"])
                                              + duration_steps))
                end_time = float(window_start) + local_end
                end_time = min(float(song_duration),
                               max(output["time"] + 0.04, end_time))
                note["end_time"] = end_time
                fixed.append(note)
            output["values"]["melody"] = fixed
        accepted.append(output)
    return accepted
