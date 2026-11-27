"""SheetSage2 v2 entry points (music audio -> lead sheet).

Model loading + whole-song transcription orchestration.
``Velox.load`` dispatches here from ``veloxvoice/api.py``; both paths
are caller-supplied (no hardcoded model dirs).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from veloxvoice.audio.music_exports import export_result
from veloxvoice.audio.music_tokenizer import SheetSage2Tokenizer

from .config import SheetSage2Config
from ....stream.grammar_ar_decode import (decode_generated_tokens, event_time_map,
                       sliding_window_plan, stitched_window_events)

_MODEL_CACHE = {}


def load_model(model_dir: str | os.PathLike,
               base_model_dir: str | os.PathLike,
               dtype: str = "bf16",
               device: str = "cuda",
               graphed_decode: bool = True):
    """Load (and LoRA-merge) SheetSage2; cached per (dirs, dtype, device)."""
    import torch

    from .graph_decoder import GraphedSheetSage2Decoder
    from .sheetsage import Sheetsage2Model

    key = (str(model_dir), str(base_model_dir), dtype, device, graphed_decode)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    cfg = SheetSage2Config.from_json(Path(model_dir) / "config.json")
    model = Sheetsage2Model(cfg)
    model.load_checkpoint(str(Path(model_dir) / "model.safetensors"),
                          str(Path(base_model_dir) / "model.safetensors"),
                          device)
    model.eval()
    model.tokenizer = SheetSage2Tokenizer(
        cfg.input_audio_length, cfg.time_hz,
        cfg.tokenizer_schema_version,
        expected_fingerprint=cfg.tokenizer_fingerprint)
    if graphed_decode and str(device).startswith("cuda"):
        _graphed = GraphedSheetSage2Decoder(model)
        model._graphed = _graphed

        def _decode(memory, decoder_input_ids, use_cache=True,
                    past_key_values=None):
            return _graphed.decode(memory, decoder_input_ids, use_cache,
                                   past_key_values)

        model.decode = _decode

        from .pool import WindowDecodePool

        model._decode_pool = WindowDecodePool(model)
    _MODEL_CACHE[key] = model
    return model


def transcribe(model, audio, output_dir=None, *, dtype="bf16",
               overlap_seconds=None, lookahead_seconds=None, serving=False,
               **kwargs):
    """Whole-song transcription: window plan -> per-window AR decode ->
    event stitching -> exports (in memory; files only when ``output_dir``)."""
    import torch

    from .audio_loader import load_audio, SAMPLE_RATE

    device = next(model.parameters()).device
    dtype_map = {"bf16": torch.bfloat16, "fp32": None, "fp16": torch.float16}
    autocast_dtype = dtype_map.get(dtype)
    started = time.monotonic()

    audio = load_audio(audio, preset="default")
    duration = len(audio) / SAMPLE_RATE
    window_length = float(model.cfg.input_audio_length)
    overlap_seconds = 200.0 if overlap_seconds is None else overlap_seconds
    lookahead_seconds = 100.0 if lookahead_seconds is None else lookahead_seconds
    plan = sliding_window_plan(duration, window_length, overlap_seconds,
                               lookahead_seconds)
    output_dir = Path(output_dir) if output_dir is not None else None

    from ....stream.grammar_ar_decode import FULL_TASK_PROMPTS

    prompts = model.tokenizer.normalize_prompts(FULL_TASK_PROMPTS)
    stitched, records, warnings = [], [], []
    global_subbeat_base = 0

    import torch as _torch

    window_samples = int(window_length * SAMPLE_RATE)
    stride = model.cfg.mert.inputs_to_logits_ratio
    segments = []
    for window in plan:
        start_sample = int(window["start"] * SAMPLE_RATE)
        segment = audio[start_sample:start_sample + window_samples]
        pad = window_samples - segment.numel()
        if pad > 0:
            segment = _torch.nn.functional.pad(segment, (0, pad))
        if window_samples % stride:
            segment = _torch.nn.functional.pad(
                segment, (0, stride - window_samples % stride))
        segments.append(segment[None])

    mem_list = []
    if len(segments) > 1 and str(device).startswith("cuda"):
        streams = [_torch.cuda.Stream(device=device) for _ in segments]
        done = []
        for segment, stream in zip(segments, streams):
            with _torch.cuda.stream(stream):
                mem_list.append(model.encode(segment.to(device)))
            done.append(_torch.cuda.Event())
            done[-1].record(stream)
        for event in done:
            event.synchronize()
    else:
        mem_list = [model.encode(segment.to(device)) for segment in segments]

    for index, (window, segment, memory) in enumerate(zip(plan, segments, mem_list)):
        prefix, base = None, 0
        if index:
            _, prefix, base = _build_prefix(stitched, model.tokenizer,
                                            prompts, window["start"],
                                            window["prefix_end"])
            if prefix is not None and len(prefix) >= \
                    model.cfg.max_output_seq_len - 128:
                raise ValueError("Overlap prefix fills the context; "
                                 "reduce overlap_seconds")

        pool = getattr(model, "_decode_pool", None)
        if pool is not None:
            tokens = pool.transcribe_window(
                segment.to(device), prompts, model.cfg.max_output_seq_len,
                prefix_tokens=prefix, autocast_dtype=autocast_dtype,
                stop_time_seconds=window["generation_stop"], memory=memory)
        else:
            tokens = _generate_window(model, segment.to(device), prompts,
                                      model.cfg.max_output_seq_len,
                                      prefix_tokens=prefix,
                                      autocast_dtype=autocast_dtype,
                                      stop_time_seconds=window["generation_stop"],
                                      memory=memory)
        decoded, warning = decode_generated_tokens(
            model.tokenizer, tokens, "window", window_index=index)
        if warning:
            warnings.append(warning)

        time_lookup = event_time_map(decoded, window["end"] - window["start"])
        accepted = stitched_window_events(
            decoded, time_lookup, window["start"], window["accept_start"],
            window["accept_end"], duration, index,
            global_subbeat_base=global_subbeat_base)
        stitched.extend(accepted)
        records.append(dict(window, tokens=[int(t) for t in tokens.tolist()],
                            prefix_tokens=len(prefix) if prefix else 0,
                            events=len(accepted)))
        global_subbeat_base += max((int(e["subbeat"]) for e in decoded["events"]),
                                   default=0) + 1

    decoded_all = {"schema_version": model.tokenizer.schema_version,
                   "prompts": prompts, "events": stitched, "has_eos": True}
    result = export_result(decoded_all, model.tokenizer,
                           output_dir=output_dir, duration=duration,
                           playback=not serving, debug=not serving)
    result = dict(result, tokens=[r["tokens"] for r in records],
                  windows=records, warnings=warnings,
                  num_events=len(stitched),
                  duration_seconds=duration,
                  elapsed_seconds=time.monotonic() - started)
    if output_dir is not None:
        import json

        slim = dict(result)
        payload = {k: v for k, v in result.get("payload", {}).items()
                   if k not in ("midis", "playback", "labs", "midi")}
        if "payload" in slim:
            slim["payload"] = payload
        (output_dir / "result.json").write_text(
            json.dumps(slim, ensure_ascii=False, default=str, indent=2))

    return result


def _build_prefix(stitched_events, tokenizer, prompts, window_start,
                  prefix_end):
    from ....stream.grammar_ar_decode import build_overlap_prefix_tokens

    return build_overlap_prefix_tokens(stitched_events, tokenizer, prompts,
                                       window_start, prefix_end)


def _decode_engine() -> str:
    """Which window-decode engine to use.

    Default: the fastest path (``step_pipeline.transcribe_window_multi``): 
    
    - K1 flashinfer cross-attn +
    - K2/K3 fused kernels.  
    
    ``VELOX_PIECEWISE_CUDA_GRAPH_ENABLE=1`` forces the old piecewise torch CUDA graph.
    """
    import os

    if os.environ.get("VELOX_PIECEWISE_CUDA_GRAPH_ENABLE") == "1":
        return "graph"
    return "pipeline"


def _generate_window(model, segment, prompts, max_seq_len, *,
                     prefix_tokens, autocast_dtype, stop_time_seconds,
                     memory=None):
    import os

    from ....stream.grammar_ar_decode import constrained_prompt_generate

    engine = _decode_engine()
    if engine == "pipeline" and getattr(model, "_graphed", None) is not None:
        # multi-kernel CUDA-graph pipeline (fastest); falls back below if the
        # graph lane is unavailable (no graphed decoder / no CUDA)
        try:
            from .step_pipeline import transcribe_window_multi

            import torch as _t

            if _t.cuda.is_available():
                return transcribe_window_multi(
                    model, segment, prompts, max_seq_len,
                    memory=memory, stop_time_seconds=stop_time_seconds,
                    prefix_tokens=prefix_tokens)
        except Exception:
            pass
    return constrained_prompt_generate(
        model, segment, prompts, max_seq_len,
        prefix_tokens=prefix_tokens, autocast_dtype=autocast_dtype,
        stop_time_seconds=stop_time_seconds, memory=memory)
