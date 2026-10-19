#!/usr/bin/env python
"""SheetSage2 model-level benchmark (mirrors bench_wenet.py's structure:
instantiate the model directly, no API layer).

Steady-state stage timing, mirroring the ASR bench convention:

    python tools/bench_sheetsage.py --model-dir /path/to/SheetSage2 \
        --base-model-dir /path/to/MERT-v2-FullSong \
        --audio song.mp3 [--iters 3] [--decoupled]

Stages: encode (paged, stream-parallel windows) -> AR decode (grammar-
constrained, CUDA-graphed) -> exports (serving set). Rendering (五线谱/钢琴)
is deferred unless --render is passed. ``--decoupled`` decodes all windows
of a song in ONE pooled batch (no overlap prefix) — the best-case
throughput number the request pool approaches under concurrency.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True,
                    help="SheetSage2 checkpoint dir")
    ap.add_argument("--base-model-dir", required=True,
                    help="MERT-v2-FullSong encoder dir")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--audio-dir", default=None)
    ap.add_argument("--output", default=None,
                    help="output dir for ABC/MIDI (default: bench_output/)")
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", action="store_true", default=True,
                    help="accepted for shell-wrapper compatibility; "
                         "warmup always runs (steady-state timing)")
    ap.add_argument("--engine", default="pipeline", choices=("pipeline", "graph"),
                    help="pipeline = multi-kernel CUDA graph (default); graph = piecewise torch graph")
    ap.add_argument("--decoupled", action="store_true",
                    help="decode all windows in one pooled batch "
                         "(no overlap prefix; best-case throughput)")
    ap.add_argument("--render", action="store_true",
                    help="also render 五线谱 SVG/PNG + piano WAV (lazy in serving)")
    args = ap.parse_args()

    from veloxvoice.models.sheetsage.v2.sheetsage import Sheetsage2Model
    from veloxvoice.models.sheetsage.v2.config import SheetSage2Config
    from veloxvoice.models.sheetsage.v2.audio_loader import load_audio, SAMPLE_RATE
    from veloxvoice.stream.grammar_ar_decode import (
        constrained_prompt_generate, batched_constrained_generate,
        decode_generated_tokens, event_time_map, sliding_window_plan,
        stitched_window_events, FULL_TASK_PROMPTS)
    from veloxvoice.audio.music_exports import export_result
    from veloxvoice.audio.music_tokenizer import SheetSage2Tokenizer

    cfg = SheetSage2Config.from_json(Path(args.model_dir) / "config.json")
    t0 = time.perf_counter()
    model = Sheetsage2Model(cfg)
    model.load_checkpoint(
        str(Path(args.model_dir) / "model.safetensors"),
        str(Path(args.base_model_dir) / "model.safetensors"),
        args.device)
    model.eval()
    model.tokenizer = SheetSage2Tokenizer(
        cfg.input_audio_length, cfg.time_hz,
        cfg.tokenizer_schema_version,
        expected_fingerprint=cfg.tokenizer_fingerprint)
    model = model.to(args.device)
    if args.device.startswith("cuda"):
        from veloxvoice.models.sheetsage.v2.graph_decoder import (
            GraphedSheetSage2Decoder)
        _graphed = GraphedSheetSage2Decoder(model)
        model._graphed = _graphed

        def _decode(memory, decoder_input_ids, use_cache=True,
                    past_key_values=None):
            return _graphed.decode(memory, decoder_input_ids, use_cache,
                                   past_key_values)

        model.decode = _decode
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")

    out_root = Path(args.output or ROOT / "bench_output")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_root}")
    songs = []
    if args.audio:
        songs.append(Path(args.audio))
    if args.audio_dir:
        exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        songs.extend(sorted(f for f in Path(args.audio_dir).iterdir()
                            if f.suffix.lower() in exts))
    if not songs:
        print("no audio; use --audio or --audio-dir")
        return

    prompts = model.tokenizer.normalize_prompts(FULL_TASK_PROMPTS)

    for song_path in songs:
        audio = load_audio(str(song_path), preset="default")
        if isinstance(audio, tuple):
            audio = audio[0]
        audio = audio.to(args.device)
        duration = len(audio) / SAMPLE_RATE
        window_length = float(cfg.input_audio_length)
        plan = sliding_window_plan(duration, window_length)
        stride = cfg.mert.inputs_to_logits_ratio
        window_samples = int(window_length * SAMPLE_RATE)

        def encode_windows():
            segments = []
            for window in plan:
                ss = int(window["start"] * SAMPLE_RATE)
                segment = audio[ss:ss + window_samples]
                pad = window_samples - segment.numel()
                if pad > 0:
                    segment = torch.nn.functional.pad(segment, (0, pad))
                if window_samples % stride:
                    segment = torch.nn.functional.pad(
                        segment, (0, stride - window_samples % stride))
                segments.append(segment[None])
            if len(segments) > 1 and str(args.device).startswith("cuda"):
                streams = [torch.cuda.Stream(device=args.device)
                           for _ in segments]
                mems = []
                for segment, stream in zip(segments, streams):
                    with torch.cuda.stream(stream):
                        mems.append(model.encode(segment.to(args.device)))
                for stream in streams:
                    stream.synchronize()
                return segments, mems
            return segments, [model.encode(s.to(args.device)) for s in segments]

        out_dir = out_root
        enc_t, dec_t = [], []
        all_decoded = {"schema_version": model.tokenizer.schema_version,
                       "prompts": prompts, "events": [], "has_eos": True}
        global_subbeat_base = 0

        for it in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            segments, mem_list = encode_windows()
            torch.cuda.synchronize()
            enc = time.perf_counter() - t0
            enc_t.append(enc)

            t0 = time.perf_counter()
            stitched = []
            if args.decoupled and len(plan) > 1:
                # best case: all windows in one pooled decode (no overlap
                # prefix — window boundaries lose their continuity hint)
                token_lists = batched_constrained_generate(
                    model, mem_list, prompts, cfg.max_output_seq_len,
                    stop_time_seconds=[w["generation_stop"] for w in plan])
                for wi, (window, tokens) in enumerate(zip(plan, token_lists)):
                    tokens = torch.as_tensor(tokens, dtype=torch.long)
                    decoded, _ = decode_generated_tokens(
                        model.tokenizer, tokens, "window", window_index=wi)
                    tl = event_time_map(decoded, window["end"] - window["start"])
                    accepted = stitched_window_events(
                        decoded, tl, window["start"], window["accept_start"],
                        window["accept_end"], duration, wi,
                        global_subbeat_base=global_subbeat_base)
                    stitched.extend(accepted)
                    global_subbeat_base += max(
                        (int(e["subbeat"]) for e in decoded["events"]),
                        default=0) + 1
            else:
                for wi, window in enumerate(plan):
                    if args.engine == "pipeline":
                        from veloxvoice.models.sheetsage.v2.step_pipeline import (
                            transcribe_window_multi)
                        tokens = transcribe_window_multi(
                            model, segments[wi].to(args.device), prompts,
                            cfg.max_output_seq_len,
                            stop_time_seconds=window["generation_stop"],
                            memory=mem_list[wi])
                    else:
                        tokens = constrained_prompt_generate(
                            model, segments[wi].to(args.device), prompts,
                            cfg.max_output_seq_len,
                            prefix_tokens=None,
                            autocast_dtype=torch.bfloat16
                            if args.dtype == "bf16" else None,
                            stop_time_seconds=window["generation_stop"],
                            memory=mem_list[wi])
                    decoded, _ = decode_generated_tokens(
                        model.tokenizer, tokens, "window", window_index=wi)
                    tl = event_time_map(decoded, window["end"] - window["start"])
                    accepted = stitched_window_events(
                        decoded, tl, window["start"], window["accept_start"],
                        window["accept_end"], duration, wi,
                        global_subbeat_base=global_subbeat_base)
                    stitched.extend(accepted)
                    global_subbeat_base += max(
                        (int(e["subbeat"]) for e in decoded["events"]),
                        default=0) + 1
            torch.cuda.synchronize()
            dec_t.append(time.perf_counter() - t0)
            all_decoded["events"] = stitched

        # steady state = median over iters (iter 0 covers graph capture etc.)
        enc_s = float(np.median(enc_t[1:])) if len(enc_t) > 1 else enc_t[0]
        dec_s = float(np.median(dec_t[1:])) if len(dec_t) > 1 else dec_t[0]
        t0 = time.perf_counter()
        result = export_result(all_decoded, model.tokenizer,
                               output_dir=str(out_dir) if out_dir else None,
                               duration=duration, playback=False, debug=False)
        exp_s = time.perf_counter() - t0
        e2e = enc_s + dec_s + exp_s
        enc_rtf = enc_s / duration
        dec_rtf = dec_s / duration
        rtf = e2e / duration
        abc_err = result.get("abc_error")
        mode = "decoupled-pool" if args.decoupled else "sequential"
        print(f"\n{song_path.name} [{mode}] {duration:.1f}s audio, "
              f"{len(plan)} window(s)")
        print(f"  encode:   {enc_s*1000:8.1f} ms  enc-RTF {enc_rtf:.5f}")
        print(f"  decode:   {dec_s*1000:8.1f} ms  dec-RTF {dec_rtf:.5f}")
        print(f"  exports:  {exp_s*1000:8.1f} ms")
        print(f"  e2e:      {e2e*1000:8.1f} ms  RTF {rtf:.5f} "
              f"({len(stitched)} events"
              f"{', ABC error: ' + abc_err if abc_err else ''})")
        print(f"  output:   {out_dir}")
        print(f"  ABC:      {out_dir / 'score.abc'}")

        if args.render and out_dir and out_dir.exists():
            from veloxvoice.audio.music_rendering import render_outputs
            from veloxvoice.audio.music_midi import export_playback

            export_playback(out_dir, None, duration)
            render_assets = Path(__file__).resolve().parents[1] / "render_assets"
            if not render_assets.exists():
                render_assets = Path("/raid/yiakwy/SheetSage2/render_assets")
            if render_assets.exists():
                try:
                    render_outputs(
                        input_dir=out_dir, output_dir=out_dir,
                        score="svg,png", audio=True,
                        assets_dir=str(render_assets))
                except Exception as exc:
                    print(f"  [render error] {exc}")


if __name__ == "__main__":
    main()
