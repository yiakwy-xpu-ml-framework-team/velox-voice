#!/usr/bin/env python
"""Side-by-side on-chip timeline: OLD single-block persistent kernel vs the
NEW multi-kernel CUDA-graph pipeline (K1/K2/K3), aligned at t=0.

Produces ``sheetsage_decode_step_v2_*`` under this directory:
    *_metadata.json        old/new per-stage (or per-K-group) times
    *_phase_trace.npy      both timelines (structured arrays)
    *_perfetto.json/.html  the NEW kernel trace as a Chrome trace
    *_phase_timeline.png   the comparison chart (OLD in blue, NEW in purple)

Run: CUDA_VISIBLE_DEVICES=5 /root/venv-music/bin/python \
    benchmark/visualize_gemm_pipelining/hopper/sheetsage_decode_step_v2/bench_sheetsage_decode_step_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
OLD_DIR = HERE.parent / "sheetsage_decode_step"
OUT = HERE
PKG = "sheetsage_decode_step_v2"
SONG = "test_data/music/bohemian_rhapsody_queen.mp3"

# OLD layout (from sheetsage_decode_step): slot -> (stage, kind); slot 10 = step start
OLD_PHASES = [                      # (slot, name, kind)
    (0, "embed", "io"),
    (1, "qkv_gemv+kvstore", "compute"),
    (2, "self_attn", "compute"),
    (3, "self_out+ln+xq", "compute"),
    (6, "cross_attn", "compute"),
    (5, "cross_out+ln+ffn1", "compute"),
    (4, "gelu+ffn2+ln", "compute"),
    (8, "lm_head", "compute"),
    (9, "argmax+fsm", "alu"),
]
OLD_START = 10
OLD_ORDER = [OLD_START] + [s for s, _, _ in OLD_PHASES]

NEW_KIND = {                        # kernel-name substring -> K-group label
    "flashinfer::SingleDecodeWithKVCacheKernel": "K1 (cross_attn)",
    "MergeStatesLargeNumIndexSetsKernel":        "K1 (cross_attn)",
    "ss_self_attn_split_kernel":                 "K3 (self-attn)",
    "ss_attn_merge_kernel":                      "K3 (self-attn)",
    "ss_kvstore_kernel":                         "K3 (self-attn)",
    "ss_gemv_ln_kernel":                         "K2 (fused gemv)",
    "ss_gemv_kernel":                            "K2 (fused gemv)",
    "ss_gemv_row_kernel":                        "K2 (fused gemv)",
    "Gelu":                                      "K2 (fused gemv)",
    "gelu":                                      "K2 (fused gemv)",
}
NEW_GROUPS = ["K1 (cross_attn)", "K2 (fused gemv)", "K3 (self-attn)", "other"]


def new_kind(name):
    for k, v in NEW_KIND.items():
        if k in name:
            return v
    return "other"


def old_timeline():
    ts = np.load(OLD_DIR / "sheetsage_decode_step_phase_trace.npy")  # [S,6,12]
    rows = [r for r in range(ts.shape[0]) if ts[r].max() > 0]
    r = rows[0]
    t = ts[r]
    # program-order chain: start slot then each phase end slot
    seq = [t[0][OLD_START]] + [t[l][slot] for l in range(6)
                               for slot, _, _ in OLD_PHASES if slot in (1, 2, 3, 6, 5, 4)]
    # rebuild spans exactly as the v1 script did
    spans = []
    prev = int(t[0][OLD_START])
    INLAYER = [(1, "qkv_gemv+kvstore"), (2, "self_attn"), (3, "self_out+ln+xq"),
               (6, "cross_attn"), (5, "cross_out+ln+ffn1"), (4, "gelu+ffn2+ln")]
    spans.append(("embed", "io", 0, prev, int(t[0][0])))
    prev = int(t[0][0])
    for l in range(6):
        for slot, name in INLAYER:
            b = int(t[l][slot])
            spans.append((name, "compute", l, prev, b))
            prev = b
    spans.append(("lm_head", "compute", 0, prev, int(t[0][8])))
    prev = int(t[0][8])
    spans.append(("argmax+fsm", "alu", 0, prev, int(t[0][9])))
    return spans


def new_timeline():
    """Profile ONE graph replay of the multi-kernel pipeline -> chrome trace."""
    from torch.profiler import profile, ProfilerActivity

    from veloxvoice.models.sheetsage.v2 import api as sheetsage_api
    from veloxvoice.models.sheetsage.v2.audio_loader import (
        load_audio, SAMPLE_RATE)
    from veloxvoice.models.sheetsage.v2.step_pipeline import (
        MultiKernelStepDecoder)
    from veloxvoice.stream.grammar_ar_decode import (
        FULL_TASK_PROMPTS, build_grammar_tables)

    m = sheetsage_api.load_model("/raid/yiakwy/SheetSage2",
                                 "/raid/yiakwy/MERT-v2-FullSong", "bf16", "cuda")
    wav = load_audio(str(ROOT / SONG), sampling_rate=SAMPLE_RATE)
    wav = wav[0] if isinstance(wav, tuple) else wav
    seg = wav[:int(300 * SAMPLE_RATE)].reshape(1, -1).contiguous().cuda()
    prompts = m.tokenizer.normalize_prompts(FULL_TASK_PROMPTS)
    prefix = m.tokenizer.prompt_prefix(prompts)
    mem = m.encode(seg).contiguous()
    g = m._graphed
    with torch.inference_mode():
        imp = g._ensure(mem)
        imp.prefill_full_sequence(
            torch.tensor([prefix], dtype=torch.long, device="cuda"))
        imp.refresh_cross(mem)
    at, sm, _ = build_grammar_tables(m.tokenizer, prompts, "cuda")

    dec = m.decoder
    parts = []
    for layer in dec.layers:
        qw = torch.cat([layer.self_attn.q_proj.weight,
                        layer.self_attn.k_proj.weight,
                        layer.self_attn.v_proj.weight], 0)
        qb = torch.cat([layer.self_attn.q_proj.bias,
                        layer.self_attn.k_proj.bias,
                        layer.self_attn.v_proj.bias], 0)
        parts += [qw.reshape(-1), qb.reshape(-1),
                  layer.self_attn.out_proj.weight.reshape(-1),
                  layer.self_attn.out_proj.bias.reshape(-1),
                  layer.encoder_attn.q_proj.weight.reshape(-1),
                  layer.encoder_attn.q_proj.bias.reshape(-1),
                  layer.fc1.weight.reshape(-1), layer.fc1.bias.reshape(-1),
                  layer.fc2.weight.reshape(-1), layer.fc2.bias.reshape(-1),
                  layer.self_attn_layer_norm.weight.reshape(-1),
                  layer.self_attn_layer_norm.bias.reshape(-1),
                  layer.encoder_attn_layer_norm.weight.reshape(-1),
                  layer.encoder_attn_layer_norm.bias.reshape(-1),
                  layer.final_layer_norm.weight.reshape(-1),
                  layer.final_layer_norm.bias.reshape(-1)]
    parts += [dec.layernorm_embedding.weight.reshape(-1),
              dec.layernorm_embedding.bias.reshape(-1)]
    deco = MultiKernelStepDecoder(
        m, imp.k_cross.view(6, 8, imp.T, 64), imp.v_cross.view(6, 8, imp.T, 64),
        imp.k_self.view(6, 8, imp.S, 64), imp.v_self.view(6, 8, imp.S, 64),
        torch.cat(parts).contiguous(), at, sm, m.config.vocab_size,
        m.tokenizer.eos_token)
    pos = 8
    deco.pos_dev.fill_(pos)
    deco.x_in.fill_(prefix[-1])
    deco.allowed_cur.copy_(deco.allowed_table[0])
    deco.capture()
    for _ in range(10):
        deco._graph.replay()
    torch.cuda.synchronize()
    # production piecewise-graph step (torch ops) for the speedup headline
    import time as _t
    from veloxvoice.stream.grammar_ar_decode import PromptGrammarState
    _gs = PromptGrammarState(m.tokenizer)
    _oi = prefix.index(m.tokenizer.out_token)
    for _tk in prefix[_oi + 1:]:
        _gs.update(_tk)
    g.decode(mem, torch.tensor([prefix], dtype=torch.long, device="cuda"), True, None)
    _cur = prefix[-1]
    for _ in range(5):
        _cur = g.constrained_step(mem, _cur, _gs)
    torch.cuda.synchronize()
    _t0 = _t.perf_counter()
    for _ in range(300):
        _cur = g.constrained_step(mem, _cur, _gs)
    torch.cuda.synchronize()
    prod_us = (_t.perf_counter() - _t0) / 300 * 1e6
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        deco._graph.replay()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(OUT / f"{PKG}_perfetto.json"))
    ev = json.loads((OUT / f"{PKG}_perfetto.json").read_text())["traceEvents"]
    kern = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
    t0 = min(e["ts"] for e in kern)
    out = [(new_kind(e["name"]), "cuda", e["ts"] - t0,
            e["ts"] - t0 + e["dur"]) for e in kern]
    # keep only this step's kernels (drop profiler tail/dup)
    return out, deco, prod_us


def main():
    old = old_timeline()
    new, deco, prod_us = new_timeline()

    old_base = old[0][3]
    old = [(nm, kd, l, a - old_base, b - old_base) for nm, kd, l, a, b in old]
    old_total = max(b for *_, b in old) / 1e3
    new_total = max(b for *_, b in new)
    print(f"OLD (single-block persistent) step: {old_total:8.1f} us")
    print(f"NEW (multi-kernel graph)      step: {new_total:8.1f} us")
    print(f"  speedup vs OLD single-block : {old_total / new_total:.1f}x")
    print(f"  production piecewise graph  : {prod_us:8.1f} us"
          f"   -> {prod_us / new_total:.2f}x")
    print()
    print("NEW breakdown by K-group:")
    grp = {g: 0.0 for g in NEW_GROUPS}
    for k, _kind, a, b in new:
        grp[k] += (b - a)
    for g in NEW_GROUPS:
        print(f"  {g:18s} {grp[g]:7.1f} us  {100 * grp[g] / new_total:5.1f}%")
    print()
    print("OLD breakdown by stage (6 layers each, summed):")
    old_grp = {}
    for name, kind, l, a, b in old:
        old_grp[name] = old_grp.get(name, 0.0) + (b - a) / 1e3
    for name, v in sorted(old_grp.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {v:9.1f} us  {100 * v / old_total:5.1f}%")

    meta = dict(
        production_graph_step_us=round(prod_us, 3),
        speedup_vs_production=round(prod_us / new_total, 3),
        old_kernel="sheetsage_ar_persistent (single-block, legacy)",
        new_kernel="multi-kernel CUDA graph (step_pipeline.py: K1 flashinfer, K2/K3 ss_* kernels)",
        old_step_us=round(old_total, 3), new_step_us=round(new_total, 3),
        speedup=round(old_total / new_total, 3),
        new_group_us={g: round(grp[g], 3) for g in NEW_GROUPS},
        old_stage_us={k: round(v, 3) for k, v in old_grp.items()},
    )
    (OUT / f"{PKG}_metadata.json").write_text(json.dumps(meta, indent=2))
    np.save(OUT / f"{PKG}_phase_trace.npy", np.array(
        [(*o[:4],) for o in old] + [(*n[:4],) for n in new], dtype=object),
        allow_pickle=True)

    # ---- chart: OLD (blue) top tracks, NEW (purple) bottom tracks, t=0 aligned
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    OLD_COL = {"io": "#4f8ef7", "compute": "#2f6fd0", "alu": "#f2c14e"}
    NEW_COL = {"K1 (cross_attn)": "#8b3fd6", "K2 (fused gemv)": "#b06ce0",
               "K3 (self-attn)": "#d9a6f2", "other": "#9aa0a6"}

    old_names = []
    for name, kind, l, a, b in old:
        if name not in old_names:
            old_names.append(name)
    yt, yl = [], []
    lane = 0
    fig, ax = plt.subplots(2, 1, figsize=(17, 11),
                           gridspec_kw={"height_ratios": [2.8, 1]})
    for name in old_names:
        for nm, kind, l, a, b in old:
            if nm != name:
                continue
            ax[0].barh(lane, max((b - a) / 1e3, 1e-3), left=a / 1e3,
                       height=0.8, color=OLD_COL[kind], edgecolor="white",
                       linewidth=0.2)
        yt.append(lane); yl.append(name); lane += 1
    lane += 1
    new_start = lane
    for g in NEW_GROUPS:
        for k, _kind, a, b in new:
            if k != g:
                continue
            ax[0].barh(lane, max(b - a, 1e-3), left=a, height=0.8,
                       color=NEW_COL[g], edgecolor="white", linewidth=0.2)
        yt.append(lane); yl.append(g); lane += 1
    ax[0].set_yticks(yt); ax[0].set_yticklabels(yl, fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("time since step start (us)")
    ax[0].set_title(
        f"SheetSage2 decode step — OLD single-block persistent "
        f"({old_total/1e3:.2f} ms) vs NEW multi-kernel graph "
        f"({new_total:.0f} us):  {old_total/new_total:.1f}x vs OLD,  "
        f"{prod_us/new_total:.2f}x vs production graph")
    ax[0].grid(axis="x", alpha=0.3)
    ax[0].legend(handles=[Patch(facecolor=c, label=l) for l, c in
                          [("old: io", "#4f8ef7"), ("old: compute", "#2f6fd0"),
                           ("new: K1 (cross_attn)", "#8b3fd6"),
                           ("new: K2 (fused gemv)", "#b06ce0"),
                           ("new: K3 (self-attn)", "#d9a6f2"),
                           ("new: other", "#9aa0a6")]],
                 loc="lower right", fontsize=8)

    # zoomed panel: NEW only, 0..new_total
    lane = 0
    yt2, yl2 = [], []
    for g in NEW_GROUPS:
        for k, _kind, a, b in new:
            if k != g:
                continue
            ax[1].barh(lane, max(b - a, 1e-3), left=a, height=0.8,
                       color=NEW_COL[g])
        yt2.append(lane); yl2.append(g); lane += 1
    ax[1].set_yticks(yt2); ax[1].set_yticklabels(yl2, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("time since step start (us) — NEW zoomed")
    ax[1].set_title("NEW multi-kernel graph zoom: K1 (cross_attn) / "
                    "K2 (fused gemv) / K3 (self-attn)")
    ax[1].grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f"{PKG}_phase_timeline.png", dpi=130)

    # perfetto html for the NEW trace
    ev_all = json.loads((OUT / f"{PKG}_perfetto.json").read_text())
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SheetSage2 decode step v2 — perfetto</title>
<style>body{{font:14px system-ui;margin:24px}}table{{border-collapse:collapse}}
th,td{{border-bottom:1px solid #ddd;padding:6px 12px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}iframe{{width:100%;height:70vh;border:1px solid #ccc}}</style>
</head><body><h2>NEW multi-kernel graph — kernel trace</h2>
<p>old {old_total/1e3:.2f} ms → new {new_total:.0f} us  ({old_total/new_total:.1f}x)</p>
<iframe id="pv" src="https://ui.perfetto.dev"></iframe>
<script>
const trace = {json.dumps(ev_all)};
const f=document.getElementById('pv');
window.addEventListener('message',e=>{{if(e.data==='PONG')f.contentWindow.postMessage(
  {{perfetto:{{buffer:new TextEncoder().encode(JSON.stringify(trace))}}}},' https://ui.perfetto.dev'.trim());}});
f.onload=()=>setInterval(()=>f.contentWindow.postMessage('PING','https://ui.perfetto.dev'),250);
</script></body></html>"""
    (OUT / f"{PKG}_perfetto.html").write_text(html)
    print(f"\n[viz] wrote {PKG}_* to {OUT}")


if __name__ == "__main__":
    main()
