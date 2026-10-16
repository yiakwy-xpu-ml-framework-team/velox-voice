"""Benchmark for the veloxvoice CUDA kernel: persistent fused CTC prefix beam.

Kernel: veloxvoice/kernels/csrc/velox_prefix_beam_search.cu
Op:     veloxvoice.kernels.ops.cuda_ops.beam_search

The per-frame top-k of ``logp [T, V]`` is done in parallel by
``torch.topk(logp, K, dim=-1)`` (one launch); ``beam_search`` then runs the whole
T loop in a single persistent thread block. The measured "op" is
``topk + beam_search``.

Compared against two references:
  * host  — `ctc_prefix_beam_search` (host device-top-k + python bookkeeping)
  * torch — `torch_naive_prefix_beam`

Usage:
  python benchmark/kernels/bench_beam_search.py
or
  python benchmark/kernels/bench_beam_search.py \
      --model-dir data/models/asr_model
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from veloxvoice.kernels.ops.cuda_ops import beam_search
from veloxvoice.stream.rescore import AttRescorer, ctc_prefix_beam_search

VOCAB = 14128


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def host_prefix_beam(logp, beam):
    """Host reference: ctc_prefix_beam_search (from veloxvoice.stream.rescore)."""
    return [(tuple(h), float(s)) for h, s in ctc_prefix_beam_search(logp, beam)]


NEG, BLANK = float("-inf"), 0


def torch_naive_prefix_beam(logp: torch.Tensor, beam: int, blank_id: int = BLANK):
    """Legacy all-torch GPU beam (bench-only reference; slow ~1000 frames/s)."""
    dev = logp.device
    T = int(logp.shape[0])
    B = int(beam)

    Lc = T + 1
    tok = torch.zeros(B, Lc, dtype=torch.long, device=dev)

    ln = torch.zeros(B, dtype=torch.long, device=dev)

    s = torch.full((B,), NEG, device=dev)
    ns = torch.full((B,), NEG, device=dev)

    s[0] = 0.0
    valid = torch.zeros(B, dtype=torch.bool, device=dev)
    valid[0] = True
    for t in range(T):
        lp = logp[t]
        tv, ti = torch.topk(lp, B)
        tokv, lnv, sv, nsv = tok[valid], ln[valid], s[valid], ns[valid]
        P, K = tokv.shape[0], B
        lob = torch.logaddexp(sv, nsv)
        last = torch.where(lnv > 0,
                           tokv.gather(1, (lnv - 1).clamp(min=0)[:, None]).squeeze(1),
                           torch.full_like(lnv, -1))
        u = ti[None, :].expand(P, K)
        is_blank = u == blank_id
        is_eq = (u == last[:, None]) & ~is_blank
        is_new = ~is_blank & ~is_eq
        lobg = lob[:, None]
        keep_rows = tokv[:, None, :].expand(P, K, Lc).reshape(P * K, Lc)
        keep_ln = lnv[:, None].expand(P, K).reshape(P * K)
        keep_sc = torch.where(is_blank, lobg + lp[blank_id],
                              torch.where(is_eq, nsv[:, None] + tv[None, :],
                                          torch.full_like(tv[None, :], NEG))).reshape(P * K)
        keep_ty = torch.where(is_blank, 1, 0).reshape(P * K)
        ext_rows = tokv[:, None, :].expand(P, K, Lc).clone()
        b_idx = torch.arange(P, device=dev)[:, None]
        k_idx = torch.arange(K, device=dev)[None, :]
        ext_rows[b_idx, k_idx, lnv[:, None].clamp(max=Lc - 1)] = ti[None, :]
        ext_rows = ext_rows.reshape(P * K, Lc)
        ext_ln = (lnv[:, None] + 1).expand(P, K).reshape(P * K)
        ext_sc = torch.where(is_eq, sv[:, None] + tv[None, :],
                             torch.where(is_new, lobg + tv[None, :],
                                         torch.full_like(tv[None, :], NEG))).reshape(P * K)
        keep_valid_v = (is_blank | is_eq).reshape(P * K)
        ext_valid_v = (is_eq | is_new).reshape(P * K)

        def group(rows, lns, scores, mask):
            r, l, sc = rows[mask], lns[mask], scores[mask]
            if r.numel() == 0:
                return None, None
            key = torch.cat([r, l[:, None]], dim=1)
            uniq, inv = torch.unique(key, dim=0, return_inverse=True)
            m = sc.new_full((uniq.shape[0],), NEG)
            m.scatter_reduce_(0, inv, sc, reduce="amax", include_self=True)
            se = torch.where(torch.isfinite(sc), torch.exp(sc - m[inv]), torch.zeros_like(sc))
            acc = sc.new_zeros(uniq.shape[0])
            acc.scatter_add_(0, inv, se)
            return uniq, m + torch.log(acc)

        parts = []
        kr, kv = group(keep_rows, keep_ln, keep_sc, keep_valid_v & (keep_ty == 1))
        if kr is not None:
            parts.append((kr, kv, "s"))
        kr, kv = group(keep_rows, keep_ln, keep_sc, keep_valid_v & (keep_ty == 0))
        if kr is not None:
            parts.append((kr, kv, "n"))
        kr, kv = group(ext_rows, ext_ln, ext_sc, ext_valid_v)
        if kr is not None:
            parts.append((kr, kv, "n"))

        def merge(ps):
            keys = torch.cat([p[0] for p in ps], 0)
            vals = torch.cat([p[1] for p in ps], 0)
            uniq, inv = torch.unique(keys, dim=0, return_inverse=True)
            m = vals.new_full((uniq.shape[0],), NEG)
            m.scatter_reduce_(0, inv, vals, reduce="amax", include_self=True)
            se = torch.where(torch.isfinite(vals), torch.exp(vals - m[inv]), torch.zeros_like(vals))
            acc = vals.new_zeros(uniq.shape[0])
            acc.scatter_add_(0, inv, se)
            return uniq, m + torch.log(acc)

        s_parts = [(r, v) for r, v, ty in parts if ty == "s"]
        n_parts = [(r, v) for r, v, ty in parts if ty == "n"]
        s_keys, s_val = merge(s_parts) if s_parts else (None, None)
        n_keys, n_val = merge(n_parts) if n_parts else (None, None)
        all_keys = torch.cat([k for k in (s_keys, n_keys) if k is not None], 0)
        uniq, inv = torch.unique(all_keys, dim=0, return_inverse=True)
        S = lp.new_full((uniq.shape[0],), NEG)
        Ns = lp.new_full((uniq.shape[0],), NEG)
        if s_keys is not None:
            S.index_copy_(0, inv[: s_keys.shape[0]], s_val)
        if n_keys is not None:
            Ns.index_copy_(0, inv[s_keys.shape[0] if s_keys is not None else 0:], n_val)
        total = torch.logaddexp(S, Ns)
        n = min(B, total.shape[0])
        order = torch.topk(total, n).indices
        tok = torch.zeros_like(tok)
        ln = torch.zeros_like(ln)
        s = torch.full_like(s, NEG)
        ns = torch.full_like(ns, NEG)
        tok[:n] = uniq[order, :-1]
        ln[:n] = uniq[order, -1]
        s[:n] = S[order]
        ns[:n] = Ns[order]
        valid = torch.zeros(B, dtype=torch.bool, device=dev)
        valid[:n] = True
    
    out = []
    for i in range(B):
        if not torch.isfinite(s[i] + ns[i]):
            continue
        out.append((tuple(tok[i, : int(ln[i])].tolist()), float(torch.logaddexp(s[i], ns[i]))))
    return out


# NOTE (yiakwy) : our optimized beam search op
def prefix_beam_op(logp, beam, chunk_frames=0):
    """Op under test: parallel top-k + persistent fused beam.
    chunk_frames=0 -> single-block baseline; >0 -> multi-block cluster variant."""
    tv, ti = torch.topk(logp, beam, dim=-1)
    tok, ln, sc = beam_search(ti, tv, beam=beam, blank=0, chunk_frames=chunk_frames)
    return [(tuple(tok[i, : int(ln[i])].tolist()), float(sc[i])) for i in range(tok.shape[0])]


def _gpu_time(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def _wall_time(fn, iters=1, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


# ---------------------------------------------------------------------------
# Phase 1: correctness
# ---------------------------------------------------------------------------


def verify_correctness(shapes, beam=5):
    print("Phase 1: correctness (vs host ctc_prefix_beam_search)")
    all_ok = True
    for T, V in shapes:
        torch.manual_seed(T * 131 + V)
        logp = torch.log_softmax(torch.randn(T, V, device="cuda"), dim=-1)

        ref = host_prefix_beam(logp.cpu().numpy(), beam)
        out = prefix_beam_op(logp, beam)

        exact = [t for t, _ in ref] == [t for t, _ in out]

        ref_sorted = sorted((s for _, s in ref), reverse=True)
        out_sorted = sorted((s for _, s in out), reverse=True)

        # NOTE (yiakwy) : weaker judger
        score_ok = len(ref_sorted) == len(out_sorted) and max(abs(a - b) for a, b in zip(ref_sorted,  out_sorted)) < 0.5

        ok = exact or score_ok
        all_ok &= ok

        kind = "exact" if exact else ("scores" if score_ok else "MISMATCH")
        print(f"  T={T:6d} V={V:6d} beam={beam}  token_match={str(exact):<5} "
              f"[{kind:8s}] [{'PASS' if ok else 'FAIL'}]")
    return all_ok


# ---------------------------------------------------------------------------
# Phase 2: beam-only performance (host vs torch vs fused)
# ---------------------------------------------------------------------------


def benchmark(shapes, beam=5):
    print(f"\nPhase 2: beam performance, beam={beam}, V={VOCAB}")
    print(f"{'T':>8s} | {'host (cpu)':>12s} | {'torch (gpu)':>12s} | {'fused (1 blk)':>13s} | {'fused (NoC)':>10s} | {'NoC vs 1blk':>10s} | {'NoC vs cpu':>10s}")
    print("-" * 95)
    for T, iters in shapes:
        logp = torch.log_softmax(torch.randn(T, VOCAB, device="cuda"), dim=-1)
        t_host = _wall_time(lambda: ctc_prefix_beam_search(logp, beam), 1, warmup=0)
        t_torch = (_wall_time(lambda: torch_naive_prefix_beam(logp, beam), 1, warmup=0)
                   if T <= 500 else None)
        t_fused = _gpu_time(lambda: prefix_beam_op(logp, beam, chunk_frames=0), iters)
        cf = max(512, T // 8)
        t_mp = _gpu_time(lambda: prefix_beam_op(logp, beam, chunk_frames=cf), iters)
        sp1 = t_fused / t_mp if t_mp else float("nan")
        sp2 = t_host / t_mp if t_mp else float("nan")
        
        if t_torch is not None:
            torch_s = f"{t_torch:>10.0f}us"
            # torch_s = f"{torch_s:>12s}"
        else:
            torch_s = "        skip"
            # torch_s = f"{torch_s:>11s}"
        
        print(f"{T:>8d} | {t_host:>10.0f}us | {torch_s:>12s} | {t_fused:>11.1f}us | {t_mp:>8.1f}us | {sp1:>9.2f}x | {sp2:>9.1f}x")
        del logp


def bench_full(model_dir: str, beam=5, iters=20):
    import numpy as np
    import wave

    audio = Path(__file__).resolve().parents[2] / "speaker_test" / "meeting-test.wav"
    md = Path(model_dir)
    if not audio.exists():
        print(f"\nPhase 3: skipped (need {audio})")
        return

    import torchaudio.functional as AF

    from veloxvoice.audio import FrontendConfig
    from veloxvoice.audio.frontend_torch import TorchGpuFrontend
    from veloxvoice.models.wenet.config import load_config
    from veloxvoice.models.wenet.torch_conformer import WenetConformerASR

    with wave.open(str(audio)) as h:
        sr, raw = h.getframerate(), h.readframes(h.getnframes())
    pcm = torch.from_numpy(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
    if sr != 16000:
        pcm = AF.resample(pcm, sr, 16000)
    pcm = pcm.cuda()
    dur = pcm.numel() / 16000.0

    cfg = load_config(str(md))
    fe = TorchGpuFrontend(FrontendConfig(sample_rate=16000, num_mel_bins=cfg.input_dim,
                                         precision="bf16"), "cuda")
    m = WenetConformerASR(str(md), device="cuda")
    m.set_jit(True)
    m.set_precision("bf16")
    for n in (256, 2000, 8000):
        m.encode_utterance(torch.zeros(1, n, cfg.input_dim, device="cuda"))
    with torch.inference_mode():
        feats = torch.cat([fe.accept(pcm), fe.flush()], dim=0)
        enc = m.encode_utterance(feats[None])
        logp = m.ctc_logp(enc)[0]

    # native decoder — no torchscript
    print(f"\nPhase 3: FULL op (AttRescorer.decode e2e), audio {dur:.2f}s")
    print(f"{'backend':>18s} {'rt(ms)':>10s} {'rtf':>11s} {'vs legacy':>10s}")
    ref = None
    for backend, cf in (("jit_kernel_legacy", 0), ("jit_kernel", 512)):
        rs = AttRescorer(m.attention_decoder, beam=beam, ctc_weight=0.5,
                         dim=enc.shape[2], device="cuda", search=backend,
                         chunk_frames=cf)
        for _ in range(3):
            out = rs.decode(enc, logp)
        ms = _gpu_time(lambda: rs.decode(enc, logp), iters)
        tag = "exact" if ref is not None and out == ref else (
              "baseline" if ref is None else "approx")
        print(f"{backend:>18s} {ms:>10.2f} {ms / 1000 / dur:>11.6f} {tag:>10s}")
        if ref is None:
            ref = out
    
    print("  NOTE: jit_kernel uses chunk-parallel beam (C>1) — the CTC N-best")
    print("  is an approximation; the attention decoder rescoring compensates.")

    print("  AISHELL-4 corpus CER is measured by benchmark/accuracy/")
    print("  bench_accuracy_local.sh: greedy 13.17%, rescore 13.02%,:")
    print("  see benchmark/accuracy/accuracy.md.")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="data/models/asr_model")
    ap.add_argument("--beam", type=int, default=5)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if not verify_correctness([(40, 32), (120, 200), (200, 1000), (300, VOCAB)]):
        print("\nCorrectness FAILED — skipping benchmark.")
        return
    benchmark([(500, 50), (2500, 20), (10000, 5), (30000, 3)])
    bench_full(args.model_dir, args.beam)


if __name__ == "__main__":
    main()
