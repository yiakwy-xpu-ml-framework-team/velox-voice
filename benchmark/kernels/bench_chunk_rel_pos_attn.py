"""Benchmark for veloxvoice CUDA kernel: chunk_rel_pos_attn (bounded-cache chunked rel-pos attention).

Math (must match velox_chunk_rel_pos_attn.cu exactly):
  per (h, i, j): usable iff j >= L - valid[h,i] (the LAST vcnt slots)
    m_ac = dot(q_u[h,i,:], k[h,j,:])
    m_bd = dot(q_v[h,i,:], pe[idx[i,j], h*DK:(h+1)*DK])
    scores = (m_ac + m_bd) / sqrt(DK);  masked (-1e30) elsewhere
  probs = softmax over j;  out = probs @ v

Phase 1: correctness vs composed torch reference (fp32, TF32 disabled)
Phase 2: performance vs composed torch baseline

Run: python benchmark/kernels/bench_chunk_rel_pos_attn.py
"""

import math

import torch

from veloxvoice.kernels.ops import chunk_rel_pos_attn


def _reference(q_u, q_v, k, pe, idx, v, valid):
    """Composed torch implementation of the same math."""
    H, Tq, DK = q_u.shape
    L = k.shape[1]
    j = torch.arange(L, device=q_u.device)
    # usable = LAST vcnt slots  -> [H, Tq, L]
    mask = j.view(1, 1, L) >= (L - valid.long()).unsqueeze(-1)
    m_ac = torch.einsum("hid,hjd->hij", q_u, k)  # [H,Tq,L]
    pe_sel = pe[idx.long()].view(Tq, L, H, DK).permute(2, 0, 1, 3)  # [H,Tq,L,DK]
    m_bd = torch.einsum("hid,hijd->hij", q_v, pe_sel)
    scores = (m_ac + m_bd) / math.sqrt(DK)
    scores = torch.where(mask, scores, torch.full_like(scores, -1e30))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hij,hjd->hid", probs.float(), v)
    return probs, out


def _make_inputs(H, Tq, L, DK, SPAN, partial_valid=False, seed=0):
    torch.manual_seed(seed)
    q_u = torch.randn(H, Tq, DK, device="cuda", dtype=torch.float32)
    q_v = torch.randn(H, Tq, DK, device="cuda", dtype=torch.float32)
    k = torch.randn(H, L, DK, device="cuda", dtype=torch.float32)
    v = torch.randn(H, L, DK, device="cuda", dtype=torch.float32)
    pe = torch.randn(SPAN, H * DK, device="cuda", dtype=torch.float32)
    ii = torch.arange(Tq, device="cuda").unsqueeze(1)
    jj = torch.arange(L, device="cuda").unsqueeze(0)
    idx = (ii - jj + L - 1).clamp(0, SPAN - 1).to(torch.int32)  # [Tq, L]
    if partial_valid:
        valid = (L - (torch.arange(Tq, device="cuda") % 4) - 1).clamp(min=0)
        valid = valid.unsqueeze(0).expand(H, Tq).contiguous().to(torch.int32)
    else:
        valid = torch.full((H, Tq), L, device="cuda", dtype=torch.int32)
    return q_u, q_v, k, pe, idx, v, valid


def _verify(cases, atol=2e-4):
    print("Phase 1: correctness (CUDA vs composed torch reference, fp32)")
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ok = True
        for H, Tq, L, DK, SPAN, partial in cases:
            q_u, q_v, k, pe, idx, v, valid = _make_inputs(H, Tq, L, DK, SPAN, partial)
            ref_probs, ref_out = _reference(q_u, q_v, k, pe, idx, v, valid)
            probs, out = chunk_rel_pos_attn(q_u, q_v, k, pe, idx, v, valid)

            d_p = (probs - ref_probs).abs().max().item()
            d_o = (out - ref_out).abs().max().item()
            status = "PASS" if (d_p < atol and d_o < atol) else "FAIL"
            ok &= d_p < atol and d_o < atol
            tag = "partial-valid" if partial else "full-valid"
            print(
                f"  H={H} Tq={Tq} L={L} DK={DK} SPAN={SPAN} ({tag})  "
                f"max|probs-ref|={d_p:.3e}  max|out-ref|={d_o:.3e}  [{status}]"
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return ok


def _bench(fn, iters=100, warmup=20):
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


def _benchmark(cases):
    print("\nPhase 2: performance (CUDA-event GPU time)")
    print(f"{'config':>26s} {'cuda':>9s} {'torch':>9s} {'speedup':>8s} {'TFLOPS':>7s}")
    for H, Tq, L, DK, SPAN in cases:
        q_u, q_v, k, pe, idx, v, valid = _make_inputs(H, Tq, L, DK, SPAN)

        cuda_us = _bench(lambda: chunk_rel_pos_attn(q_u, q_v, k, pe, idx, v, valid))
        torch_us = _bench(lambda: _reference(q_u, q_v, k, pe, idx, v, valid))

        # scores 2*H*Tq*L*DK + ctx 2*H*Tq*L*DK + softmax
        flops = 4 * H * Tq * L * DK + 2 * H * Tq * L
        print(
            f"H={H:2d} Tq={Tq:3d} L={L:3d} DK={DK:3d} {cuda_us:8.2f}us {torch_us:8.2f}us "
            f"{torch_us/cuda_us:7.2f}x {flops/(cuda_us*1e-6)/1e12:6.2f}"
        )


def main():
    verify_cases = [
        (2, 8, 16, 32, 32, False),
        (2, 8, 16, 32, 32, True),
        (4, 16, 64, 64, 64, False),
        (4, 16, 64, 64, 64, True),
        (8, 32, 128, 64, 128, False),
    ]
    if not _verify(verify_cases):
        print("\nCorrectness FAILED — skipping benchmark.")
        return
    bench_cases = [
        (8, 64, 128, 64, 128),
        (8, 128, 256, 64, 256),
        (16, 128, 256, 128, 256),
    ]
    _benchmark(bench_cases)


if __name__ == "__main__":
    main()
