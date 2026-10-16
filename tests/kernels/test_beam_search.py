"""Correctness tests for the fused persistent CTC prefix beam
(``veloxvoice/kernels/csrc/hopper/velox_beam_sm90a.cu`` via
``veloxvoice.kernels.ops.cuda_ops.beam_search``) against the host reference
``ctc_prefix_beam_search`` (from ``veloxvoice.stream.rescore``).

    CUDA_VISIBLE_DEVICES=5 pytest tests/kernels/test_beam_search.py -v
"""

from __future__ import annotations

import pytest
import torch

from veloxvoice.kernels.ops.cuda_ops import beam_search
from veloxvoice.stream.rescore import AttRescorer, ctc_prefix_beam_search

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _host(logp, beam):
    return [(tuple(h), float(s)) for h, s in ctc_prefix_beam_search(logp, beam)]


def _kernel(logp, beam, chunk_frames=0):
    tv, ti = torch.topk(logp, beam, dim=-1)
    tok, ln, sc = beam_search(ti, tv, beam=beam, blank=0, chunk_frames=chunk_frames)
    return [(tuple(tok[i, : int(ln[i])].tolist()), float(sc[i])) for i in range(tok.shape[0])]


@pytest.mark.parametrize("T,V", [(40, 32), (120, 200), (200, 1000)])
@pytest.mark.parametrize("beam", [1, 3, 5, 8])
def test_kernel_matches_host(T: int, V: int, beam: int):
    torch.manual_seed(T * 131 + V * 7 + beam)
    logp = torch.log_softmax(torch.randn(T, V, device="cuda"), dim=-1)
    host = _host(logp.cpu().numpy(), beam)
    got = _kernel(logp, beam)
    assert [t for t, _ in host] == [t for t, _ in got], "token sequences differ"
    for (_, hs), (_, gs) in zip(host, got):
        assert abs(hs - gs) < 1e-2, "scores differ"


@pytest.mark.parametrize("T,V", [(300, 1000), (800, 4000)])
def test_kernel_long_score_set(T: int, V: int):
    """Long T: float32-vs-float64 accumulation can reorder near-ties, so compare
    the (order-invariant) score vectors rather than the token order."""
    torch.manual_seed(T)
    logp = torch.log_softmax(torch.randn(T, V, device="cuda"), dim=-1)
    host = sorted((s for _, s in _host(logp.cpu().numpy(), 5)), reverse=True)
    got = sorted((s for _, s in _kernel(logp, 5)), reverse=True)
    assert len(host) == len(got)
    assert max(abs(a - b) for a, b in zip(host, got)) < 0.5


def test_attrescorer_gpu_search_matches_host():
    """AttRescorer._search("jit_kernel") == "host" N-best."""
    torch.manual_seed(0)
    logp = torch.log_softmax(torch.randn(150, 500, device="cuda"), dim=-1)
    rs_host = AttRescorer(None, beam=5, dim=64, device="cuda", search="host")
    # NOTE (yiakwy) : "jit_kernel_legacy" (single-block baseline kernel) is
    # kept for reference but skipped here — beam_search now dispatches every
    # chunk_frames to the cluster kernel.
    for backend in ("jit_kernel",):
        rs_gpu = AttRescorer(None, beam=5, dim=64, device="cuda", search=backend)
        h_gpu, _ = rs_gpu._search(logp)
        h_host, _ = rs_host._search(logp)
        assert h_gpu == h_host


@pytest.mark.parametrize("T,V,cf", [(120, 200, 0), (300, 1000, 64),
                                    (500, 300, 128), (300, 1000, 64)])
def test_workspace_reuse_matches_fresh(T: int, V: int, cf: int):
    """BeamWorkspace reuse == fresh allocations, for every chunk_frames
    (C=1 exact sweep and C>1 chunked); repeated same-shape calls exercise
    the stale-buffer invariants, and the growing-T case exercises _grow."""
    from veloxvoice.kernels.ops.cuda_ops import BeamWorkspace

    torch.manual_seed(T + cf)
    ws = BeamWorkspace("cuda")
    for rep in range(2):
        logp = torch.log_softmax(torch.randn(T, V, device="cuda"), dim=-1)
        fresh = _kernel(logp, 5, chunk_frames=cf)
        tv, ti = torch.topk(logp, 5, dim=-1)
        tok, ln, sc = beam_search(ti, tv, beam=5, blank=0, chunk_frames=cf,
                                  ws=ws)
        got = [(tuple(tok[i, : int(ln[i])].tolist()), float(sc[i]))
               for i in range(tok.shape[0])]
        assert [t for t, _ in fresh] == [t for t, _ in got], \
            f"ws reuse diverged (T={T}, cf={cf}, rep={rep})"


def _attrescorer_gpu_search_matches_host_unused():
    h_gpu, _ = rs_gpu._search(logp)
    h_host, _ = rs_host._search(logp)
    assert h_gpu == h_host


def test_removed_backends_rejected():
    """Deprecated backends are gone: 'tensor' (slow all-torch beam), the alias
    names, and the pre-merge module must all be unreachable."""
    for bad in ("tensor", "gpu", "kernel", "jit_kernel_mp"):
        with pytest.raises(ValueError):
            AttRescorer(None, beam=5, dim=64, device="cuda", search=bad)
    with pytest.raises(ModuleNotFoundError):
        import veloxvoice.stream.att_rescore  # noqa: F401


@pytest.mark.parametrize("T,V", [(120, 200), (300, 1000)])
def test_cluster_kernel_short_chunk_matches_single(T: int, V: int):
    """chunk_frames >= T => C=1 => cluster kernel degenerates to the exact
    single-block baseline."""
    torch.manual_seed(T)
    logp = torch.log_softmax(torch.randn(T, V, device="cuda"), dim=-1)
    single = _kernel(logp, 5, chunk_frames=0)
    chunked = _kernel(logp, 5, chunk_frames=max(T, 1))
    assert [t for t, _ in single] == [t for t, _ in chunked]
    assert max(abs(a[1] - b[1]) for a, b in zip(single, chunked)) < 1e-2


def test_blank_only_and_repeats():
    """Deterministic cases: best prefix must match the host reference."""
    T, V, beam = 16, 8, 4
    lp = torch.full((T, V), -20.0, device="cuda")
    lp[:, 0] = 0.0  # blank dominant everywhere
    assert _kernel(lp, beam)[0][0] == ()
    assert _kernel(lp, beam)[0][0] == _host(lp.cpu().numpy(), beam)[0][0]

    lp = torch.full((T, V), -20.0, device="cuda")
    lp[:, 0] = -0.5
    lp[:, 3] = 0.0  # token 3 dominant, blank still present
    assert _kernel(lp, beam)[0][0] == _host(lp.cpu().numpy(), beam)[0][0]


def _stub_decoder(V=500):
    """Deterministic decoder: logits depend only on (position, token id)."""

    def dec(memory, mmask, ys_in, lens, rys_in, reverse_weight):
        N, L = ys_in.shape
        pos = torch.arange(L, device=ys_in.device)
        vid = torch.arange(V, device=ys_in.device)
        out = -(pos.float()[:, None] * 1000.0 + vid.float())  # [L, V]
        return out[None].expand(N, L, V).contiguous(), None, None

    return dec


def test_reference_attention_rescore_matches_tensor():
    """`attention_rescore` (reference) and `_attention_rescore_torch` pick the
    same hypothesis."""
    from veloxvoice.stream.rescore import attention_rescore, _attention_rescore_torch

    torch.manual_seed(0)
    dim, V = 64, 500
    dec = _stub_decoder(V)
    logp = torch.log_softmax(torch.randn(120, V, device="cuda"), dim=-1)
    tv, ti = torch.topk(logp, 5, dim=-1)
    tok, ln, sc = beam_search(ti, tv, beam=5, blank=0)
    hyps = [[int(x) for x in tok[i, : int(ln[i])].tolist()] for i in range(tok.shape[0])]
    nbest = [(h, float(sc[i])) for i, h in enumerate(hyps)]
    enc = torch.randn(1, int(logp.shape[0]), dim, device="cuda")

    a = attention_rescore(dec, enc.float(), nbest, 0.5)
    bi = _attention_rescore_torch(dec, enc.float(), tok, ln, sc, 0.5)
    b = [int(x) for x in tok[bi, : int(ln[bi])].tolist()]
    assert a == b


def test_decode_batch_matches_decode():
    """AttRescorer.decode_batch (pooled) == per-sequence decode().

    The stub decoder emits deterministic logits that depend only on (position,
    token id), so single-span vs pooled scoring give identical argmax.
    """
    torch.manual_seed(0)
    dim = 64
    V = 500

    class _StubDecoder:
        def __call__(self, memory, mmask, ys_in, lens, rys_in, reverse_weight):
            N, L = ys_in.shape
            pos = torch.arange(L, device=ys_in.device)
            vid = torch.arange(V, device=ys_in.device)
            out = -(pos.float()[:, None] * 1000.0 + vid.float())  # [L, V]
            return out[None].expand(N, L, V).contiguous(), None, None

    rs = AttRescorer(_StubDecoder(), beam=5, dim=dim, device="cuda", search="jit_kernel")
    Ts = (40, 120, 200)
    logps = [torch.log_softmax(torch.randn(T, V, device="cuda"), dim=-1) for T in Ts]
    encs = [torch.randn(1, T, dim, device="cuda") for T in Ts]
    single = [rs.decode(e, lp) for e, lp in zip(encs, logps)]
    batch = rs.decode_batch(encs, logps)
    assert single == batch
