"""Attention rescoring: CTC prefix beam + WeNet BiTransformerDecoder prefill.

Runtime: decoder scoring is a *prefill-type* computation, one batched forward
over the N-best hypotheses against the encoder memory:

    rs = AttRescorer(decoder, beam=5, ctc_weight=0.5, dim=512, device="cuda")
    ids = rs.decode(encoder_out, ctc_logp)      # [1,T,D], [T,V] -> list[int]

The CTC N-best source is pluggable via **search**:
  * "jit_kernel" : multi-block cluster kernel
    (``csrc/velox_prefix_beam_search.cu`` via ``kernels.ops.cuda_ops.beam_search``,
    chunk_frames>0; C=1 degenerates to the exact single-block baseline),
  * "host" : python device-top-k beam reference.

"""
from __future__ import annotations

import math
import threading
from collections import defaultdict

import numpy as np
import torch

SOS = 2
EOS = 2
BLANK = 0
NEG = float("-inf")


def _log_add(a: float, b: float) -> float:
    if a == NEG:
        return b
    if b == NEG:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def ctc_prefix_beam_search(logp, beam_size: int, blank_id: int = BLANK):
    """WeNet ctc_prefix_beam_search over log-softmax frames [T, V].

    Args:
        logp: log-softmax [T, V], CUDA tensor (preferred) or numpy array.
        beam_size: top tokens per frame and prefixes to keep.
        blank_id: CTC blank token index.

    Returns:
        list of ``(prefix_tokens, log_prob)`` sorted by log_prob desc.
    """
    if isinstance(logp, np.ndarray):
        logp = torch.from_numpy(logp)
    else:
        pass

    T = int(logp.shape[0])
    cur = {tuple(): [0.0, NEG]}
    for t in range(T):
        top = torch.topk(logp[t], beam_size)
        vals, idx = top.values.tolist(), top.indices.tolist()
        nxt = defaultdict(lambda: [NEG, NEG])
        for u, prob in zip(idx, vals):
            u = int(u)
            for prefix, (s, ns) in cur.items():
                last = prefix[-1] if prefix else None
                if u == blank_id:
                    # ctc collapse rule 1: blank keeps prefix, merges s and ns
                    nxt[prefix][0] = _log_add(nxt[prefix][0], _log_add(s, ns) + prob)
                elif u == last:
                    # ctc collapse rule 2: prefix unchanged / prefix extended
                    nxt[prefix][1] = _log_add(nxt[prefix][1], ns + prob)
                    nxt[prefix + (u,)][1] = _log_add(nxt[prefix + (u,)][1], s + prob)
                else:
                    nxt[prefix + (u,)][1] = _log_add(
                        nxt[prefix + (u,)][1], _log_add(s, ns) + prob)
        cur = dict(sorted(nxt.items(), key=lambda kv: _log_add(*kv[1]),
                          reverse=True)[:beam_size])
    ordered = sorted(cur.items(), key=lambda kv: _log_add(*kv[1]), reverse=True)
    return [(list(p), _log_add(*s)) for p, s in ordered]


@torch.inference_mode()
def attention_rescore(decoder, encoder_out: torch.Tensor, nbest, ctc_weight: float):
    """Score host ``nbest`` [(tokens, ctc_score)] with the decoder.

    Returns the best token list. Allocating reference form — identical math to
    :meth:`AttRescorer.decode`.
    """
    hyps = [h for h, _ in nbest]
    ctc_scores = [s for _, s in nbest]
    maxlen = max(len(h) for h in hyps)
    n = len(hyps)
    dev = encoder_out.device
    ys_in = torch.zeros((n, maxlen + 1), dtype=torch.long, device=dev)
    r_ys_in = torch.full((n, maxlen + 1), EOS, dtype=torch.long, device=dev)

    lens = torch.empty(n, dtype=torch.int32, device=dev)
    for i, h in enumerate(hyps):
        seq = [SOS] + h
        ys_in[i, : len(seq)] = torch.tensor(seq, device=dev)
        rseq = [SOS] + list(reversed(h))
        r_ys_in[i, : len(rseq)] = torch.tensor(rseq, device=dev)
        lens[i] = len(seq)
    memory = encoder_out.repeat(n, 1, 1)
    mmask = torch.ones(n, 1, encoder_out.shape[1], dtype=torch.bool, device=dev)

    logp_att, _, _ = decoder(memory, mmask, ys_in, lens, r_ys_in, 0.0)
    logp_att = torch.log_softmax(logp_att.float(), dim=-1)  # bundle emits logits

    scores = []
    for i, h in enumerate(hyps):
        s = 0.0
        for j, w in enumerate(h):
            s += float(logp_att[i, j, w])
        s += float(logp_att[i, len(h), EOS])
        scores.append(s + ctc_weight * ctc_scores[i])
    return hyps[int(np.argmax(scores))]


@torch.inference_mode()
def _attention_rescore_torch(decoder, encoder_out, tok, ln, ctc_scores, ctc_weight,
                              buf=None):
    """Vectorized attention scoring over the CTC N-best tensors."""
    B = int(tok.shape[0])
    Lmax = max(1, int(ln.max().item()) if ln.numel() else 1)
    tok = tok[:, :Lmax]  # trim N-best padding (the kernel pads to T)
    dev = encoder_out.device
    T = int(encoder_out.shape[1])
    width = Lmax + 1

    if buf is None:
        ys_in = torch.zeros(B, width, dtype=torch.long, device=dev)
        r_ys_in = torch.full((B, width), EOS, dtype=torch.long, device=dev)
        lens = (ln + 1).to(torch.int32)
        memory = encoder_out.repeat(B, 1, 1)
        mmask = torch.ones(B, 1, T, dtype=torch.bool, device=dev)
        rows = torch.arange(B, device=dev)
    else:
        buf._grow(T)
        if width > buf._ys.shape[1]:
            buf._ys = torch.zeros(buf.beam, width, dtype=torch.long, device=dev)
            buf._rys = torch.full((buf.beam, width), EOS, dtype=torch.long, device=dev)
        ys_in = buf._ys[:B, :width]
        r_ys_in = buf._rys[:B, :width]
        lens = buf._lens[:B]
        memory = buf._mem[:B]
        mmask = buf._mmask[:B]
        rows = buf._rows[:B]
        ys_in.zero_()
        r_ys_in.fill_(EOS)
        memory[:, :T].copy_(encoder_out[0])
        mmask[:, :, :T] = True
        mmask[:, :, T:] = False
        lens[:] = (ln + 1).to(torch.int32)

    ys_in[:, 0] = SOS
    ys_in[:, 1:1 + Lmax] = tok
    r_ys_in[:, 0] = SOS
    idx = ln[:, None] - 1 - torch.arange(Lmax, device=dev)[None, :]
    mask = idx >= 0
    r_tok = torch.gather(tok, 1, idx.clamp(min=0))
    r_ys_in[:, 1:1 + Lmax] = torch.where(mask, r_tok, torch.full_like(tok, EOS))

    logp_att, _, _ = decoder(memory, mmask, ys_in, lens, r_ys_in, 0.0)
    logp_att = torch.log_softmax(logp_att.float(), dim=-1)
    gpos = torch.arange(Lmax, device=dev)[None, :] < ln[:, None]
    fwd = logp_att[:, :Lmax, :].gather(2, tok[:, :, None].clamp(min=0)).squeeze(2)
    score = torch.where(gpos, fwd, torch.zeros_like(fwd)).sum(1)
    score = score + logp_att[rows, ln, EOS] + ctc_weight * ctc_scores
    return int(torch.argmax(score))


def load_attention_decoder(model, model_dir=None, device=None):
    """Resolve the attention decoder for rescoring.

    Preference order:
      1. ``model.attention_decoder`` — native eager decoder built from the
         checkpoint's own ``decoder.*`` weights at WenetConformerASR load
         time (no torchscript);
      2. lazy torchscript-bundle load from ``model_dir`` (older artifacts).

    Returns None when no decoder is available.
    """
    dec = getattr(model, "attention_decoder", None)
    if dec is not None:
        return dec
    md = model_dir if model_dir is not None else getattr(model, "model_dir", "")
    if not md:
        return None
    import os

    from ..models.wenet.torch_model import TorchWenEtModel

    bundle = os.path.join(md, "offline")
    if not os.path.exists(os.path.join(bundle, "final.zip")):
        bundle = md
    if not os.path.exists(os.path.join(bundle, "final.zip")):
        return None
    return TorchWenEtModel(bundle, device=str(device)).script.decoder


class AttRescorer:
    """CTC N-best search (``_search_tensor``) + attention scoring
    (``_attention_rescore_torch``) with preallocated reusable buffers."""

    def __init__(self, decoder, beam: int = 5, ctc_weight: float = 0.5,
                 blank_id: int = BLANK, dim: int = 512, max_len: int = 8192,
                 device: str | torch.device = "cuda", search: str = "jit_kernel",
                 chunk_frames: int = 512):
        self.decoder = decoder
        self.beam = int(beam)
        self.ctc_weight = float(ctc_weight)
        self.blank_id = int(blank_id)
        self.dim = int(dim)
        self.device = torch.device(device)
        # backends: "jit_kernel"        -> multi-block cluster kernel (default,
        #                                  chunk_frames>0, block 0 stitches the
        #                                  N-best; C=1 degenerates to exact)
        #          "jit_kernel_legacy" -> accepted for compat; beam_search now
        #                                  dispatches every chunk_frames to the
        #                                  same cluster kernel (C=1 here)
        #          "host"              -> python device-top-k beam reference
        # Deprecated aliases ("gpu", "kernel", "jit_kernel_mp", "tensor") are
        # gone on purpose.
        if search not in ("jit_kernel", "jit_kernel_legacy", "host"):
            raise ValueError(
                f"AttRescorer search={search!r} not in "
                "('jit_kernel', 'jit_kernel_legacy', 'host')")
        self.search = search
        # multi-block chunked variant: frames per chunk (experiment)
        self.chunk_frames = int(chunk_frames)
        # reusable GPU scratch for the beam op (allocated on first use, grown
        # on demand; created here so requests after model load never allocate)
        from ..kernels.ops.cuda_ops import BeamWorkspace

        self._ws = BeamWorkspace(self.device)
        n = self.beam
        # decoder-input scratch (grown to the worst-case hypothesis length)
        self._ys = torch.zeros(n, max_len + 1, dtype=torch.long, device=self.device)
        self._rys = torch.full((n, max_len + 1), EOS, dtype=torch.long, device=self.device)
        self._lens = torch.zeros(n, dtype=torch.int32, device=self.device)
        # encoder-memory scratch; grown on demand to the span length
        self._mem = torch.zeros(n, 1, self.dim, device=self.device)
        self._mmask = torch.zeros(n, 1, 1, dtype=torch.bool, device=self.device)
        self._rows = torch.arange(n, device=self.device)

    def _grow(self, tmax: int) -> None:
        if self._mem.shape[1] >= tmax:
            return
        self._mem = torch.zeros(self.beam, tmax, self.dim, device=self.device)
        self._mmask = torch.zeros(self.beam, 1, tmax, dtype=torch.bool, device=self.device)

    def _search_tensor(self, logp: torch.Tensor):
        """CTC N-best as tensors: ``(tokens [B,L], lens [B], ctc_scores [B])``."""
        if self.search in ("jit_kernel", "jit_kernel_legacy"):
            from ..kernels.ops.cuda_ops import beam_search

            tv, ti = torch.topk(logp, self.beam, dim=-1)
            tok, ln, sc = beam_search(ti, tv, beam=self.beam, blank=self.blank_id,
                                      chunk_frames=(self.chunk_frames
                                                    if self.search == "jit_kernel"
                                                    else 0),
                                      ws=self._ws)
            return tok, ln, sc.float()

        nb = ctc_prefix_beam_search(logp.float(), self.beam, self.blank_id)

        B = len(nb)
        Lc = max(1, max((len(h) for h, _ in nb), default=0))
        tok = torch.zeros(B, Lc, dtype=torch.long, device=self.device)
        ln = torch.zeros(B, dtype=torch.long, device=self.device)
        scores = torch.tensor([s for _, s in nb], device=self.device, dtype=torch.float32)
        for i, (h, _) in enumerate(nb):
            if h:
                tok[i, :len(h)] = torch.as_tensor(h, dtype=torch.long, device=self.device)
            ln[i] = len(h)
        return tok, ln, scores

    def _search(self, logp: torch.Tensor):
        """CTC N-best as ``(hyps: list[list[int]], ctc_scores: Tensor)``."""
        tok, ln, scores = self._search_tensor(logp)
        hyps = [[int(x) for x in tok[i, : int(ln[i])].tolist()]
                for i in range(tok.shape[0])]
        return hyps, scores

    @torch.inference_mode()
    def decode(self, encoder_out: torch.Tensor, logp: torch.Tensor) -> list[int]:
        """encoder_out [1,T,D], logp [T,V] -> best token list."""
        tok, ln, ctc_scores = self._search_tensor(logp)
        if tok.shape[0] == 0:
            return []
        best = _attention_rescore_torch(self.decoder, encoder_out, tok, ln,
                                        ctc_scores, self.ctc_weight, buf=self)
        return [int(x) for x in tok[best, : int(ln[best])].tolist()]

    @torch.inference_mode()
    def decode_batch(self, encoders: list, logps: list) -> list[list[int]]:
        """Pooled attention rescoring over many sequences in ONE decoder forward.

        All sequences' N-best are concatenated into one decoder batch, amortizing 
        decoder GEMMs/launches across the pool. 
        
        Returns one token list per sequence.
        """
        flat = []  # (owner, tokens, ctc_score)
        for oi, lg in enumerate(logps):
            hyps, scores = self._search(lg)
            for j, hh in enumerate(hyps):
                flat.append((oi, hh, float(scores[j])))
        if not flat:
            return [[] for _ in encoders]

        N = len(flat)
        dev = self.device
        Lmax = max(1, max(len(f[1]) for f in flat))
        Tmax = max(int(e.shape[1]) for e in encoders)
        D = int(encoders[0].shape[2])

        ys = torch.zeros(N, Lmax + 1, dtype=torch.long, device=dev)
        rys = torch.full((N, Lmax + 1), EOS, dtype=torch.long, device=dev)
        lens = torch.zeros(N, dtype=torch.int32, device=dev)
        memory = torch.zeros(N, Tmax, D, device=dev)
        mmask = torch.zeros(N, 1, Tmax, dtype=torch.bool, device=dev)
        ctc = torch.empty(N, dtype=torch.float32, device=dev)

        for i, (oi, hh, sc) in enumerate(flat):
            ys[i, 0] = SOS
            rys[i, 0] = SOS
            L = len(hh)
            if L:
                ys[i, 1:1 + L] = torch.as_tensor(hh, dtype=torch.long, device=dev)
                rys[i, 1:1 + L] = torch.as_tensor(hh[::-1], dtype=torch.long, device=dev)
            lens[i] = L + 1
            t = int(encoders[oi].shape[1])
            memory[i, :t] = encoders[oi][0]
            mmask[i, 0, :t] = True
            ctc[i] = sc

        logp_att, _, _ = self.decoder(memory, mmask, ys, lens, rys, 0.0)
        logp_att = torch.log_softmax(logp_att.float(), dim=-1)

        rows = torch.arange(N, device=dev)
        ln = lens.long() - 1
        gpos = torch.arange(Lmax, device=dev)[None, :] < ln[:, None]
        fwd = logp_att[:, :Lmax, :].gather(2, ys[:, 1:1 + Lmax, None].clamp(min=0)).squeeze(2)
        score = torch.where(gpos, fwd, torch.zeros_like(fwd)).sum(1)
        score = score + logp_att[rows, ln, EOS] + self.ctc_weight * ctc

        best_of = {}
        for i, (oi, _, _) in enumerate(flat):
            if oi not in best_of or score[i] > score[best_of[oi]]:
                best_of[oi] = i

        out = []
        for oi in range(len(encoders)):
            i = best_of.get(oi)
            out.append([] if i is None else [int(x) for x in flat[i][1]])
        return out


_RESCORER_CACHE: dict = {}
_RESCORER_LOCK = threading.Lock()


def get_att_rescorer(decoder, beam: int = 5, ctc_weight: float = 0.5,
                     dim: int = 512, device: str | torch.device = "cuda",
                     search: str = "jit_kernel", chunk_frames: int = 512):
    """Cached :class:`AttRescorer` factory (GPU buffers created once, grown on
    demand). Callers must serialize decode on the returned instance (the
    serving path already does via ``Velox._lock``)."""
    key = (decoder, int(beam), float(ctc_weight), int(dim),
           torch.device(device), str(search), int(chunk_frames))
    with _RESCORER_LOCK:
        rs = _RESCORER_CACHE.get(key)
        if rs is None:
            rs = AttRescorer(decoder, beam=beam, ctc_weight=ctc_weight, dim=dim,
                             device=device, search=search,
                             chunk_frames=chunk_frames)
            _RESCORER_CACHE[key] = rs
    return rs


def decode(encoder_out: torch.Tensor, logp: torch.Tensor, decoder,
           beam: int = 5, ctc_weight: float = 0.5) -> list[int]:
    """attention_rescoring decode for one span (encoder_out [1,T,D], logp [T,V]).

    Uses a cached AttRescorer (buffers allocated once and reused)."""
    rs = get_att_rescorer(decoder, beam=beam, ctc_weight=ctc_weight,
                          dim=encoder_out.shape[2], device=encoder_out.device)
    return rs.decode(encoder_out, logp)


def decode_batch(encoders: list, logps: list, decoder,
                 beam: int = 5, ctc_weight: float = 0.5) -> list[list[int]]:
    """Pooled attention rescoring (fused beam + one decoder forward)."""
    rs = get_att_rescorer(decoder, beam=beam, ctc_weight=ctc_weight,
                          dim=encoders[0].shape[2], device=encoders[0].device)
    return rs.decode_batch(encoders, logps)
