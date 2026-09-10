"""Gluon reference for chunk rel-pos attention (bounded cache, 2-stage kernels)."""

from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _scores_kernel(
    q_u_ptr,
    q_v_ptr,
    k_ptr,
    pe_ptr,
    idx_ptr,
    pr_ptr,
    H: gl.constexpr,
    Tq: gl.constexpr,
    L: gl.constexpr,
    DK: gl.constexpr,
    SPAN: gl.constexpr,
    inv_sqrt: gl.constexpr,
    LP2: gl.constexpr,
    DKP2: gl.constexpr,
):
    lay_2d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[8, 4],
        warps_per_cta=[4, 1],
        order=[0, 1],
    )
    lay_1d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0]
    )
    h = gl.program_id(1)
    i = gl.program_id(0)
    offs_j1 = gl.arange(0, LP2, layout=gl.SliceLayout(1, lay_2d))
    offs_j = offs_j1[:, None]
    offs_d = gl.arange(0, DKP2, layout=gl.SliceLayout(0, lay_2d))[None, :]
    jmask = offs_j1 < L
    dmask = offs_d < DK
    qu = gl.load(q_u_ptr + (h * Tq + i) * DK + offs_d, mask=dmask, other=0.0)
    qv = gl.load(q_v_ptr + (h * Tq + i) * DK + offs_d, mask=dmask, other=0.0)
    kj = gl.load(
        k_ptr + (h * L + offs_j) * DK + offs_d,
        mask=((offs_j1 < L)[:, None]) & dmask,
        other=0.0,
    )
    ac = gl.sum(qu * kj, 1)
    ij = gl.load(idx_ptr + i * L + offs_j, mask=(offs_j1 < L)[:, None], other=0)
    pj = gl.load(
        pe_ptr + (ij * (H * DK)) + h * DK + offs_d,
        mask=((offs_j1 < L)[:, None]) & dmask,
        other=0.0,
    )
    bd = gl.sum(qv * pj, 1)
    s = (ac + bd) * inv_sqrt
    s = gl.where(jmask, s, -1e30)
    m = gl.max(s, 0)
    e = gl.exp(s - m)
    denom = gl.sum(e, 0)
    prob = e / denom
    gl.store(pr_ptr + (h * Tq + i) * L + offs_j1, prob, mask=jmask)


@gluon.jit
def _ctx_kernel(
    pr_ptr,
    v_ptr,
    o_ptr,
    H: gl.constexpr,
    Tq: gl.constexpr,
    L: gl.constexpr,
    DK: gl.constexpr,
    LP2: gl.constexpr,
):
    lay_d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0]
    )
    h = gl.program_id(2)
    i = gl.program_id(1)
    d = gl.program_id(0)
    offs_j = gl.arange(0, LP2, layout=lay_d)
    jmask = offs_j < L
    p = gl.load(pr_ptr + (h * Tq + i) * L + offs_j, mask=jmask, other=0.0)
    vv = gl.load(v_ptr + (h * L + offs_j) * DK + d, mask=jmask, other=0.0)
    o = gl.sum(p * vv, 0)
    gl.store(o_ptr + (h * Tq + i) * DK + d, o)


def chunk_rel_pos_attn(q_u, q_v, k, pe, idx, v):
    import math

    H, Tq, DK = q_u.shape
    L = v.shape[1]
    probs = torch.empty(H, Tq, L, device=q_u.device, dtype=torch.float32)
    LP2 = 2 ** math.ceil(math.log2(L))
    DKP2 = 2 ** math.ceil(math.log2(DK))
    _scores_kernel[(Tq, H)](
        q_u, q_v, k, pe, idx, probs, H, Tq, L, DK, 2 * L - 1, DK**-0.5, LP2, DKP2
    )
    out = torch.empty(H, Tq, DK, device=probs.device, dtype=torch.float32)
    _ctx_kernel[(DK, Tq, H)](probs, v, out, H, Tq, L, DK, LP2)
    return probs, out
