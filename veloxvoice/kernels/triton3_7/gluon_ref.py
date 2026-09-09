"""Quick-verification gluon (triton>=3.6) reference kernels.

First tier of the kernel harnessing loop: verify the hand-written CUDA kernels in
csrc/ against these compact gluon kernels on the same GPU before benchmarking.
"""

from __future__ import annotations

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _silu_glu_kernel(
    x_ptr, y_ptr, C: gl.constexpr, total: gl.constexpr, BLOCK: gl.constexpr
):
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0]
    )
    pid = gl.program_id(0)
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < total
    t = offs // C
    c = offs % C
    a = gl.load(x_ptr + t * 2 * C + c, mask=mask, other=0.0)
    b = gl.load(x_ptr + t * 2 * C + C + c, mask=mask, other=0.0)
    gl.store(y_ptr + offs, a * (1.0 / (1.0 + gl.exp(-b))), mask=mask)


def silu_glu(x, out=None):
    import torch

    total = x.shape[0] * x.shape[1] // 2
    C = x.shape[1] // 2
    out = (
        torch.empty(x.shape[0], C, device=x.device, dtype=x.dtype)
        if out is None
        else out
    )
    BLOCK = 512
    grid = ((total + BLOCK - 1) // BLOCK,)
    _silu_glu_kernel[grid](x, out, C, total, BLOCK)
    return out


@gluon.jit
def _dw_conv1d_kernel(
    x_ptr,
    cache_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    nc_ptr,
    T: gl.constexpr,
    K: gl.constexpr,
    C: gl.constexpr,
    BLOCK: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0]
    )
    tid = gl.program_id(1)
    offs = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=layout)  # channel ids
    mask = offs < C
    acc = gl.load(b_ptr + offs, mask=mask, other=0.0)
    for j in range(K):
        tt = tid + j - (K - 1)
        v = gl.where(
            tt < 0,
            gl.load(cache_ptr + (tt + K - 1) * C + offs, mask=mask, other=0.0),
            gl.load(x_ptr + tt * C + offs, mask=mask & (tt >= 0), other=0.0),
        )
        acc += gl.load(w_ptr + offs * K + j, mask=mask, other=0.0) * v
    gl.store(out_ptr + tid * C + offs, acc, mask=mask)
    n_it: gl.constexpr = ((K - 1) + T - 1) // T
    for i in range(n_it):  # static unroll over y threads (T & K are constexpr)
        r = tid + i * T
        rmask = mask & (r < K - 1)
        src = T + r
        cv = gl.load(cache_ptr + src * C + offs, mask=rmask & (src < K - 1), other=0.0)
        xv = gl.load(
            x_ptr + (src - (K - 1)) * C + offs, mask=rmask & (src >= K - 1), other=0.0
        )
        gl.store(nc_ptr + r * C + offs, cv + xv, mask=rmask)


def dw_causal_conv1d(x, cache, weight, bias):
    import torch

    T, C = x.shape
    K = weight.shape[1]
    out = torch.empty_like(x)
    new_cache = torch.empty_like(cache)
    BLOCK = 128
    _dw_conv1d_kernel[((C + BLOCK - 1) // BLOCK, T)](
        x, cache, weight, bias, out, new_cache, T, K, C, BLOCK
    )
    return out, new_cache
