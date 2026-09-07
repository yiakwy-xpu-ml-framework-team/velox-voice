"""FP8 block-wise quantization kernels (Triton), flash-float-1p2c framework semantics (Apache-2.0).

`act_quant` — per-row, per-K-block scale (activation).
`fp8_weight_block_wise_quant` — per-M-block × per-N-block scale (weights).
These produce the (fp8_tensor, scale) pairs consumed by block-scaled GEMM entries.
"""

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0  # OCP FP8 e4m3 max magnitude


@triton.jit
def fp8_weight_block_wise_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    M,
    N,
    SCALED_BLOCK_SIZE_M: tl.constexpr,
    SCALED_BLOCK_SIZE_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, SCALED_BLOCK_SIZE_N)
    offs_m = pid_m * SCALED_BLOCK_SIZE_M + tl.arange(0, SCALED_BLOCK_SIZE_M)
    offs_n = pid_n * SCALED_BLOCK_SIZE_N + tl.arange(0, SCALED_BLOCK_SIZE_N)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.max(tl.abs(x)) / FP8_MAX
    s = tl.where(s > 0, s, 1.0)  # avoid 0/0 -> NaN on all-zero blocks
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, s)


def fp8_weight_block_wise_quant(
    x: torch.Tensor,
    scaled_block_size_m: int = 128,
    scaled_block_size_n: int = 128,
):
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(
        triton.cdiv(M, scaled_block_size_m),
        triton.cdiv(N, scaled_block_size_n),
        dtype=torch.float32,
    )
    grid = lambda meta: (
        triton.cdiv(M, meta["SCALED_BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["SCALED_BLOCK_SIZE_N"]),
    )
    fp8_weight_block_wise_quant_kernel[grid](
        x,
        y,
        s,
        M,
        N,
        SCALED_BLOCK_SIZE_M=scaled_block_size_m,
        SCALED_BLOCK_SIZE_N=scaled_block_size_n,
        FP8_MAX=FP8_MAX,
    )
    return y, s


@triton.jit
def act_quant_kernel(
    x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, FP8_MAX: tl.constexpr
):
    """Per-row, per-K-block activation quantization."""
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / FP8_MAX
    s = tl.where(s > 0, s, 1.0)  # avoid 0/0 -> NaN on all-zero blocks
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128):
    """Quantize `x` per-row/per-K-block; returns (fp8 tensor, scale fp32 [..., K // block_size])."""
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0,
        (
            f"Last dimension size must be divisible by block_size (block_size={block_size})"
        ),
    )
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size, FP8_MAX=FP8_MAX)
    return y, s
