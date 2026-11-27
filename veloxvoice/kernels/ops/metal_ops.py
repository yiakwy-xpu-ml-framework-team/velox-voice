"""MLX Metal JIT kernels for the Apple-silicon lane.

The kernels mirror the CUDA/DGX contracts in ``cuda_ops.py``:

  * ``fused_layernorm``  – fp32 statistics, dtype-preserving output
  * ``silu_glu``         – ``a * sigmoid(b)`` with the gate in the second half
  * ``mel_log_norm``     – power -> 10*log10(clamp) -> per-bin normalization
  * ``gemv``             – M=1 decode GEMV (one simdgroup per output row)
  * ``masked_argmax``    – grammar-masked greedy token selection

They are deliberately small, standalone and reference-checkable so the
harness in ``tools/kernel_harness.py`` can gate them with the 5-stage
smoke/shape/stability/determinism/edge-case protocol before the model uses
them.  Set ``VELOX_METAL_JIT=0`` to fall back to native MLX ops.
"""

from __future__ import annotations

import os
import pathlib

import mlx.core as mx

_METAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "kernels" / "csrc" / "metal"


def _load_metal(name: str) -> str:
    return (_METAL_DIR / name).read_text()

_JIT = os.environ.get("VELOX_METAL_JIT", "1") not in ("0", "false", "False")


def jit_enabled() -> bool:
    return _JIT


def set_jit(enabled: bool) -> None:
    global _JIT
    _JIT = bool(enabled)


# ---------------------------------------------------------------------------
# fused layernorm
# ---------------------------------------------------------------------------
_LN_SRC = _load_metal("fused_layernorm.metal")
_ln_kernel = mx.fast.metal_kernel(
    name="velox_fused_layernorm",
    input_names=["x", "w", "b", "eps"],
    output_names=["out"],
    source=_LN_SRC,
    header="#include <metal_simdgroup>\n",
    ensure_row_contiguous=True,
)


def fused_layernorm(x: mx.array, weight: mx.array, bias: mx.array,
                    eps: float = 1e-5) -> mx.array:
    """``layer_norm(x, (D,), weight, bias, eps)`` for x [N, D]."""
    if not _JIT:
        return mx.fast.layer_norm(x, weight, bias, eps)
    d = x.shape[-1]
    x2 = x.reshape(-1, d)
    n = x2.shape[0]
    warps = 8
    tg = warps * 32
    out = _ln_kernel(
        inputs=[x2, weight, bias, mx.array([eps], dtype=mx.float32)],
        template=[("T", x.dtype), ("D", d), ("N", n), ("WARPS", warps)],
        grid=(((n + warps - 1) // warps) * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[x2.shape],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(x.shape)


# ---------------------------------------------------------------------------
# GLU
# ---------------------------------------------------------------------------
_GLU_SRC = _load_metal("silu_glu.metal")
_glu_kernel = mx.fast.metal_kernel(
    name="velox_silu_glu",
    input_names=["x"],
    output_names=["out"],
    source=_GLU_SRC,
    header="#include <metal_math>\n",
    ensure_row_contiguous=True,
)


def silu_glu(x: mx.array) -> mx.array:
    """``a * sigmoid(b)`` over the last axis of x [..., 2C] -> [..., C]."""
    if not _JIT:
        a, b = mx.split(x, 2, axis=-1)
        return a * mx.sigmoid(b)
    c = x.shape[-1] // 2
    x2 = x.reshape(-1, 2 * c)
    n = x2.shape[0]
    vec = 4 if c % 4 == 0 else 1
    out = _glu_kernel(
        inputs=[x2],
        template=[("T", x.dtype), ("N", n), ("C", c), ("VEC", vec)],
        grid=((n * c + vec - 1) // vec, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n, c)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(*x.shape[:-1], c)


# ---------------------------------------------------------------------------
# mel power -> log -> normalize
# ---------------------------------------------------------------------------
_MEL_SRC = _load_metal("mel_log_norm.metal")

_mel_kernel = mx.fast.metal_kernel(
    name="velox_mel_log_norm",
    input_names=["x", "mean", "std"],
    output_names=["out"],
    source=_MEL_SRC,
    header="#include <metal_math>\n",
    ensure_row_contiguous=True,
)


def mel_log_norm(power: mx.array, mean: mx.array, std: mx.array) -> mx.array:
    """power [T, M] fp32 -> normalized log-mel [T-1, M] in ``power.dtype``."""
    t, m = power.shape
    if not _JIT:
        out = 10.0 * mx.log10(mx.maximum(power, mx.array(1e-10, dtype=power.dtype)))
        out = out[:-1]
        return (out - mean) / mx.maximum(std, mx.array(1e-5, dtype=power.dtype))
    out = _mel_kernel(
        inputs=[power, mean, std],
        template=[("T", power.dtype), ("TF", t), ("M", m)],
        grid=((t - 1) * m, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(t - 1, m)],
        output_dtypes=[power.dtype],
    )[0]
    return out


# ---------------------------------------------------------------------------
# M=1 GEMV (decode)
# ---------------------------------------------------------------------------
_GEMV_SRC = _load_metal("gemv.metal")

_gemv_kernel = mx.fast.metal_kernel(
    name="velox_gemv",
    input_names=["x", "w"],
    output_names=["out"],
    source=_GEMV_SRC,
    header="#include <metal_simdgroup>\n",
    ensure_row_contiguous=True,
)


def gemv(x: mx.array, weight: mx.array) -> mx.array:
    """M=1 GEMV: x [K] (or [1, K]) @ weight [N, K].T -> [N]."""
    if not _JIT:
        return (x.reshape(1, -1) @ weight.T)[0]
    k = weight.shape[1]
    n = weight.shape[0]
    rows_per_tg = 8
    tg = rows_per_tg * 32
    out = _gemv_kernel(
        inputs=[x.reshape(-1), weight],
        template=[("T", weight.dtype), ("N", n), ("K", k),
                  ("ROWS_PER_TG", rows_per_tg)],
        grid=(((n + rows_per_tg - 1) // rows_per_tg) * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[weight.dtype],
    )[0]
    return out


# ---------------------------------------------------------------------------
# grammar-masked argmax
# ---------------------------------------------------------------------------
_ARGMAX_SRC = _load_metal("masked_argmax.metal")

_argmax_kernel = mx.fast.metal_kernel(
    name="velox_masked_argmax",
    input_names=["logits", "allowed"],
    output_names=["out"],
    source=_ARGMAX_SRC,
    ensure_row_contiguous=True,
)


def masked_argmax(logits: mx.array, allowed: mx.array) -> int:
    """Greedy masked argmax over a [V] logits vector; returns a Python int."""
    if not _JIT:
        masked = mx.where(allowed, logits, mx.array(float("-inf"),
                                                   dtype=logits.dtype))
        return int(mx.argmax(masked))
    v = logits.shape[0]
    out = _argmax_kernel(
        inputs=[logits.reshape(-1), allowed.reshape(-1)],
        template=[("V", v)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.int32],
    )[0]
    return int(out[0])


# ---------------------------------------------------------------------------
# batched M=1..16 GEMV (paged windows): 64-bit vector loads, one warp/row,
# weight reused across M activation columns
# ---------------------------------------------------------------------------
_GEMV_BATCH_SRC = _load_metal("gemv_batch.metal")

_gemv_batch_kernel = mx.fast.metal_kernel(
    name="velox_gemv_batch",
    input_names=["x", "w"],
    output_names=["out"],
    source=_GEMV_BATCH_SRC,
    header="#include <metal_simdgroup>\n",
    ensure_row_contiguous=True,
)


def gemv_batch(x: mx.array, weight: mx.array) -> mx.array:
    """[B, K] @ weight [N, K].T -> [B, N] for B <= 16 (bf16).

    One warp owns one output row and keeps B accumulators, so a weight vector
    is loaded once and multiplied by all B paged windows.  K must be a
    multiple of 4 (true for 512/1024/2048).
    """
    if x.ndim == 1:
        x = x[None]
    m, k = x.shape
    n = weight.shape[0]
    if not _JIT or m > 16 or k % 4 or x.dtype != mx.bfloat16:
        return x @ weight.T
    return _gemv_batch_tuned(x, weight)


_GEMV_TUNE: dict = {}


def _gemv_batch_impl(x, weight, warps: int):
    m, k = x.shape
    n = weight.shape[0]
    tg = warps * 32
    n_tg = (n + warps - 1) // warps
    return _gemv_batch_kernel(
        inputs=[x, weight],
        template=[("T", x.dtype), ("M", m), ("N", n), ("K", k),
                  ("WARPS", warps)],
        grid=(n_tg * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]


def _gemv_batch_tuned(x, weight):
    """Tiny autotuner over the number of warps per threadgroup.

    The shape key is stable for paged serving (B, N, K), so the cost is paid
    once per bucket.  Set ``VELOX_METAL_TUNE=0`` to pin ``warps=8``.
    """
    import time
    m, k = x.shape
    n = weight.shape[0]
    key = (m, n, k)
    if os.environ.get("VELOX_METAL_TUNE", "1") in ("0", "false", "False"):
        return _gemv_batch_impl(x, weight, 8)
    best = _GEMV_TUNE.get(key)
    if best is None:
        best_w, best_t = 8, float("inf")
        for w in (4, 8, 16):
            if n < w:
                continue
            for _ in range(3):
                mx.eval(_gemv_batch_impl(x, weight, w))
            t0 = time.perf_counter()
            for _ in range(8):
                mx.eval(_gemv_batch_impl(x, weight, w))
            t = (time.perf_counter() - t0) / 8
            if t < best_t:
                best_w, best_t = w, t
        best = best_w
        _GEMV_TUNE[key] = best
    return _gemv_batch_impl(x, weight, best)


# ---------------------------------------------------------------------------
# fused LM-head GEMV + grammar-masked argmax
# ---------------------------------------------------------------------------
_LM_PARTIAL_SRC = _load_metal("lm_head_partial.metal")

_lm_partial_kernel = mx.fast.metal_kernel(
    name="velox_lm_head_partial",
    input_names=["x", "w", "allowed"],
    output_names=["partial_key", "partial_idx"],
    source=_LM_PARTIAL_SRC,
    header="#include <metal_simdgroup>\n",
    ensure_row_contiguous=True,
)

_LM_REDUCE_SRC = _load_metal("lm_head_reduce.metal")

_lm_reduce_kernel = mx.fast.metal_kernel(
    name="velox_lm_head_reduce",
    input_names=["partial_key", "partial_idx"],
    output_names=["out"],
    source=_LM_REDUCE_SRC,
    ensure_row_contiguous=True,
)


def lm_head_argmax(x: mx.array, weight: mx.array, allowed: mx.array) -> int:
    """Fused ``argmax(where(allowed, x @ weight.T, -inf))`` for M=1 decode.

    Avoids materializing the [V] logits vector and the second host round-trip.
    """
    if not _JIT:
        logits = (x.reshape(1, -1) @ weight.T)[0]
        return masked_argmax(logits, allowed)
    k = weight.shape[1]
    n = weight.shape[0]
    warps = 8
    tg = warps * 32
    n_tg = (n + warps - 1) // warps
    pk, pi = _lm_partial_kernel(
        inputs=[x.reshape(-1), weight, allowed.reshape(-1)],
        template=[("N", n), ("K", k), ("WARPS", warps)],
        grid=(n_tg * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n_tg * warps,), (n_tg * warps,)],
        output_dtypes=[mx.uint32, mx.int32],
    )
    out = _lm_reduce_kernel(
        inputs=[pk, pi],
        template=[("P", n_tg * warps)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.int32],
    )[0]
    return int(out[0])


# ---------------------------------------------------------------------------
# reference implementations + harness metadata
# ---------------------------------------------------------------------------
ATOL = 2e-2
RTOL = 2e-2
ALIGNMENT = 4


def ref_fused_layernorm(x, weight, bias, eps=1e-5):
    mu = mx.mean(x.astype(mx.float32), axis=-1, keepdims=True)
    var = mx.var(x.astype(mx.float32), axis=-1, keepdims=True)
    y = (x.astype(mx.float32) - mu) / mx.sqrt(var + eps)
    return y * weight.astype(mx.float32) + bias.astype(mx.float32)


def ref_silu_glu(x):
    a, b = mx.split(x, 2, axis=-1)
    return a * mx.sigmoid(b)


def ref_mel_log_norm(power, mean, std):
    mel = 10.0 * mx.log10(mx.maximum(power, 1e-10))
    mel = mel[:-1]
    return (mel - mean) / mx.maximum(std, 1e-5)


def ref_gemv(x, weight):
    return weight @ x


def ref_gemv_batch(x, weight):
    return x @ weight.T


def ref_masked_argmax(logits, allowed):
    masked = mx.where(allowed, logits, mx.array(float("-inf"),
                                                dtype=logits.dtype))
    return int(mx.argmax(masked))


def ref_lm_head_argmax(x, weight, allowed):
    logits = weight @ x
    return ref_masked_argmax(logits, allowed)


def _gen_glu(shape, fill=None):
    m = shape[0]
    if fill is None:
        return (mx.random.normal((m, 2048)).astype(mx.bfloat16),)
    return (mx.full((m, 2048), fill, dtype=mx.bfloat16),)


def _gen_ln(shape, fill=None):
    m = shape[0]
    if fill is None:
        return (mx.random.normal((m, 1024)).astype(mx.bfloat16),
                mx.random.normal((1024,)).astype(mx.bfloat16),
                mx.random.normal((1024,)).astype(mx.bfloat16))
    return (mx.full((m, 1024), fill, dtype=mx.bfloat16),
            mx.ones((1024,), dtype=mx.bfloat16),
            mx.zeros((1024,), dtype=mx.bfloat16))


def _gen_mel(shape, fill=None):
    t = max(2, shape[0])
    if fill is None:
        return (mx.random.uniform(1e-3, 10.0, (t, 128)).astype(mx.float32),
                mx.random.normal((128,)).astype(mx.float32),
                mx.random.uniform(0.5, 1.5, (128,)).astype(mx.float32))
    return (mx.full((t, 128), fill, dtype=mx.float32),
            mx.zeros((128,), dtype=mx.float32),
            mx.ones((128,), dtype=mx.float32))


def _gen_gemv(shape, fill=None):
    k = 512
    n = max(64, shape[0])
    if fill is None:
        return (mx.random.normal((k,)).astype(mx.bfloat16),
                mx.random.normal((n, k)).astype(mx.bfloat16))
    return (mx.full((k,), fill, dtype=mx.bfloat16),
            mx.full((n, k), fill, dtype=mx.bfloat16))


def _gen_argmax(shape, fill=None):
    n = max(64, shape[0])
    if fill is None:
        return (mx.random.normal((n,)).astype(mx.float32),
                mx.random.normal((n,)) > 0.5)
    return (mx.full((n,), fill, dtype=mx.float32),
            mx.ones((n,), dtype=mx.bool_))


def _gen_lm(shape, fill=None):
    k = 512
    n = 31678
    if fill is None:
        x = mx.random.normal((k,)).astype(mx.bfloat16)
        w = mx.random.normal((n, k)).astype(mx.bfloat16)
        w[0] = x * 4.0
        allowed = mx.zeros((n,), dtype=mx.bool_)
        allowed[0] = True
        return (x, w, allowed)
    return (mx.full((k,), fill, dtype=mx.bfloat16),
            mx.full((n, k), fill, dtype=mx.bfloat16),
            mx.ones((n,), dtype=mx.bool_))


def _gen_gemv_batch(shape, fill=None):
    b, k = 8, 512
    n = max(64, shape[0])
    if fill is None:
        return (mx.random.normal((b, k)).astype(mx.bfloat16),
                mx.random.normal((n, k)).astype(mx.bfloat16))
    return (mx.full((b, k), fill, dtype=mx.bfloat16),
            mx.full((n, k), fill, dtype=mx.bfloat16))


# Integration to Harness Loop

HARNESS_SPECS = {
    "fused_layernorm": {"fn": fused_layernorm, "ref": ref_fused_layernorm,
                        "shape_generator": _gen_ln, "num_inputs": 3},
    "silu_glu": {"fn": silu_glu, "ref": ref_silu_glu,
                 "shape_generator": _gen_glu, "num_inputs": 1},
    "mel_log_norm": {"fn": mel_log_norm, "ref": ref_mel_log_norm,
                     "shape_generator": _gen_mel, "num_inputs": 3},
    "gemv": {"fn": gemv, "ref": ref_gemv,
             "shape_generator": _gen_gemv, "num_inputs": 2},
    "gemv_batch": {"fn": gemv_batch, "ref": ref_gemv_batch,
                   "shape_generator": _gen_gemv_batch, "num_inputs": 2},
    "masked_argmax": {"fn": masked_argmax, "ref": ref_masked_argmax,
                      "shape_generator": _gen_argmax, "num_inputs": 2},
    "lm_head_argmax": {"fn": lm_head_argmax, "ref": ref_lm_head_argmax,
                       "shape_generator": _gen_lm, "num_inputs": 3},
}
