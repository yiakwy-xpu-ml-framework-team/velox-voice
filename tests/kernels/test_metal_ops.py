import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from veloxvoice.kernels.ops import metal_ops as mo


pytestmark = pytest.mark.skipif(not mo.jit_enabled(), reason="Metal JIT disabled")


def _close(out, ref, atol):
    mx.eval(out, ref)
    a = np.array(out.astype(mx.float32))
    b = np.array(ref.astype(mx.float32))
    np.testing.assert_allclose(a, b, atol=atol, rtol=atol)


@pytest.mark.parametrize("dt,atol", [
    (mx.float32, 5e-5), (mx.float16, 2e-3), (mx.bfloat16, 2e-2)])
def test_layernorm(dt, atol):
    x = mx.random.normal((7, 1024)).astype(dt)
    w = mx.random.normal((1024,)).astype(dt)
    b = mx.random.normal((1024,)).astype(dt)
    _close(mo.fused_layernorm(x, w, b),
           mx.fast.layer_norm(x, w, b, 1e-5), atol)


@pytest.mark.parametrize("dt,atol", [
    (mx.float32, 5e-5), (mx.float16, 2e-3), (mx.bfloat16, 2e-2)])
def test_silu_glu(dt, atol):
    x = mx.random.normal((7, 2048)).astype(dt)
    a, b = mx.split(x, 2, axis=-1)
    _close(mo.silu_glu(x), a * mx.sigmoid(b), atol)


@pytest.mark.parametrize("dt,atol", [
    (mx.float32, 5e-5), (mx.float16, 2e-3), (mx.bfloat16, 2e-2)])
def test_mel_log_norm(dt, atol):
    p = mx.abs(mx.random.normal((13, 128))).astype(dt) + 1e-3
    mu = mx.random.normal((128,)).astype(dt)
    sd = mx.abs(mx.random.normal((128,))).astype(dt) + 0.5
    ref = 10 * mx.log10(mx.maximum(p, mx.array(1e-10, dtype=dt)))
    ref = (ref[:-1] - mu) / mx.maximum(sd, mx.array(1e-5, dtype=dt))
    _close(mo.mel_log_norm(p, mu, sd), ref, atol)


@pytest.mark.parametrize("dt,atol", [
    (mx.float32, 5e-5), (mx.float16, 2e-3), (mx.bfloat16, 2e-2)])
def test_gemv(dt, atol):
    x = mx.random.normal((1024,)).astype(dt)
    w = mx.random.normal((300, 1024)).astype(dt)
    _close(mo.gemv(x, w), (x[None] @ w.T)[0], atol)


def test_gemv_batch():
    x = mx.random.normal((8, 512)).astype(mx.bfloat16)
    w = mx.random.normal((2048, 512)).astype(mx.bfloat16)
    _close(mo.gemv_batch(x, w), x @ w.T, 2e-2)


def test_masked_argmax():
    logits = mx.random.normal((31678,)).astype(mx.float32)
    allowed = mx.zeros((31678,), dtype=mx.bool_)
    allowed[100] = True
    allowed[20000] = True
    ref = int(mx.argmax(mx.where(
        allowed, logits, mx.array(float("-inf"), dtype=mx.float32))))
    assert mo.masked_argmax(logits, allowed) == ref


def test_lm_head_argmax():
    x = mx.random.normal((512,)).astype(mx.bfloat16)
    w = mx.random.normal((31678, 512)).astype(mx.bfloat16)
    w[0] = x * 4.0
    allowed = mx.zeros((31678,), dtype=mx.bool_)
    allowed[0] = True
    logits = (x[None] @ w.T)[0]
    ref = int(mx.argmax(mx.where(
        allowed, logits, mx.array(float("-inf"), dtype=mx.bfloat16))))
    assert mo.lm_head_argmax(x, w, allowed) == ref
