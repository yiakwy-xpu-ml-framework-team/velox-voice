"""Tests for veloxvoice CUDA kernels: layernorm, silu_glu, fused_qkv,
chunk_rel_pos_attn, dw_causal_conv1d.

Run: pytest tests/kernels/test_cuda_ops.py -v
"""

import math

import pytest
import torch

from veloxvoice.kernels.ops import (
    chunk_rel_pos_attn,
    dw_causal_conv1d,
    fused_layernorm,
    fused_qkv,
    reference_fused_qkv,
    reference_layernorm,
    silu_glu,
)


# ── layernorm ────────────────────────────────────────────────────────────────
class TestLayerNorm:
    @pytest.mark.parametrize("T,D", [(1, 64), (4, 128), (32, 256), (128, 512)])
    def test_correctness(self, T, D):
        torch.manual_seed(0)
        x = torch.randn(T, D, device="cuda", dtype=torch.float32)
        w = torch.randn(D, device="cuda", dtype=torch.float32)
        b = torch.randn(D, device="cuda", dtype=torch.float32)

        ref = reference_layernorm(x, w, b, eps=1e-5)
        out = fused_layernorm(x, w, b, eps=1e-5)

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_eps_stability(self):
        """Test that eps prevents division by zero for constant input."""
        T, D = 8, 64
        x = torch.ones(T, D, device="cuda", dtype=torch.float32) * 5.0
        w = torch.ones(D, device="cuda", dtype=torch.float32)
        b = torch.zeros(D, device="cuda", dtype=torch.float32)

        out = fused_layernorm(x, w, b, eps=1e-5)
        torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-5, rtol=1e-5)


# ── silu_glu ─────────────────────────────────────────────────────────────────
class TestSiLUGLU:
    @pytest.mark.parametrize("T,C", [(1, 32), (8, 64), (32, 128), (128, 256)])
    def test_correctness(self, T, C):
        torch.manual_seed(0)
        x = torch.randn(T, 2 * C, device="cuda", dtype=torch.float32)

        out = silu_glu(x)
        ref = x[:, :C] * torch.sigmoid(x[:, C:])

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        T, C = 16, 48
        x = torch.randn(T, 2 * C, device="cuda", dtype=torch.float32)
        out = silu_glu(x)
        assert out.shape == (T, C)


# ── fused_qkv ────────────────────────────────────────────────────────────────
class TestFusedQKV:
    @pytest.mark.parametrize("M,K,N", [(1, 64, 32), (8, 128, 64), (16, 256, 128)])
    def test_correctness(self, M, K, N):
        torch.manual_seed(0)
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        w = torch.randn(3 * N, K, device="cuda", dtype=torch.float32)
        bias = torch.randn(3 * N, device="cuda", dtype=torch.float32)

        ref = reference_fused_qkv(x, w, bias)
        out = fused_qkv(x, w, bias)

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_small_M(self):
        """fused_qkv is optimized for M <= 32."""
        M, K, N = 4, 128, 64
        torch.manual_seed(0)
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        w = torch.randn(3 * N, K, device="cuda", dtype=torch.float32)
        bias = torch.randn(3 * N, device="cuda", dtype=torch.float32)

        ref = reference_fused_qkv(x, w, bias)
        out = fused_qkv(x, w, bias)

        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


# ── chunk_rel_pos_attn ──────────────────────────────────────────────────────
class TestChunkRelPosAttn:
    def _make_inputs(self, H, Tq, L, DK, SPAN):
        torch.manual_seed(0)
        q_u = torch.randn(H, Tq, DK, device="cuda", dtype=torch.float32)
        q_v = torch.randn(H, Tq, DK, device="cuda", dtype=torch.float32)
        k = torch.randn(H, L, DK, device="cuda", dtype=torch.float32)
        v = torch.randn(H, L, DK, device="cuda", dtype=torch.float32)
        pe = torch.randn(SPAN, H * DK, device="cuda", dtype=torch.float32)
        # Index table: each query attends to a contiguous window of L keys
        idx = torch.zeros(Tq, L, device="cuda", dtype=torch.int32)
        for i in range(Tq):
            start = max(0, i - L // 2)
            end = min(L, start + L)
            idx[i, start:end] = torch.arange(
                start, end, device="cuda", dtype=torch.int32
            )
        valid = torch.full((H, Tq), L, device="cuda", dtype=torch.int32)
        return q_u, q_v, k, pe, idx, v, valid

    def test_output_shapes(self):
        H, Tq, L, DK, SPAN = 4, 8, 16, 64, 32
        q_u, q_v, k, pe, idx, v, valid = self._make_inputs(H, Tq, L, DK, SPAN)
        probs, out = chunk_rel_pos_attn(q_u, q_v, k, pe, idx, v, valid)
        assert probs.shape == (H, Tq, L)
        assert out.shape == (H, Tq, DK)

    def test_probs_sum_to_one(self):
        """Attention probs should sum to ~1 for each (h, i)."""
        H, Tq, L, DK, SPAN = 2, 4, 8, 32, 16
        q_u, q_v, k, pe, idx, v, valid = self._make_inputs(H, Tq, L, DK, SPAN)
        probs, _ = chunk_rel_pos_attn(q_u, q_v, k, pe, idx, v, valid)
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4)

    def test_deterministic(self):
        """Same inputs → same outputs."""
        H, Tq, L, DK, SPAN = 2, 4, 8, 32, 16
        q_u, q_v, k, pe, idx, v, valid = self._make_inputs(H, Tq, L, DK, SPAN)
        probs1, out1 = chunk_rel_pos_attn(q_u, q_v, k, pe, idx, v, valid)
        probs2, out2 = chunk_rel_pos_attn(q_u, q_v, k, pe, idx, v, valid)
        torch.testing.assert_close(probs1, probs2)
        torch.testing.assert_close(out1, out2)


# ── dw_causal_conv1d ────────────────────────────────────────────────────────
class TestDwCausalConv1d:
    def _reference(self, x, cache, weight, bias):
        """Reference: standard causal conv1d with cache.
        weight is [C, K] — depthwise: weight[c, k] is k-th tap for channel c.
        """
        T, C = x.shape
        K = weight.shape[1]
        # Prepend cache
        x_cat = torch.cat([cache, x], dim=0)  # [K-1+T, C]
        out = torch.zeros(T, C, device=x.device, dtype=x.dtype)
        for t in range(T):
            window = x_cat[t : t + K]  # [K, C]
            # Depthwise: out[t, c] = sum_k window[k, c] * weight[c, k]
            out[t] = (window * weight.T).sum(dim=0) + bias
        new_cache = x_cat[T : T + K - 1]  # last K-1 rows
        return out, new_cache

    @pytest.mark.parametrize("T,C,K", [(8, 16, 3), (16, 32, 5), (32, 64, 7)])
    def test_correctness(self, T, C, K):
        torch.manual_seed(0)
        x = torch.randn(T, C, device="cuda", dtype=torch.float32)
        cache = torch.randn(K - 1, C, device="cuda", dtype=torch.float32)
        weight = torch.randn(C, K, device="cuda", dtype=torch.float32)
        bias = torch.randn(C, device="cuda", dtype=torch.float32)

        out, new_cache = dw_causal_conv1d(x, cache, weight, bias)
        ref_out, ref_cache = self._reference(x, cache, weight, bias)

        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(new_cache, ref_cache, atol=1e-5, rtol=1e-5)

    def test_cache_update(self):
        """Verify cache is correctly updated after convolution."""
        T, C, K = 4, 8, 3
        torch.manual_seed(0)
        x = torch.randn(T, C, device="cuda", dtype=torch.float32)
        cache = torch.randn(K - 1, C, device="cuda", dtype=torch.float32)
        weight = torch.randn(C, K, device="cuda", dtype=torch.float32)
        bias = torch.zeros(C, device="cuda", dtype=torch.float32)

        _, new_cache = dw_causal_conv1d(x, cache, weight, bias)
        # new_cache should be the last K-1 rows of [cache; x]
        x_cat = torch.cat([cache, x], dim=0)
        expected = x_cat[T:]  # last K-1 rows
        torch.testing.assert_close(new_cache, expected, atol=1e-5, rtol=1e-5)
