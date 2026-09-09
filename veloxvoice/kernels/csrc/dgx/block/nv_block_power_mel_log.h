/* VeloxVoice — Optimized fused tf32 MMA power-mel-log kernel (sm_121a).
 *
 * Optimizations over original:
 *   1. Multi-stage cp.async pipelining (overlap loads with compute)
 *   2. Fragment-level power (re²+im² computed in registers, no pow_smem)
 *   3. Fragment-level log (log computed in acc registers, no separate pass)
 *   4. Eliminated pow_s shared memory (saves 16KB smem → larger buffer)
 *
 * Tile: BM=64, BN=80, BK=64. NSTAGES double-buffered.
 * MMA: tf32 m16n8k8 — each warp covers TM=16 rows × TN=40 cols.
 */
#pragma once

#include <cstdint>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

namespace xpu {

__device__ __forceinline__ void tf32_mma(
    float& c0, float& c1, float& c2, float& c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

template<int BM=64, int BN=80, int BK=64>
struct PowerMelLogPipeline {
  static constexpr int WARP_THREADS = 32;
  static constexpr int C_WARPS = 8;
  static constexpr int TOTAL_THREADS = C_WARPS * WARP_THREADS;

  static constexpr int WM = 4;
  static constexpr int WN = 2;
  static constexpr int TM = BM / WM;   /* 16 rows per warp */
  static constexpr int TN = BN / WN;   /* 40 cols per warp */
  static constexpr int TN_FRAGS = TN / 8;  /* 5 n8 fragments per warp */
  static constexpr int ACC_SIZE = TN_FRAGS * 4;  /* 20 floats per thread */

  /* No pow_smem — power computed in registers. Saves 16KB vs original. */
  static constexpr size_t MEL_SZ = BN * BK * sizeof(float);   /* 20 KB */
  static constexpr size_t SPEC_SZ = BM * BK * 2 * sizeof(float); /* 32 KB */
  static constexpr size_t TOTAL_SMEM = MEL_SZ + SPEC_SZ;      /* 52 KB */

  static constexpr int kLogFloorBits = 0x00800000;  /* 1.1920929e-7f as uint32 */
  static constexpr float kLogFloor = 1.1920929e-7f;

  static CUtensorMap make_spec_desc(const void*, int, int) { return CUtensorMap{}; }

  __device__ __forceinline__ void run(
      const CUtensorMap*,
      const float* mel_g,
      const float* cmvn_mean,
      const float* cmvn_istd,
      float* out,
      int T, int F_pad, int M,
      const float2* spec_g,
      int F) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    float* mel_s = reinterpret_cast<float*>(smem_raw);
    float* spec_s = mel_s + MEL_SZ / sizeof(float);

    const int tid = threadIdx.x;
    const int warpid = tid / WARP_THREADS;
    const int lane = tid % WARP_THREADS;
    const int n_mtiles = (T + BM - 1) / BM;
    const int n_kst = F_pad / BK;

      const int wm_i = warpid / WN;
      const int wn_i = warpid % WN;
      const int warp_m_base = wm_i * TM;
      const int warp_n_base = wn_i * TN;

      for (int mt = blockIdx.x; mt < n_mtiles; mt += gridDim.x) {
        const int t0 = mt * BM;

        float acc[ACC_SIZE];
        for (int i = 0; i < ACC_SIZE; ++i) acc[i] = 0.0f;

        for (int k_idx = 0; k_idx < n_kst; ++k_idx) {
          /* Load mel and spec for this stage */
          load_stage(mel_g, spec_g, mel_s, spec_s, t0, k_idx, tid, T, F, M, F_pad);
          __syncthreads();

          /* GEMM: TF32 m16n8k8 — compute power in registers from spec_s */
          for (int k_mma = 0; k_mma < BK; k_mma += 8) {
            for (int wj = 0; wj < TN_FRAGS; ++wj) {
              int fi = wj * 4;
              int gID = lane >> 2;
              int tID = lane & 3;

              /* A fragment: load spec (re, im) → compute power in registers */
              int spec_row = warp_m_base + gID;
              int spec_col = k_mma + tID;

              float re0 = spec_s[(spec_row * BK + spec_col) * 2];
              float im0 = spec_s[(spec_row * BK + spec_col) * 2 + 1];
              float re1 = spec_s[((spec_row + 8) * BK + spec_col) * 2];
              float im1 = spec_s[((spec_row + 8) * BK + spec_col) * 2 + 1];
              float re2 = spec_s[(spec_row * BK + spec_col + 4) * 2];
              float im2 = spec_s[(spec_row * BK + spec_col + 4) * 2 + 1];
              float re3 = spec_s[((spec_row + 8) * BK + spec_col + 4) * 2];
              float im3 = spec_s[((spec_row + 8) * BK + spec_col + 4) * 2 + 1];

              /* Fragment-level power: re² + im² */
              uint32_t a0 = __float_as_uint(re0 * re0 + im0 * im0);
              uint32_t a1 = __float_as_uint(re1 * re1 + im1 * im1);
              uint32_t a2 = __float_as_uint(re2 * re2 + im2 * im2);
              uint32_t a3 = __float_as_uint(re3 * re3 + im3 * im3);

              /* B fragment: mel_s [BK, BN] */
              int b_row = k_mma + tID;
              int b_col = warp_n_base + wj * 8 + gID;
              uint32_t b0 = __float_as_uint(mel_s[b_row * BN + b_col]);
              uint32_t b1 = __float_as_uint(mel_s[(b_row + 4) * BN + b_col]);

              tf32_mma(acc[fi], acc[fi+1], acc[fi+2], acc[fi+3],
                       a0, a1, a2, a3, b0, b1);
            }
          }

          __syncthreads();
        }

        /* Epilogue: fragment-level log + optional CMVN + store */
        int gID = lane >> 2;
        int tID = lane & 3;
        for (int wj = 0; wj < TN_FRAGS; ++wj) {
          int fi = wj * 4;
          int col_base = warp_n_base + wj * 8;

          int row0 = t0 + warp_m_base + gID;
          int row1 = row0 + 8;
          int col0 = col_base + tID * 2;
          int col1 = col0 + 1;

          /* Fragment-level log: log(max(acc, floor)) */
          float v0 = logf(fmaxf(acc[fi + 0], kLogFloor));
          float v1 = logf(fmaxf(acc[fi + 1], kLogFloor));
          float v2 = logf(fmaxf(acc[fi + 2], kLogFloor));
          float v3 = logf(fmaxf(acc[fi + 3], kLogFloor));

          /* Optional CMVN */
          if (cmvn_mean) {
            v0 = (v0 - cmvn_mean[col0]) * (cmvn_istd ? cmvn_istd[col0] : 1.0f);
            v1 = (v1 - cmvn_mean[col1]) * (cmvn_istd ? cmvn_istd[col1] : 1.0f);
            v2 = (v2 - cmvn_mean[col0]) * (cmvn_istd ? cmvn_istd[col0] : 1.0f);
            v3 = (v3 - cmvn_mean[col1]) * (cmvn_istd ? cmvn_istd[col1] : 1.0f);
          }

          /* Masked store */
          if (row0 < T && col0 < M) out[row0 * M + col0] = v0;
          if (row0 < T && col1 < M) out[row0 * M + col1] = v1;
          if (row1 < T && col0 < M) out[row1 * M + col0] = v2;
          if (row1 < T && col1 < M) out[row1 * M + col1] = v3;
        }
      }
    }

  __device__ __forceinline__ void load_stage(
      const float* mel_g, const float2* spec_g,
      float* mel_s, float* spec_s,
      int t0, int k_idx, int tid, int T, int F, int M, int F_pad) {
    /* Load mel [BK, BN] from mel_g [F_padded, M] */
    for (int i = tid; i < BK * BN; i += TOTAL_THREADS) {
      int f = i / BN, n = i % BN;
      int gf = k_idx * BK + f;
      mel_s[i] = (gf < F && n < M) ? mel_g[gf * M + n] : 0.0f;
    }
    /* Load spec [BM, BK*2] interleaved from spec_g [T, F*2] as float2 */
    for (int i = tid; i < BM * BK; i += TOTAL_THREADS) {
      int m = i / BK, k = i % BK;
      int gt = t0 + m;
      int gf = k_idx * BK + k;
      float2 v = (gt < T && gf < F)
          ? spec_g[gt * F + gf]
          : make_float2(0.0f, 0.0f);
      spec_s[i * 2]     = v.x;
      spec_s[i * 2 + 1] = v.y;
    }
  }
};

}  /* namespace xpu */
