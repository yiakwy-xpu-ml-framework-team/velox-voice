/* VeloxVoice — Blackwell (sm_121a) mxfp4 accumulator + fragment view.
 *
 * Follows flash-float pattern:
 *   Mxfp4Accumulator — register file + mma_scaled (load fragments + MMA loop)
 *   FragmentView     — shared memory tile view for epilogue staging
 *
 * ldmatrix path: USE_LDMATRIX=1 triggers cooperative warp-level fragment
 * loads (fewer instructions than scalar lds).
 */
#pragma once

#include <cstdint>
#include <cuda.h>

#include "fragment/nv_frag_mxfp4.h"

#ifndef USE_LDMATRIX
#define USE_LDMATRIX 0
#endif

namespace xpu {

template <int _TM, int _TN>
struct Mxfp4Accumulator {
  static constexpr int TM = _TM;
  static constexpr int TN = _TN;
  static constexpr int REGS = TM * TN * 4;

  float acc[REGS];

  __device__ inline void clear() {
#pragma unroll
    for (int i = 0; i < REGS; ++i) acc[i] = 0.f;
  }

#if USE_LDMATRIX

  __device__ inline void load_af(const uint32_t* shmem_X, int warpM,
                                 int KWP, int lane_id, uint32_t af[TM][4]) {
    const int g = lane_id / 4, t = lane_id % 4;
#pragma unroll
    for (int mi = 0; mi < TM; ++mi) {
      const int base = warpM + mi * 16;
      /* 4 ldmatrix.x4: rows [0..3], [4..7], [8..11], [12..15] */
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        int row = base + k * 4 + (lane_id % 4);
        uint32_t ptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shmem_X[row * KWP + t * 2]));
        asm volatile(
            "ldmatrix.sync.aligned.m8n4.x4.shared.b128 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(af[mi][k*4+0]), "=r"(af[mi][k*4+1]),
              "=r"(af[mi][k*4+2]), "=r"(af[mi][k*4+3])
            : "r"(ptr));
      }
    }
  }

  __device__ inline void load_bf(const uint32_t* shmem_W, int warpN,
                                 int KWP, int lane_id, uint32_t bf[TN][2]) {
    const int g = lane_id / 4, t = lane_id % 4;
#pragma unroll
    for (int ni = 0; ni < TN; ++ni) {
      int row = warpN + ni * 8 + g;
      uint32_t ptr = static_cast<uint32_t>(
          __cvta_generic_to_shared(&shmem_W[row * KWP + t * 2]));
      asm volatile(
          "ldmatrix.sync.aligned.m8n4.x2.shared.b128 {%0,%1}, [%2];\n"
          : "=r"(bf[ni][0]), "=r"(bf[ni][1])
          : "r"(ptr));
    }
  }

#else
  /* Scalar path: 6 lds.u32 per (mi,ni) pair. */
  __device__ inline void load_af(const uint32_t* shmem_X, int warpM,
                                 int KWP, int lane_id, uint32_t af[TM][4]) {
    const int g = lane_id / 4, t = lane_id % 4;
#pragma unroll
    for (int mi = 0; mi < TM; ++mi) {
      int rE = warpM + mi * 16 + 2 * g, rO = rE + 1;
      af[mi][0] = shmem_X[rE * KWP + t * 2];
      af[mi][1] = shmem_X[rO * KWP + t * 2];
      af[mi][2] = shmem_X[rE * KWP + t * 2 + 1];
      af[mi][3] = shmem_X[rO * KWP + t * 2 + 1];
    }
  }

  __device__ inline void load_bf(const uint32_t* shmem_W, int warpN,
                                 int KWP, int lane_id, uint32_t bf[TN][2]) {
    const int g = lane_id / 4, t = lane_id % 4;
#pragma unroll
    for (int ni = 0; ni < TN; ++ni) {
      int col = warpN + ni * 8 + g;
      bf[ni][0] = shmem_W[col * KWP + t * 2];
      bf[ni][1] = shmem_W[col * KWP + t * 2 + 1];
    }
  }
#endif

  __device__ inline void mma_scaled(const uint32_t* shmem_X, const uint32_t* shmem_W,
                                    int warpM, int warpN,
                                    int KWP, int lane_id) {
    uint32_t af[TM][4], bf[TN][2];
    load_af(shmem_X, warpM, KWP, lane_id, af);
    load_bf(shmem_W, warpN, KWP, lane_id, bf);

#pragma unroll
    for (int mi = 0; mi < TM; ++mi)
#pragma unroll
      for (int ni = 0; ni < TN; ++ni)
        mxfp4_mma(acc + (mi * TN + ni) * 4, af[mi], bf[ni]);
  }

  /* Block-scaled MMA: feeds real per-block ue8m0 scales into the hardware
   * mxf4nvf4.block_scale.scale_vec::2X instruction (sm_121a, probed).
   *
   * Validated instruction semantics:
   *   - thread (g,t): scale u32 byte0 = ue8m0 for own k-slice [16t,16t+8),
   *                   byte1 = ue8m0 for k-slice [16t+8,16t+16);
   *   - A-side scale is SHARED across the thread's row-pair (2g,2g+1);
   *   - B-side scale is independent per column; bytes 2-3 ignored; z u16 = 0.
   *
   * Quantization contract (per 16-column block, one ue8m0):
   *   - A: rows grouped pairwise; scale grid [K/16][BM/2]
   *   - B: per column;            scale grid [K/16][BN]
   * Per stage (BK=64 -> 4 blocks of 16): thread t owns block #t, so its two
   * scale slots are identical:  u32 = u8 * 0x0101.
   *
   * sx: [STAGES][4 * BM/2] u8 A-scale tile; sy: [STAGES][4 * BN] u8 B-scale tile.
   */
  __device__ inline void mma_scaled_blk(const uint32_t* shmem_X, const uint32_t* shmem_W,
                                        const uint8_t* sx, const uint8_t* sy,
                                        int BM2 /* = BM/2 */, int BN,
                                        int warpM, int warpN, int KWP, int lane_id) {
    const int g = lane_id / 4, t = lane_id % 4;
    uint32_t af[TM][4], bf[TN][2];
    load_af(shmem_X, warpM, KWP, lane_id, af);
    load_bf(shmem_W, warpN, KWP, lane_id, bf);

#pragma unroll
    for (int mi = 0; mi < TM; ++mi) {
      const int gp = warpM / 2 + mi * 8 + g;  /* row-pair local index */
      uint32_t sa = (uint32_t)sx[t * BM2 + gp] * 0x0101u;
#pragma unroll
      for (int ni = 0; ni < TN; ++ni) {
        const int c = warpN + ni * 8 + g;  /* column local index */
        uint32_t sb = (uint32_t)sy[t * BN + c] * 0x0101u;
        mxfp4_mma(acc + (mi * TN + ni) * 4, af[mi], bf[ni], sa, sb);
      }
    }
  }

  template <int BM, int BN>
  __device__ inline void store(float* smem, int warpM, int warpN,
                               int lane_id, int M, int N) {
    const int g = lane_id / 4, t = lane_id % 4;
#pragma unroll
    for (int mi = 0; mi < TM; ++mi) {
#pragma unroll
      for (int ni = 0; ni < TN; ++ni) {
        const int r0 = warpM + mi * 16 + 2 * g;
        const int c0 = warpN + ni * 8 + 2 * t;
        const int base = (mi * TN + ni) * 4;
        if (r0   < M && c0   < N) smem[r0 * BN + c0]         = acc[base];
        if (r0   < M && c0+1 < N) smem[r0 * BN + c0 + 1]     = acc[base+1];
        if (r0+1 < M && c0   < N) smem[(r0+1) * BN + c0]     = acc[base+2];
        if (r0+1 < M && c0+1 < N) smem[(r0+1) * BN + c0 + 1] = acc[base+3];
      }
    }
  }

  /* mxfp4 epilogue: raw write, no scale multiply (scales were consumed in-MMA). */
  template <int BM, int BN>
  __device__ inline void store_raw(float* Out, int baseM, int baseN, int M, int N,
                                   int warpM, int warpN, int lane_id) {
    const int g = lane_id / 4, t = lane_id % 4;
#pragma unroll
    for (int mi = 0; mi < TM; ++mi) {
#pragma unroll
      for (int ni = 0; ni < TN; ++ni) {
        const int row0 = baseM + warpM + mi * 16 + 2 * g;
        const int col0 = baseN + warpN + ni * 8 + 2 * t;
        const int base = (mi * TN + ni) * 4;
        if (row0   < M && col0   < N) Out[(row0+0)*N+col0+0] = acc[base+0];
        if (row0   < M && col0+1 < N) Out[(row0+0)*N+col0+1] = acc[base+1];
        if (row0+1 < M && col0   < N) Out[(row0+1)*N+col0+0] = acc[base+2];
        if (row0+1 < M && col0+1 < N) Out[(row0+1)*N+col0+1] = acc[base+3];
      }
    }
  }

  template <int BM, int BN>
  __device__ inline void store(float* Out, const float* xs_fp32,
                               const float* ws_fp32,
                               int baseM, int baseN, int M, int N,
                               int warpM, int warpN, int lane_id) {
    const int g = lane_id / 4, t = lane_id % 4;
#pragma unroll
    for (int mi = 0; mi < TM; ++mi) {
#pragma unroll
      for (int ni = 0; ni < TN; ++ni) {
        const int row0 = baseM + warpM + mi * 16 + 2 * g;
        const int col0 = baseN + warpN + ni * 8 + 2 * t;
        const int base = (mi * TN + ni) * 4;
        const float s_r0 = (row0   < M) ? xs_fp32[row0 - baseM] : 0.f;
        const float s_r1 = (row0+1 < M) ? xs_fp32[row0+1 - baseM] : 0.f;
        const float s_c0 = (col0   < N) ? ws_fp32[col0 - baseN] : 0.f;
        const float s_c1 = (col0+1 < N) ? ws_fp32[col0+1 - baseN] : 0.f;

        // TODO (yiakwy) : using TMA store
        if (row0   < M && col0   < N) Out[(row0+0)*N+col0+0] = acc[base+0] * s_r0 * s_c0;
        if (row0   < M && col0+1 < N) Out[(row0+0)*N+col0+1] = acc[base+1] * s_r0 * s_c1;
        if (row0+1 < M && col0   < N) Out[(row0+1)*N+col0+0] = acc[base+2] * s_r1 * s_c0;
        if (row0+1 < M && col0+1 < N) Out[(row0+1)*N+col0+1] = acc[base+3] * s_r1 * s_c1;
      } // TN
    } // TM
  }

};

template <typename _T, int BM, int BN>
struct FragmentView {
  using T = _T;
  T* shared_ptr;

  __device__ inline FragmentView(T* smem) : shared_ptr(smem) {}

  __device__ inline T& operator()(int m, int n) {
    return shared_ptr[m * BN + n];
  }

  __device__ inline const T& operator()(int m, int n) const {
    return shared_ptr[m * BN + n];
  }
};

}  // namespace xpu
