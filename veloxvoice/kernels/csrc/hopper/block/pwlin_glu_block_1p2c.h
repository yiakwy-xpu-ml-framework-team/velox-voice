/* Copyright 2026 veloxVoice authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../arch/wgmma/gmma_sm90.h"
#include "../arch/thread/pdl_sm90.h"
#include "../arch/tma/mbarrier_sm90.h"
#include "../arch/tma/tma_sm90.h"
#include "sched.h"
#include "wasp_producer.h"
#include "../fragment/wgmma_accumulator_bf16.h"

namespace velox {
namespace hopper {
namespace pwlin_glu {

#ifndef PWLIN_TILE_M
#define PWLIN_TILE_M 128
#endif

#ifndef PWLIN_TILE_N
#define PWLIN_TILE_N 128
#endif

constexpr int BM = PWLIN_TILE_M; // rows per CTA tile
constexpr int BN = PWLIN_TILE_N; // columns per GLU half
constexpr int N_FRAGS = BN / 64; // WGMMA m64n64 fragments per GLU half
constexpr int BK = 64;

#ifndef PWLIN_NSTAGES
#define PWLIN_NSTAGES 4
#endif

constexpr int NSTAGES = PWLIN_NSTAGES;
constexpr int N_HALF = 512;  // GLU half width
constexpr int K_BLOCK = 512; // fixed K for the conformer projection

constexpr int A_ELEMS = BM * BK;
constexpr int B_ELEMS = BN * BK;
constexpr int STAGE_ELEMS = A_ELEMS + 2 * B_ELEMS;
constexpr int STAGE_BYTES = STAGE_ELEMS * 2;

constexpr int CONSUMER_WGS = BM / 64;
constexpr int CONSUMER_THREADS = CONSUMER_WGS * 128;

constexpr int N_THREADS = CONSUMER_THREADS + 32;

// TODO (yiakwy) : optimze epilogue of out = (a + b_lo) * sigmoid(b + b_hi).
__device__ __forceinline__ void glu_epilogue(
    const WgmmaM64N64Frag* accA, const WgmmaM64N64Frag* accB,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out, int block_m, int n0, int M, int wg,
    int warp_in_wg, int lane) {

  const int m_base = block_m + wg * 64;
  const int group = lane >> 2;
  const int tig = lane & 3;

#pragma unroll
  for (int nt = 0; nt < N_FRAGS; ++nt) {
    const WgmmaM64N64Frag& a_frag = accA[nt];
    const WgmmaM64N64Frag& b_frag = accB[nt];

#pragma unroll
    for (int c = 0; c < 8; ++c) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int i = c * 4 + h * 2;
        const int row = m_base + warp_in_wg * 16 + group + 8 * h;
        if (row >= M) continue;
        const int col0 = n0 + nt * 64 + c * 8 + tig * 2;
        const float a0 = a_frag.r[i] + __bfloat162float(bias[col0]);
        const float a1 = a_frag.r[i + 1] + __bfloat162float(bias[col0 + 1]);
        const float b0 = b_frag.r[i] + __bfloat162float(bias[N_HALF + col0]);
        const float b1 =
            b_frag.r[i + 1] + __bfloat162float(bias[N_HALF + col0 + 1]);

        __nv_bfloat162 o;
        o.x = __float2bfloat16(a0 / (1.f + __expf(-b0)));
        o.y = __float2bfloat16(a1 / (1.f + __expf(-b1)));

        *reinterpret_cast<__nv_bfloat162*>(out + (long)row * N_HALF + col0) = o;
      } // for-h
    }   // for-c
  }     // for-nt
}

__device__ __forceinline__ void run_pwlin_glu(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    int M, const CUtensorMap* tmap_x, const CUtensorMap* tmap_w) {
  extern __shared__ char smem_raw[];
  __nv_bfloat16* smem =
      reinterpret_cast<__nv_bfloat16*>((reinterpret_cast<uintptr_t>(smem_raw) +
                                        1023) & ~static_cast<uintptr_t>(1023));
  __nv_bfloat16* stages = smem;
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + NSTAGES * STAGE_ELEMS);
  uint64_t* empty = full + NSTAGES;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wg = warp >> 2;             // 0,1 consumer warpgroups
  const int warp_in_wg = warp & 3;
  const int k_steps = K_BLOCK / BK;
  const bool wg_leader = (warp_in_wg == 0) && (lane == 0);
  const bool is_producer = tid >= CONSUMER_THREADS;
  const bool producer_leader = tid == CONSUMER_THREADS;

  __shared__ __nv_bfloat16 s_bias[2 * N_HALF];
  for (int i = tid; i < 2 * N_HALF; i += N_THREADS) s_bias[i] = bias[i];

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < NSTAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], CONSUMER_WGS);
    }
  }
  __syncthreads();

  pdl_wait();

  const int n_tiles = N_HALF / BN;
  const int m_tiles = (M + BM - 1) / BM;
  const int total_tiles = m_tiles * n_tiles;

  if (is_producer) {
    if (!producer_leader) return;
    WaspBf16Producer<NSTAGES, BM, BN, BK, 2>::State producer_state;
    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
      const int n_tile = tile / m_tiles;
      const int m_tile = tile % m_tiles;
      const int block_m = m_tile * BM;
      const int n0 = n_tile * BN;
      WaspBf16Producer<NSTAGES, BM, BN, BK, 2>::load(
          block_m, n0, N_HALF, 0, k_steps, 1, stages, full, empty, tmap_x,
          tmap_w, producer_state);
    }
    return;
  }

  // consumers
  int gstep = 0;
  int empty_arrived = 0;

  for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    const int n_tile = tile / m_tiles;
    const int m_tile = tile % m_tiles;
    const int block_m = m_tile * BM;

    const int n0 = n_tile * BN;

    WgmmaM64N64Frag accA[N_FRAGS], accB[N_FRAGS];
#pragma unroll
    for (int nt = 0; nt < N_FRAGS; ++nt) {
      accA[nt].zero();
      accB[nt].zero();
    }

    for (int s = 0; s < k_steps; ++s, ++gstep) {
      const int stage = gstep % NSTAGES;
      const int use = gstep / NSTAGES;

      mbar_wait(&full[stage], use & 1);

      const __nv_bfloat16* base = stages + stage * STAGE_ELEMS;
      const __nv_bfloat16* sA = base + wg * 64 * BK;
      const __nv_bfloat16* sB0 = base + A_ELEMS;
      const __nv_bfloat16* sB1 = base + A_ELEMS + B_ELEMS;

      wgmma_fence();
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk) {
        const uint64_t dA = make_gmma_desc_b128(sA + kk * 16);
#pragma unroll
        for (int nt = 0; nt < N_FRAGS; ++nt) {
          const uint64_t dB0 = make_gmma_desc_b128(sB0 + nt * 64 * BK + kk * 16);
          const uint64_t dB1 = make_gmma_desc_b128(sB1 + nt * 64 * BK + kk * 16);
          accA[nt].mma(dA, dB0, 1u);
          accB[nt].mma(dA, dB1, 1u);
        }
      }
      wgmma_commit();

      if (gstep - empty_arrived >= NSTAGES - 1) {
        wgmma_wait<NSTAGES - 1>();
        if (wg_leader) mbar_arrive(&empty[empty_arrived % NSTAGES]);
        ++empty_arrived;
      }
    }

    wgmma_wait<0>();
    while (empty_arrived < gstep) {
      if (wg_leader) mbar_arrive(&empty[empty_arrived % NSTAGES]);
      ++empty_arrived;
    }

    glu_epilogue(accA, accB, s_bias, out, block_m, n0, M, wg,
                 warp_in_wg, lane);
  }
}

}  // namespace pwlin_glu
}  // namespace hopper
}  // namespace velox
