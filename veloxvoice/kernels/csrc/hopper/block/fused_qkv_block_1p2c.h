/* Copyright 2026 veloxVoice authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

// block/fused_qkv_block_1p2c.h — persistent 1-producer / 2-consumer pipeline
// for the fused q/k/v projection on sm_90a:
//
//   out[M, N_OUT=1536] = x[M, K=512] @ Wqkv[N_OUT, 512]^T + b[N_OUT]
//
// The N-width WGMMA specialization is selected by
// hopper/fragment/fused_qkv_tile_bf16.h.  Only the production pipeline,
// cluster stream-K, and the opt-in phase trace live in this file.
#pragma once

#ifndef FQKV_STREAM_K
#define FQKV_STREAM_K 0
#endif

#ifndef FQKV_PHASE_TRACE
#define FQKV_PHASE_TRACE 0
#endif

#ifndef FQKV_GROUP_SIZE_M
#define FQKV_GROUP_SIZE_M 8
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../arch/wgmma/gmma_sm90.h"
#include "../arch/thread/pdl_sm90.h"
#include "../arch/thread/thread_barrier.h"
#include "../arch/tma/mbarrier_sm90.h"
#include "../arch/tma/tma_sm90.h"
#include "../arch/warpgroup/warpgroup_barrier.h"
#include "sched.h"
#include "wasp_producer.h"
#include "streamk_reduce.h"
#include "../fragment/fused_qkv_tile_bf16.h"

#if FQKV_STREAM_K
static_assert(FQKV_TILE_N == 256,
              "cluster stream-K epilogue is N256-only");
#endif

namespace velox {
namespace hopper {
namespace fused_qkv {

constexpr int BM = 128;
constexpr int BN = FQKV_TILE_N;
constexpr int BK = 64;

constexpr int N_OUT = 1536;

constexpr int K_BLOCK = 512;

constexpr int NSTAGES = FQKV_TILE_NSTAGES; // 3 or 4

constexpr int PRODUCER_THREADS = FQKV_TILE_PRODUCER_THREADS; // 128 or 256

constexpr int A_ELEMS = BM * BK;

constexpr int B_ELEMS = BN * BK;

constexpr int STAGE_ELEMS = A_ELEMS + B_ELEMS;

constexpr int STAGE_BYTES = STAGE_ELEMS * 2;

constexpr int CONSUMER_THREADS = 256;

constexpr int N_THREADS = CONSUMER_THREADS + PRODUCER_THREADS;

constexpr int EPI_ELEMS = BM * BN;

constexpr int EPI_STAGES = 1;

constexpr int EPI_BYTES = EPI_STAGES * EPI_ELEMS * 2;

constexpr int EPI_THREADS = CONSUMER_THREADS;

// TODO (yiakwy) : using vx::hopper::warpgroup_reg_dealloc
__device__ __forceinline__ void regs_producer() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
}

// TODO (yiakwy) : using vx::hopper::warpgroup_reg_alloc
__device__ __forceinline__ void regs_consumer() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
}

__device__ __forceinline__ void consumer_sync() {
  thread_barrier(7, EPI_THREADS);
}

__device__ __forceinline__ void consumer_wg_sync(int wg) {
  warpgroup_barrier(wg);
}

__device__ __forceinline__ uint32_t fqkv_pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ void fqkv_stmatrix_x4(
    __nv_bfloat16* dst, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
  asm volatile(
      "stmatrix.sync.aligned.x4.m8n8.shared::cta.b16 [%0], {%1, %2, %3, %4};" ::
          "r"(smem_u32(dst)), "r"(s0), "r"(s1), "r"(s2), "r"(s3));
}

// TODO (yiakwy) : move to arch/tracer/tracer.h
__device__ __forceinline__ unsigned long long fqkv_globaltime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void fqkv_fence_acc(float& reg) {
  asm volatile("" : "+f"(reg) :: "memory");
}

template <typename Acc>
__device__ __forceinline__ void fqkv_fence_acc_frag(Acc& acc) {
#pragma unroll
  for (int i = 0; i < sizeof(acc.r) / sizeof(acc.r[0]); ++i)
    fqkv_fence_acc(acc.r[i]);
}

__device__ __forceinline__ int fqkv_smid() {
  int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

// NOTE (yiakwy) : move to fagment, take reference to qkv_n256_bias
__device__ __forceinline__ float qkv_n192_bias(
    const WgmmaM64N192Frag& acc, const __nv_bfloat16* bias, int n0,
    int warp_in_wg, int lane, int index) {
  int row, col;
  acc.get_mn(index, warp_in_wg, lane, &row, &col);
  return acc.r[index] + __bfloat162float(bias[n0 + col]);
}


// NOTE (yiakwy) : move to fagment, take reference to qkv_epilogue_n256
__device__ __forceinline__ void qkv_epilogue_n192(
    const WgmmaM64N192Frag& acc, const __nv_bfloat16* bias,
    __nv_bfloat16* smem_d, const CUtensorMap* tmap_out, int block_m, int n0,
    int wg, int warp_in_wg, int lane) {

#define TILE_M 64
#define TILE_N 192

#define COL_STRIDE 8

#define STMATRIX_COLS 32

  int smem_offset =
      ((wg * 4 + warp_in_wg) * 16 + (lane & 15)) * STMATRIX_COLS + COL_STRIDE * (lane >> 4);

  int tma_offset = wg * TILE_M * STMATRIX_COLS;

#pragma unroll
  for (int j = 0; j < 6; ++j) {
    const int i0 = j * 2;
    const int i1 = i0 + 1;

    // cols 0..15
    fqkv_stmatrix_x4(
        smem_d + smem_offset,
        fqkv_pack_bf16x2(qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 0),
                         qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 1)),
        fqkv_pack_bf16x2(qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 2),
                         qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 3)),
        fqkv_pack_bf16x2(qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 4),
                         qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 5)),
        fqkv_pack_bf16x2(qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 6),
                         qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 7)));

    // cols 16..31
    fqkv_stmatrix_x4(
        smem_d + smem_offset + 16,
        fqkv_pack_bf16x2(qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 0),
                         qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 1)),
        fqkv_pack_bf16x2(qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 2),
                         qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 3)),
        fqkv_pack_bf16x2(qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 4),
                         qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 5)),
        fqkv_pack_bf16x2(qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 6),
                         qkv_n192_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 7)));

    smem_offset += BM * STMATRIX_COLS;

    tma_store_fence();
    consumer_wg_sync(wg + 8);

    if (warp_in_wg == 0 && lane == 0) {
      tma2d_store_async_bulk(tmap_out, smem_d + tma_offset, n0 + j * STMATRIX_COLS,
                             block_m + wg * TILE_M);
    }

    __syncwarp();

    tma_offset += BM * STMATRIX_COLS;
  } // repeat 6 times to cover 192 columns

  if (warp_in_wg == 0 && lane == 0) {
    tma_store_commit();
  }
}

// NOTE (yiakwy) : move to fagment
__device__ __forceinline__ float qkv_n256_bias(
    const WgmmaM64N256Frag& acc, const __nv_bfloat16* bias, int n0,
    int warp_in_wg, int lane, int index) {
  int row, col;
  acc.get_mn(index, warp_in_wg, lane, &row, &col);
  return acc.r[index] + __bfloat162float(bias[n0 + col]);
}

// NOTE (yiakwy) : move to fagment
__device__ __forceinline__ void qkv_epilogue_n256(
    const WgmmaM64N256Frag& acc, const __nv_bfloat16* bias,
    __nv_bfloat16* smem_d, const CUtensorMap* tmap_out, int block_m, int n0,
    int wg, int warp_in_wg, int lane) {

  // NOTE (yiakwy) : output 128x256 fp32 tile, each wg process 64x256 tile
  // NOTE (yiakwy) : see WgmmaM64N256Frag::get_mn (fragments/wgmma_accumulator_bf16.h) for the mapping of (thr_row, thr_col) to fragment index
#define TILE_M 64
#define TILE_N 256

#define WARP_GROUP_SIZE 4

#define ROWS_PER_WARP  16
#define THREADS_PER_ROW 4

#define ROW_STRIDE 8
#define COL_STRIDE 8

#define ROW_REPEATS 2
#define _VEC_SIZE 2

#define ELE_PER_THREAD ((ROW_REPEATS) * (_VEC_SIZE))

  // Register to fragemnt localtion mapping:
  //
  // const int thr_row_off = lane_id / WARP_GROUP_SIZE;
  // const int thr_col_off = lane_id % THREADS_PER_ROW; // ~ lane_id & 3
  // const int i = reg_idx / ELE_PER_THREAD;
  // const int sub_idx = reg_idx % ELE_PER_THREAD;
  //
  //            rows
  // warp 0 :  0..15
  // warp 1 : 16..31
  // warp 2 : 32..47
  // warp 3 : 48..63
  //
  // int local_row = warp_in_wg * ROWS_PER_WARP + thr_row_off + ROW_STRIDE * ( sub_idx / _VEC_SIZE);
  // int col = i * COL_STRIDE + thr_col_off * _VEC_SIZE + ( sub_idx % _VEC_SIZE);
  //
  // int row = local_row + wg * TILE_M;
  //
  // flash-float-jit-kernel uses msteps to extend the Fragment coverage

#define STMATRIX_COLS 32

  // stmatrix mapping (16 x 256 = 16 x STMATRIX_COLS x 8):
  //
  // smem_offset for stmatrix.4 (16x16 per warp) :
  //   - warp_in_frag : wg * WARP_GROUP_SIZE + warp_in_wg
  //   - thr_off_row : lane_id & 15 (lane_id % 16)
  //   - thr_off_col : lane_id >> 4 (lane_id / 16)
  int smem_offset =
      ((wg * 4 + warp_in_wg) * 16 + (lane & 15)) * STMATRIX_COLS + COL_STRIDE * (lane >> 4);

  int tma_offset = wg * TILE_M * STMATRIX_COLS;

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int i0 = j * 2;
    const int i1 = i0 + 1;

    // cols 0..15
    fqkv_stmatrix_x4(
        smem_d + smem_offset,
        fqkv_pack_bf16x2(qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 0),
                         qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 1)),
        fqkv_pack_bf16x2(qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 2),
                         qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 3)),
        fqkv_pack_bf16x2(qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 4),
                         qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 5)),
        fqkv_pack_bf16x2(qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 6),
                         qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i0 * 8 + 7)));

    // cols 16..31
    fqkv_stmatrix_x4(
        smem_d + smem_offset + 16,
        fqkv_pack_bf16x2(qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 0),
                         qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 1)),
        fqkv_pack_bf16x2(qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 2),
                         qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 3)),
        fqkv_pack_bf16x2(qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 4),
                         qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 5)),
        fqkv_pack_bf16x2(qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 6),
                         qkv_n256_bias(acc, bias, n0, warp_in_wg, lane, i1 * 8 + 7)));

    smem_offset += BM * STMATRIX_COLS;

    tma_store_fence();
    consumer_wg_sync(wg + 8);

    if (warp_in_wg == 0 && lane == 0) {
      tma2d_store_async_bulk(tmap_out, smem_d + tma_offset, n0 + j * STMATRIX_COLS,
                             block_m + wg * TILE_M);
    }

    __syncwarp();

    tma_offset += BM * STMATRIX_COLS;
  } // repeat 8 times to cover 256 columns

  if (warp_in_wg == 0 && lane == 0) {
    tma_store_commit();
  }
}


struct DefaultFqkvEpilogue {
  static __device__ __forceinline__ void store(
      const FqkvAccum& acc, const __nv_bfloat16* bias, __nv_bfloat16* stg,
      const CUtensorMap* tmap_out, int block_m, int n0, int M, int wg,
      int warp_in_wg, int lane) {
    tma_store_wait<0>();
    consumer_wg_sync(wg + 8);
    qkv_epilogue(acc, bias, stg, tmap_out, block_m, n0, wg, warp_in_wg, lane);
  }

 private:
  static __device__ __forceinline__ void qkv_epilogue(
      const WgmmaM64N192Frag& acc, const __nv_bfloat16* bias,
      __nv_bfloat16* smem_d, const CUtensorMap* tmap_out, int block_m, int n0,
      int wg, int warp_in_wg, int lane) {
    qkv_epilogue_n192(acc, bias, smem_d, tmap_out, block_m, n0, wg,
                      warp_in_wg, lane);
  }

  static __device__ __forceinline__ void qkv_epilogue(
      const WgmmaM64N256Frag& acc, const __nv_bfloat16* bias,
      __nv_bfloat16* smem_d, const CUtensorMap* tmap_out, int block_m, int n0,
      int wg, int warp_in_wg, int lane) {
    qkv_epilogue_n256(acc, bias, smem_d, tmap_out, block_m, n0, wg,
                      warp_in_wg, lane);
  }
};


template <typename Epilogue>
__device__ __forceinline__ void run_fused_qkv_persistent_splitk_pipeline_impl(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    int M, const CUtensorMap* tmap_x, const CUtensorMap* tmap_w,
    const CUtensorMap* tmap_out, uint64_t* trace, int split_k,
    int split_start) {
  extern __shared__ char smem_raw[];

  __nv_bfloat16* smem =
      reinterpret_cast<__nv_bfloat16*>((reinterpret_cast<uintptr_t>(smem_raw) +
                                        1023) & ~static_cast<uintptr_t>(1023));

  __nv_bfloat16* stages = smem;

  __nv_bfloat16* stg = smem + NSTAGES * STAGE_ELEMS;

  uint64_t* full = reinterpret_cast<uint64_t*>(stg + EPI_ELEMS);
  uint64_t* empty = full + NSTAGES;

  uint64_t* epi_ready = empty + NSTAGES;
  uint64_t* epi_free = epi_ready + EPI_STAGES;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wg = warp >> 2;
  const int warp_in_wg = warp & 3;

  const int k_steps = K_BLOCK / BK;

  const bool wg_leader = (warp_in_wg == 0) && (lane == 0);
  const bool is_producer = tid >= CONSUMER_THREADS;
  const bool producer_leader = tid == CONSUMER_THREADS;

  __shared__ __nv_bfloat16 s_bias[N_OUT];

  for (int i = tid; i < N_OUT; i += N_THREADS) {
    s_bias[i] = bias[i];
  }

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < NSTAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], 2);
    }
#pragma unroll
    for (int b = 0; b < EPI_STAGES; ++b) {
      if constexpr (FQKV_STREAM_K) {
        mbar_init(&epi_ready[b], split_k > 1 ? split_k - 1 : 1);
        mbar_init(&epi_free[b], 1);
      } else {
        mbar_init(&epi_ready[b], EPI_THREADS);
        mbar_init(&epi_free[b], 1);
      }
    }
  }

  if constexpr (FQKV_STREAM_K) {
    mbar_init_release_cluster();
    cluster_barrier();
  } else {
    __syncthreads();
  }

  pdl_wait();

  const int n_tiles = (N_OUT + BN - 1) / BN;
  const int m_tiles = (M + BM - 1) / BM;
  const int logical_tiles = m_tiles * n_tiles;
  const int total_tiles = FQKV_STREAM_K ? logical_tiles
                                        : logical_tiles * split_k;

  if (is_producer) {
    regs_producer();

#if FQKV_PHASE_TRACE
    const unsigned long long p_start = fqkv_globaltime();
#endif
    if (!producer_leader) return;
    WaspBf16Producer<NSTAGES, BM, BN, BK, 1>::State producer_state;

#if FQKV_PHASE_TRACE

    int p_ctr = 0;
    producer_state.trace =
        reinterpret_cast<unsigned long long*>(trace);
    producer_state.trace_cta = blockIdx.x;
    producer_state.trace_tile = p_ctr;

#endif // FQKV_PHASE_TRACE

    for (int tile = FQKV_STREAM_K ? blockIdx.y : blockIdx.x;
         tile < total_tiles;
         tile += FQKV_STREAM_K ? gridDim.y : gridDim.x) {
      int logical_tile = tile;
      int split_id = 0;

      if constexpr (FQKV_STREAM_K) {
        logical_tile = tile;
        split_id = blockIdx.x;
      } else {
        split_id = tile % split_k;
        logical_tile = tile / split_k;
      }

      int n_tile, m_tile;
      swizzle2d<FQKV_GROUP_SIZE_M>(logical_tile, m_tiles, n_tiles, &m_tile,
                                   &n_tile);

      int k_begin, k_end, k_stride;
      if constexpr (FQKV_STREAM_K) {
        const int k_per_slice = (k_steps + split_k - 1) / split_k;
        k_begin = split_id * k_per_slice;
        k_end = min(k_steps, k_begin + k_per_slice);
        k_stride = 1;
      } else {
        k_begin = split_id;
        k_end = k_steps;
        k_stride = split_k;
      }

#if FQKV_PHASE_TRACE
      producer_state.trace_tile = p_ctr;
#endif // FQKV_PHASE_TRACE

      WaspBf16Producer<NSTAGES, BM, BN, BK, 1>::load(
          m_tile * BM, n_tile * BN, 0, k_begin, k_end, k_stride, stages, full,
          empty, tmap_x, tmap_w, producer_state);

#if FQKV_PHASE_TRACE
      ++p_ctr;
#endif // FQKV_PHASE_TRACE
    }
    return;
  }

  // consumers
  regs_consumer();

  int gstep = 0;
  int empty_arrived = 0;
  int t_ctr = 0;
  for (int tile = FQKV_STREAM_K ? blockIdx.y : blockIdx.x;
       tile < total_tiles;
       tile += FQKV_STREAM_K ? gridDim.y : gridDim.x, ++t_ctr) {

    int logical_tile = tile;
    int split_id = 0;

    if constexpr (FQKV_STREAM_K) {
      logical_tile = tile;
      split_id = blockIdx.x;
    } else {
      split_id = tile % split_k;
      logical_tile = tile / split_k;
    }

    int n_tile, m_tile;
    swizzle2d<FQKV_GROUP_SIZE_M>(logical_tile, m_tiles, n_tiles, &m_tile,
                                 &n_tile);

    const int block_m = m_tile * BM;
    const int n0 = n_tile * BN;
    int k_begin, k_end, k_stride;

    if constexpr (FQKV_STREAM_K) {
      const int k_per_slice = (k_steps + split_k - 1) / split_k;
      k_begin = split_id * k_per_slice;
      k_end = min(k_steps, k_begin + k_per_slice);
      k_stride = 1;
    } else {
      k_begin = split_id;
      k_end = k_steps;
      k_stride = split_k;
    }

    const int k_iters = (k_end - k_begin + k_stride - 1) / k_stride;

    FqkvAccum acc;
    acc.zero();

#if FQKV_PHASE_TRACE
    const bool trace_on =
        trace != nullptr && blockIdx.x < FQKV_PHASE_TRACE_CTAS;
#endif //  FQKV_PHASE_TRACE

    for (int s = 0; s < k_iters; ++s, ++gstep) {
      const int stage = gstep % NSTAGES;
      const int use = gstep / NSTAGES;

#if FQKV_PHASE_TRACE
      const unsigned long long full0 = fqkv_globaltime();
#endif // FQKV_PHASE_TRACE
      mbar_wait(&full[stage], use & 1);

#if FQKV_PHASE_TRACE
      const unsigned long long full1 = fqkv_globaltime();
      const unsigned long long wg0 = fqkv_globaltime();
#endif // FQKV_PHASE_TRACE

      const __nv_bfloat16* base = stages + stage * STAGE_ELEMS;
      const __nv_bfloat16* sA = base + wg * 64 * BK;
      const __nv_bfloat16* sB = base + A_ELEMS;

      wgmma_fence();
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk) {
        const uint64_t dA = make_gmma_desc_b128(sA + kk * 16);
        const uint64_t dB = make_gmma_desc_b128(sB + kk * 16);
        acc.mma(dA, dB, 1u);
      }
      wgmma_commit();
      wgmma_wait<0>();

#if FQKV_PHASE_TRACE
      const unsigned long long wg1 = fqkv_globaltime();
      if (trace_on && tid == 0 && t_ctr < FQKV_PHASE_TRACE_TILES &&
          s < FQKV_PHASE_TRACE_K) {
        unsigned long long* tr =
            reinterpret_cast<unsigned long long*>(trace) +
            blockIdx.x * FQKV_PHASE_TRACE_STRIDE +
            (t_ctr * FQKV_PHASE_TRACE_K + s) * 8;
        tr[4] = full0;
        tr[5] = full1;
        tr[6] = wg0;
        tr[7] = wg1;
      }
#endif // FQKV_PHASE_TRACE

      if (wg_leader) mbar_arrive(&empty[empty_arrived % NSTAGES]);
      ++empty_arrived;
    }

    wgmma_wait<0>();

#if FQKV_PHASE_TRACE
    const unsigned long long e_start_t = fqkv_globaltime();
    const unsigned long long prev_store_done_t = e_start_t;
#endif // FQKV_PHASE_TRACE

    while (empty_arrived < gstep) {
      if (wg_leader) mbar_arrive(&empty[empty_arrived % NSTAGES]);
      ++empty_arrived;
    }

#if FQKV_STREAM_K

    streamk::store_partial<WgmmaM64N256Frag, BM, BN, 64>(
        acc, stg, wg, warp_in_wg, lane);
    consumer_sync();
    tma_store_fence();
    streamk::reduce_store<BM, BN, EPI_THREADS>(
        stg, out, s_bias, &epi_ready[0], &epi_free[0], M, block_m, n0, N_OUT,
        split_k, blockIdx.x, t_ctr & 1);
    consumer_sync();

#else

    // NOTE (yiakwy) : optimized with stmatrix
    Epilogue::store(acc, s_bias, stg, tmap_out, block_m, n0, M, wg,
                    warp_in_wg, lane);

#endif // FQKV_STREAM_K

#if FQKV_PHASE_TRACE
    const unsigned long long e_end_t = fqkv_globaltime();
    if (trace_on && tid == 0 && t_ctr < FQKV_PHASE_TRACE_TILES) {
      unsigned long long* tr =
          reinterpret_cast<unsigned long long*>(trace) +
          blockIdx.x * FQKV_PHASE_TRACE_STRIDE +
          FQKV_PHASE_TRACE_KSTEP_EVENTS + t_ctr * 4;
      tr[0] = e_start_t;
      tr[1] = prev_store_done_t;
      tr[2] = e_end_t;
      if (t_ctr == 0) tr[3] = static_cast<unsigned long long>(fqkv_smid());
    }
#endif // FQKV_PHASE_TRACE
  }
}

__device__ __forceinline__ void run_fused_qkv_persistent_splitk_pipeline(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    int M, const CUtensorMap* tmap_x, const CUtensorMap* tmap_w,
    const CUtensorMap* tmap_out, uint64_t* trace, int split_k,
    int split_start) {
  run_fused_qkv_persistent_splitk_pipeline_impl<DefaultFqkvEpilogue>(x, w, bias, out, M, tmap_x, tmap_w,
                                          tmap_out, trace, split_k,
                                          split_start);
}

}  // namespace fused_qkv
}  // namespace hopper
}  // namespace velox
