/* Copyright 2026 veloxVoice authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../arch/tma/mbarrier_sm90.h"
#include "../arch/tma/tma_sm90.h"

#ifndef FQKV_PHASE_TRACE
#define FQKV_PHASE_TRACE 0
#endif

namespace velox {
namespace hopper {

#if FQKV_PHASE_TRACE
constexpr int FQKV_PHASE_TRACE_CTAS = 8;
constexpr int FQKV_PHASE_TRACE_TILES = 8;
constexpr int FQKV_PHASE_TRACE_K = 8;
constexpr int FQKV_PHASE_TRACE_KSTEP_EVENTS =
    FQKV_PHASE_TRACE_TILES * FQKV_PHASE_TRACE_K * 8;
constexpr int FQKV_PHASE_TRACE_STRIDE =
    FQKV_PHASE_TRACE_KSTEP_EVENTS + FQKV_PHASE_TRACE_TILES * 4;

__device__ __forceinline__ unsigned long long wasp_globaltime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#endif

template <int STAGES, int BM, int BN, int BK, int NB_TILES>
struct WaspBf16Producer {
  struct State {
    int gstep = 0;
#if FQKV_PHASE_TRACE
    unsigned long long* trace = nullptr;
    int trace_cta = -1;
    int trace_tile = -1;
#endif
  };

  static __device__ __forceinline__ void load_once(
      int k, int block_m, int n0, int b_n_stride, __nv_bfloat16* stages,
      uint64_t* full, const CUtensorMap* tmap_x, const CUtensorMap* tmap_w,
      int stage) {
    __nv_bfloat16* base = stages + stage * (BM + NB_TILES * BN) * BK;
    tma_expect_bytes(full + stage,
                     (BM * BK + NB_TILES * BN * BK) * sizeof(__nv_bfloat16));
#if FQKV_TMA_HINT
    constexpr uint64_t kCacheHint = 0x1000000000000000ull;
    tma2d_load_async_hint(base, tmap_x, full + stage, k * BK, block_m,
                          kCacheHint);
#pragma unroll
    for (int b = 0; b < NB_TILES; ++b) {
      tma2d_load_async_hint(base + BM * BK + b * BN * BK, tmap_w,
                            full + stage, k * BK, n0 + b * b_n_stride,
                            kCacheHint);
    }
#else
    tma2d_load_async(base, tmap_x, full + stage, k * BK, block_m);
#pragma unroll
    for (int b = 0; b < NB_TILES; ++b) {
      tma2d_load_async(base + BM * BK + b * BN * BK, tmap_w, full + stage,
                       k * BK, n0 + b * b_n_stride);
    }
#endif
  }

  static __device__ __forceinline__ void load_once_multicast2(
      int k, int block_m, int n0, int b_n_stride, __nv_bfloat16* stages,
      uint64_t* full, const CUtensorMap* tmap_x, const CUtensorMap* tmap_w,
      int stage, int cluster_rank) {
    __nv_bfloat16* base = stages + stage * (BM + NB_TILES * BN) * BK;
    tma_expect_bytes(full + stage,
                     (BM * BK + NB_TILES * BN * BK) * sizeof(__nv_bfloat16));
    tma2d_load_async(base, tmap_x, full + stage, k * BK, block_m);
#pragma unroll
    for (int b = 0; b < NB_TILES; ++b) {
      if (cluster_rank == 0)
        tma2d_load_async_multicast2(base + BM * BK + b * BN * BK, tmap_w,
                                    full + stage, k * BK, n0 + b * b_n_stride);
    }
  }

  static __device__ __forceinline__ void load_multicast2(
      int block_m, int n0, int b_n_stride, int k_begin, int k_end, int k_stride,
      __nv_bfloat16* stages, uint64_t* full, uint64_t* empty,
      const CUtensorMap* tmap_x, const CUtensorMap* tmap_w, State& state,
      int cluster_rank) {
    for (int k = k_begin; k < k_end; k += k_stride) {
      const int stage = state.gstep % STAGES;
      const int use = state.gstep / STAGES;
      if (use > 0) mbar_wait(empty + stage, (use - 1) & 1);
      load_once_multicast2(k, block_m, n0, b_n_stride, stages, full, tmap_x,
                           tmap_w, stage, cluster_rank);
      ++state.gstep;
    }
  }

  static __device__ __forceinline__ void load(
      int block_m, int n0, int b_n_stride, int k_begin, int k_end, int k_stride,
      __nv_bfloat16* stages, uint64_t* full, uint64_t* empty,
      const CUtensorMap* tmap_x, const CUtensorMap* tmap_w, State& state) {
#if FQKV_PHASE_TRACE
    const bool trace_on =
        state.trace != nullptr && state.trace_cta >= 0 &&
        state.trace_cta < FQKV_PHASE_TRACE_CTAS &&
        state.trace_tile >= 0 && state.trace_tile < FQKV_PHASE_TRACE_TILES;
    int k_idx = 0;
#endif
    for (int k = k_begin; k < k_end; k += k_stride) {
      const int stage = state.gstep % STAGES;
      const int use = state.gstep / STAGES;
#if FQKV_PHASE_TRACE
      const unsigned long long wait0 = wasp_globaltime();
#endif
      if (use > 0) mbar_wait(empty + stage, (use - 1) & 1);
#if FQKV_PHASE_TRACE
      const unsigned long long wait1 = wasp_globaltime();
      const unsigned long long issue0 = wasp_globaltime();
#endif
      load_once(k, block_m, n0, b_n_stride, stages, full, tmap_x, tmap_w,
                stage);
#if FQKV_PHASE_TRACE
      const unsigned long long issue1 = wasp_globaltime();
      if (trace_on && state.trace_tile < FQKV_PHASE_TRACE_TILES &&
          k_idx < FQKV_PHASE_TRACE_K) {
        unsigned long long* tr = state.trace +
            state.trace_cta * FQKV_PHASE_TRACE_STRIDE +
            (state.trace_tile * FQKV_PHASE_TRACE_K + k_idx) * 8;
        tr[0] = wait0;
        tr[1] = wait1;
        tr[2] = issue0;
        tr[3] = issue1;
      }
      ++k_idx;
#endif
      ++state.gstep;
    }
  }
};

}  // namespace hopper
}  // namespace velox
