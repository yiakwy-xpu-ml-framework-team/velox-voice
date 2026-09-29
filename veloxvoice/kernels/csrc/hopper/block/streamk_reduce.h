/* Copyright 2026 veloxVoice authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

// block/streamk_reduce.h — reusable cluster split-K reduction for one tile.
//
// Layout: each cluster contains all K slices of the same output tile
// (clusterDim.x = split_k, blockIdx.y = tile).  Every rank writes its bf16
// partial into local smem, publishes it through a cross-CTA mbarrier, and
// rank0 reduces the partials in smem/NoC and stores the tile once.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "../arch/tma/mbarrier_sm90.h"
#include "../arch/tma/tma_sm90.h"

namespace velox {
namespace hopper {
namespace streamk {

__device__ __forceinline__ uint32_t map_shared_cluster(uint32_t addr,
                                                       uint32_t rank) {
  uint32_t mapped;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(mapped)
               : "r"(addr), "r"(rank));
  return mapped;
}

template <typename T>
__device__ __forceinline__ T* map_shared_rank(T* ptr, uint32_t rank) {
  uint64_t generic;
  asm volatile(
      "{\n\t"
      ".reg .u64 ext;\n\t"
      "cvt.u64.u32 ext, %1;\n\t"
      "cvta.shared.u64 %0, ext;\n\t"
      "}"
      : "=l"(generic)
      : "r"(map_shared_cluster(smem_u32(ptr), rank)));
  return reinterpret_cast<T*>(generic);
}

__device__ __forceinline__ void arrive_remote(uint64_t* bar, uint32_t rank) {
  asm volatile(
      "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0], 1;" ::"r"(
          map_shared_cluster(smem_u32(bar), rank))
      : "memory");
}

// Write one [BM, BN] bf16 partial from a wgmma accumulator.  `get_mn` is the
// standard wgmma register-to-(row,col) mapping.
template <typename Acc, int BM, int BN, int WG_M>
__device__ __forceinline__ void store_partial(const Acc& acc, __nv_bfloat16* smem,
                                              int wg, int warp_in_wg, int lane) {
  constexpr int NUM_REGS = sizeof(acc.r) / sizeof(acc.r[0]);
  for (int i = 0; i < NUM_REGS; i += 2) {
    int row, col;
    acc.get_mn(i, warp_in_wg, lane, &row, &col);
    const int r = wg * WG_M + row;
    *reinterpret_cast<__nv_bfloat162*>(smem + r * BN + col) =
        __floats2bfloat162_rn(acc.r[i], acc.r[i + 1]);
  }
}


template <int BM, int BN, int THREADS>
__device__ __forceinline__ void reduce_store(
    __nv_bfloat16* partial, __nv_bfloat16* out, const __nv_bfloat16* bias,
    uint64_t* ready, uint64_t* done, int M, int block_m, int n0,
    int out_stride, int split_k, int rank, uint32_t phase) {
  constexpr int VEC = 2;
  const int tiles = (BM * BN) / VEC;

  if (rank != 0) {
    if (threadIdx.x == 0) arrive_remote(ready, 0);
    mbar_wait(done, phase);
    return;
  }

  if (split_k > 1) {
    mbar_wait(ready, phase);
  }

  for (int r = 1; r < split_k; ++r) {
    const __nv_bfloat16* remote = map_shared_rank(partial, r);
    for (int v = threadIdx.x; v < tiles; v += THREADS) {
      const int row = v / (BN / VEC);
      const int col = (v - row * (BN / VEC)) * VEC;
      const float2 a = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(partial + row * BN + col));
      const float2 b = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(remote + row * BN + col));
      *reinterpret_cast<__nv_bfloat162*>(partial + row * BN + col) =
          __floats2bfloat162_rn(a.x + b.x, a.y + b.y);
    }
  }

  for (int v = threadIdx.x; v < tiles; v += THREADS) {
    const int row = v / (BN / VEC);
    const int col = (v - row * (BN / VEC)) * VEC;
    const int gr = block_m + row;
    if (gr >= M) continue;
    const float2 x = __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162*>(partial + row * BN + col));
    const __nv_bfloat162 o = __floats2bfloat162_rn(
        x.x + __bfloat162float(bias[n0 + col]),
        x.y + __bfloat162float(bias[n0 + col + 1]));
    *reinterpret_cast<__nv_bfloat162*>(out + (long)gr * out_stride +
                                       n0 + col) = o;
  }

  if (threadIdx.x == 0) {
    for (int r = 1; r < split_k; ++r) arrive_remote(done, r);
  }
}

}  // namespace streamk
}  // namespace hopper
}  // namespace velox
