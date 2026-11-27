/* Copyright 2026 veloxVoice authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/
#pragma once

#include <cuda_bf16.h>

namespace velox {
namespace sheetsage {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float gemv_tof(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 gemv_tob(float v) { return __float2bfloat16(v); }

// Block-level: y[n] = W[n,:] . x + b[n] for all n < N.
template <int N, int K, int BLOCK>
__device__ __forceinline__ void gemv_block(const bf16* __restrict__ w,
                                           const bf16* __restrict__ b,
                                           const bf16* __restrict__ x,
                                           bf16* __restrict__ y) {
  constexpr int kW = BLOCK / 32;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int n = warp; n < N; n += kW) {
    const bf16* wrow = w + (long)n * K;
    float acc = (b != nullptr) ? gemv_tof(b[n]) : 0.f;
#pragma unroll 4
    for (int d = lane; d < K; d += 32) acc += gemv_tof(wrow[d]) * gemv_tof(x[d]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[n] = gemv_tob(acc);
  }
  __syncthreads();
}

// Grid-level: same math, n striped across ALL warps of the grid.
__device__ __forceinline__ void gemv_grid(const bf16* __restrict__ w,
                                          const bf16* __restrict__ b,
                                          const bf16* __restrict__ x,
                                          bf16* __restrict__ y, int N,
                                          int kModel) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const long gw = (long)blockIdx.x * (blockDim.x / 32) + warp;
  const long gs = (long)gridDim.x * (blockDim.x / 32);
  for (long n = gw; n < N; n += gs) {
    const bf16* wrow = w + n * kModel;
    float acc = (b != nullptr) ? gemv_tof(b[n]) : 0.f;
#pragma unroll 4
    for (int d = lane; d < kModel; d += 32)
      acc += gemv_tof(wrow[d]) * gemv_tof(x[d]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[n] = gemv_tob(acc);
  }
  __syncthreads();
}

}  // namespace sheetsage
}  // namespace velox
