/* Copyright 2026 veloxVoice authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace velox {
namespace sheetsage {

using bf16 = __nv_bfloat16;

#define FULL_MASK 0xffffffffu

__device__ __forceinline__ float tof(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 tob(float v) { return __float2bfloat16(v); }


template <int BLOCK>
__device__ __forceinline__ float block_sum(float v, float* red) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, off);
  }

  constexpr int kW = BLOCK / 32;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  if (lane == 0) { 
    red[warp] = v; 
  }
  __syncthreads();

  // warp 0 reduces the kW partials; broadcast via shared
  if (warp == 0) {
    v = (lane < kW) ? red[lane] : 0.f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }

    if (lane == 0) { 
      red[0] = v; 
    }
  }
  __syncthreads();

  return red[0];
}


// NOTE (yiakwy) : using VAR = sum{sqare(x_i - mean)}, since 
// VAR = E[x^2]-E[x]^2 is not numerical stable.
template <int BLOCK, int kModel>
__device__ __forceinline__ void layernorm_frag(
    bf16* h, const bf16* __restrict__ w, const bf16* __restrict__ b,
    float eps, float* red) {
  float s = 0.f;
  for (int i = threadIdx.x; i < kModel; i += BLOCK) { 
    s += tof(h[i]); 
  }

  s = block_sum<BLOCK>(s, red);

  const float mean = s / kModel;
  float sq = 0.f;
  for (int i = threadIdx.x; i < kModel; i += BLOCK) {
    const float d = tof(h[i]) - mean;
    sq += d * d;
  }

  sq = block_sum<BLOCK>(sq, red);

  float var = sq / kModel;
  float rstd = rsqrtf(var + eps);
  for (int i = threadIdx.x; i < kModel; i += BLOCK) {
    h[i] = tob((tof(h[i]) - mean) * rstd * tof(w[i]) + tof(b[i]));
  }
  __syncthreads();
}

}  // namespace sheetsage
}  // namespace velox
