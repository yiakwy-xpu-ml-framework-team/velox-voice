/* Copyright 2026 veloxVoice authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/
#pragma once

#include <cuda_bf16.h>

namespace velox {
namespace sheetsage {

using bf16 = __nv_bfloat16;

#define SS_NEG_INF (-3.402823466e38f)

__device__ __forceinline__ float attn_tof(bf16 v) { return __bfloat162float(v); }

// Online-softmax attention partials for a single query vs KV [H, R, D].
// Global (head, row) pairs striped across (block, warp) workers.
// Per-warp per-head state in shared memory; block merge → pm2/pacc2.
template <int BLOCK, int kHeads, int kD>
__device__ __forceinline__ void attn_stage_frag(
    const bf16* __restrict__ q, const bf16* __restrict__ K,
    const bf16* __restrict__ V, int R, int rows_valid, float scale,
    int gbid, float* sm_m, float* sm_l, float* sm_acc,
    float* __restrict__ pm2, float* __restrict__ pacc2) {
  constexpr int kW = BLOCK / 32;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const long npairs = (long)kHeads * rows_valid;
  const long stride = (long)gridDim.x * kW;
  const long me = (long)gbid * kW + warp;

  for (int wh = threadIdx.x; wh < kW * kHeads; wh += BLOCK) {
    sm_m[wh] = SS_NEG_INF;
    sm_l[wh] = 0.f;
  }
  for (int i = threadIdx.x; i < kW * kHeads * kD; i += BLOCK) sm_acc[i] = 0.f;
  __syncthreads();

  for (long hj = me; hj < npairs; hj += stride) {
    int head = (int)(hj / rows_valid);
    int j = (int)(hj % rows_valid);
    const bf16* qrow = q + head * kD;
    const bf16* krow = K + ((long)head * R + j) * kD;
    float s = 0.f;
#pragma unroll
    for (int d = lane; d < kD; d += 32) s += attn_tof(krow[d]) * attn_tof(qrow[d]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      s += __shfl_down_sync(0xffffffffu, s, off);
    s = __shfl_sync(0xffffffffu, s, 0) * scale;

    float mw = sm_m[warp * kHeads + head];
    float rescale = 1.f;
    if (s > mw) {
      rescale = __expf(mw - s);
      sm_m[warp * kHeads + head] = s;
      mw = s;
    }
    float pw = __expf(s - mw);
#pragma unroll
    for (int d = lane; d < kD; d += 32)
      sm_acc[(warp * kHeads + head) * kD + d] =
          sm_acc[(warp * kHeads + head) * kD + d] * rescale +
          pw * attn_tof(V[((long)head * R + j) * kD + d]);
    if (lane == 0)
      sm_l[warp * kHeads + head] = sm_l[warp * kHeads + head] * rescale + pw;
  }
  __syncthreads();

  // per-head block merge → one partial per (block, head)
  for (int head = 0; head < kHeads; ++head) {
    float mstar = SS_NEG_INF;
    for (int w = 0; w < kW; ++w) mstar = fmaxf(mstar, sm_m[w * kHeads + head]);
    for (int d = threadIdx.x; d < kD; d += BLOCK) {
      float num = 0.f, den = 0.f;
      for (int w = 0; w < kW; ++w) {
        float mw = sm_m[w * kHeads + head];
        if (mw == SS_NEG_INF) continue;
        float r = __expf(mw - mstar);
        den += sm_l[w * kHeads + head] * r;
        num += sm_acc[(w * kHeads + head) * kD + d] * r;
      }
      if (threadIdx.x == 0) {
        float* mp2 = pm2 + ((long)gbid * kHeads + head) * 2;
        mp2[0] = mstar;
        mp2[1] = den;
      }
      pacc2[((long)gbid * kHeads + head) * kD + d] = num / fmaxf(den, 1e-30f);
    }
    __syncthreads();
  }
}

// Merge per-block partials → attn_out[h, d] (bf16, normalised).
// Grid-wide: block b owns hd = b*512 + tid (only blocks with hd < H*D work).
__device__ __forceinline__ void attn_merge_frag(
    const float* __restrict__ pm2, const float* __restrict__ pacc2,
    int nblocks, int kHeads, int kD, bf16* __restrict__ attn_out) {
  for (int hd = blockIdx.x * blockDim.x + threadIdx.x; hd < kHeads * kD;
       hd += gridDim.x * blockDim.x) {
    int head = hd / kD, d = hd % kD;
    float mstar = SS_NEG_INF;
    for (int g = 0; g < nblocks; ++g)
      mstar = fmaxf(mstar, pm2[((long)g * kHeads + head) * 2]);
    float num = 0.f, den = 0.f;
    for (int g = 0; g < nblocks; ++g) {
      const long idx = (long)g * kHeads + head;
      float m = pm2[idx * 2];
      if (m == SS_NEG_INF) continue;
      float r = __expf(m - mstar);
      den += pm2[idx * 2 + 1] * r;
      num += pacc2[idx * kD + d] * r;
    }
    attn_out[hd] = attn_tof(num / fmaxf(den, 1e-30f));
  }
  __syncthreads();
}

}  // namespace sheetsage
}  // namespace velox
