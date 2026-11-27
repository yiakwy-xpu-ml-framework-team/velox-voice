/* Copyright 2026 veloxVoice authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/
#pragma once

#ifndef SHEETSAGE_AR_MAX_STEPS
#define SHEETSAGE_AR_MAX_STEPS 5120
#endif

#ifndef SHEETSAGE_AR_VOCAB
#define SHEETSAGE_AR_VOCAB 31678
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../arch/thread/pdl_sm90.h"
#include "../arch/thread/thread_barrier.h"
#include "../arch/warpgroup/warpgroup_barrier.h"
#include "../fragment/sheetsage_ln.h"
#include "../fragment/sheetsage_gemv.h"
#include "../fragment/sheetsage_attn.h"

namespace velox {
namespace hopper {
namespace sheetsage_ar {

using namespace ::velox::sheetsage;

using bf16 = __nv_bfloat16;

// SheetSage2 decoder dims
constexpr int kLayers = 6;
constexpr int kHeads = 8;
constexpr int kD = 64;
constexpr int kModel = 512;          // d_model
constexpr int kFFN = 2048;           // d_ffn
constexpr int kQKV = 3 * kModel;     // packed qkv output
constexpr int kVocab = SHEETSAGE_AR_VOCAB;
constexpr int kMaxSteps = SHEETSAGE_AR_MAX_STEPS;
constexpr int kThreads = 512;
constexpr int kWarp = kThreads / 32;  // 16 warps

constexpr int kComputeWarps = kThreads / 32;         // 16
constexpr int kComputeThreads = kComputeWarps * 32;  // 512


// Grammar state encoding
//
// State fields: (can_end, allow_shift, incomplete ∈ {0,1,2}, last_field ∈ 0..5)
// Total: 2 × 2 × 3 × 6 = 72 states
constexpr int kNumGrammarStates = 128;
constexpr int kGrammarTableSize = kNumGrammarStates * kVocab;

constexpr int kSmemH = 0;
constexpr int kSmemQkv = kSmemH + kModel;
constexpr int kSmemAttn = kSmemQkv + kQKV;
constexpr int kSmemTmp = kSmemAttn + kModel;
constexpr int kSmemTotal = kSmemTmp + kModel;  // 5120 bf16 = 10 KB

__device__ __forceinline__ float ar_tof(bf16 v) { return __bfloat162float(v); }

// nanosecond wall clock (Hopper %globaltimer, ns since device boot) — used
// by the phase-trace instrumentation (benchmark/visualize_gemm_pipelining).
__device__ __forceinline__ long long ar_gtimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ bf16 ar_tob(float v) { return __float2bfloat16(v); }

// warp-level reduction for a single fp32 value
__device__ __forceinline__ float ar_warp_reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ float ar_warp_reduce_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));

  // broadcast
  return __shfl_sync(0xffffffffu, v, 0);
}


// SIMT gemv
template <int BLOCK>
__global__ void ss_gemv_kernel(const bf16* __restrict__ w,
                               const bf16* __restrict__ b,
                               const bf16* __restrict__ x,
                               bf16* __restrict__ y, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5, nwarp = BLOCK >> 5;

  const int nv = K / 8;
  const uint4* __restrict__ xv = reinterpret_cast<const uint4*>(x);

  for (int r = blockIdx.x * nwarp + warp; r < N; r += gridDim.x * nwarp) {
    const uint4* __restrict__ wv =
        reinterpret_cast<const uint4*>(w + (long)r * K);
    
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;

#pragma unroll 4
    for (int i = lane; i < nv; i += 32) {
      uint4 q = wv[i], xq = xv[i];
      const bf16* h = reinterpret_cast<const bf16*>(&q);
      const bf16* hx = reinterpret_cast<const bf16*>(&xq);
      a0 += ar_tof(h[0]) * ar_tof(hx[0]);
      a1 += ar_tof(h[1]) * ar_tof(hx[1]);
      a2 += ar_tof(h[2]) * ar_tof(hx[2]);
      a3 += ar_tof(h[3]) * ar_tof(hx[3]);
      a0 += ar_tof(h[4]) * ar_tof(hx[4]);
      a1 += ar_tof(h[5]) * ar_tof(hx[5]);
      a2 += ar_tof(h[6]) * ar_tof(hx[6]);
      a3 += ar_tof(h[7]) * ar_tof(hx[7]);
    }

    float acc = (a0 + a1) + (a2 + a3);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      acc += __shfl_down_sync(~0u, acc, o);
    }

    if (lane == 0) { 
      y[r] = ar_tob(acc + (b ? ar_tof(b[r]) : 0.f)); 
    }
  } // r
}


template <int BLOCK, int K>
__global__ void ss_ln_res_kernel(bf16* __restrict__ h,
                                 const bf16* __restrict__ res,
                                 const bf16* __restrict__ w,
                                 const bf16* __restrict__ b,
                                 float eps) {
  __shared__ float sred[BLOCK / 32];
  if (res != nullptr)
    for (int i = threadIdx.x; i < K; i += BLOCK) {
      h[i] = ar_tob(ar_tof(h[i]) + ar_tof(res[i]));
    }
  __syncthreads();
  velox::sheetsage::layernorm_frag<BLOCK, K>(h, w, b, eps, sred);
}


template <int BLOCK>
__global__ void ss_self_attn_kernel(const bf16* __restrict__ q,   // [H,D]
                                    const bf16* __restrict__ K,   // [H,S,D]
                                    const bf16* __restrict__ V,
                                    bf16* __restrict__ out,       // [H,D]
                                    int S, const int* __restrict__ rows_dev,
                                    float scale) {
  // 1 block per head
  const int h = blockIdx.x;
  const int rows_valid = *rows_dev + 1;

  const int lane = threadIdx.x & 31;

  const bf16* qr = q + (long)h * kD;
  const bf16* Kh = K + (long)h * S * kD;
  const bf16* Vh = V + (long)h * S * kD;

  uint4 qreg[kD / 8];

#pragma unroll
  for (int u = 0; u < kD / 8; ++u) {
    qreg[u] = reinterpret_cast<const uint4*>(qr)[u];
  }

  float m = SS_NEG_INF, l = 0.f;
  float acc[kD];

#pragma unroll
  for (int d = 0; d < kD; ++d) { 
    acc[d] = 0.f;
  }

  for (int j = lane; j < rows_valid; j += 32) {
    const uint4* kv = reinterpret_cast<const uint4*>(Kh + (long)j * kD);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;

#pragma unroll
    for (int u = 0; u < kD / 8; ++u) {
      uint4 q = kv[u];
      const bf16* hh = reinterpret_cast<const bf16*>(&q);
      const bf16* hq = reinterpret_cast<const bf16*>(&qreg[u]);
      a0 += ar_tof(hh[0]) * ar_tof(hq[0]);
      a1 += ar_tof(hh[1]) * ar_tof(hq[1]);
      a2 += ar_tof(hh[2]) * ar_tof(hq[2]);
      a3 += ar_tof(hh[3]) * ar_tof(hq[3]);
      a0 += ar_tof(hh[4]) * ar_tof(hq[4]);
      a1 += ar_tof(hh[5]) * ar_tof(hq[5]);
      a2 += ar_tof(hh[6]) * ar_tof(hq[6]);
      a3 += ar_tof(hh[7]) * ar_tof(hq[7]);
    }

    float s = ((a0 + a1) + (a2 + a3)) * scale;
    if (s > m) {
      float r = __expf(m - s);
      l *= r;
#pragma unroll
      for (int d = 0; d < kD; ++d) { 
        acc[d] *= r;
      }
      m = s;
    }

    float pp = __expf(s - m);
    l += pp;
    const uint4* vv = reinterpret_cast<const uint4*>(Vh + (long)j * kD);

#pragma unroll
    for (int u = 0; u < kD / 8; ++u) {
      uint4 q = vv[u];
      const bf16* hh = reinterpret_cast<const bf16*>(&q);
      acc[u * 8 + 0] += pp * ar_tof(hh[0]);
      acc[u * 8 + 1] += pp * ar_tof(hh[1]);
      acc[u * 8 + 2] += pp * ar_tof(hh[2]);
      acc[u * 8 + 3] += pp * ar_tof(hh[3]);
      acc[u * 8 + 4] += pp * ar_tof(hh[4]);
      acc[u * 8 + 5] += pp * ar_tof(hh[5]);
      acc[u * 8 + 6] += pp * ar_tof(hh[6]);
      acc[u * 8 + 7] += pp * ar_tof(hh[7]);
    }
  } // j

  float mb = ar_warp_reduce_max(m);
  float rs = __expf(m - mb);
  float lf = ar_warp_reduce_sum(l * rs);  // sum over lanes
  for (int d = 0; d < kD; ++d) {
    float num = ar_warp_reduce_sum(acc[d] * rs);
    if (lane == 0) {
      out[(long)h * kD + d] = ar_tob(num / fmaxf(lf, 1e-30f));
    }
  } // d
}


template <int BLOCK>
__global__ void ss_kvstore_kernel(const bf16* __restrict__ qkv,  // [3D]
                                  bf16* __restrict__ K,          // [H,S,D]
                                  bf16* __restrict__ V,
                                  int S, const int* __restrict__ pos_dev) {
  int i = blockIdx.x * BLOCK + threadIdx.x;

  if (i >= kModel) return;

  int pos = *pos_dev;
  int h = i / kD, d = i % kD;

  K[(long)h * S * kD + (long)pos * kD + d] = qkv[kModel + i];
  V[(long)h * S * kD + (long)pos * kD + d] = qkv[2 * kModel + i];
}


template <int BLOCK>
__global__ void ss_lmhead_kernel(const bf16* __restrict__ lm,   // [V,K]
                                 const bf16* __restrict__ h,    // [K]
                                 const unsigned char* __restrict__ allowed,
                                 float* __restrict__ pbest,      // [gridDim]
                                 int* __restrict__ pidx,
                                 int V, int K) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5, nwarp = BLOCK >> 5;

  const int nv = K / 8;
  const uint4* __restrict__ xv = reinterpret_cast<const uint4*>(h);

  float best = SS_NEG_INF;
  int bi = 0;
  const int chunk = (V + gridDim.x - 1) / gridDim.x;
  const int v0 = blockIdx.x * chunk, v1 = min(V, v0 + chunk);
  for (int r = v0 + warp; r < v1; r += nwarp) {

    if (!allowed[r]) continue;

    const uint4* __restrict__ wv =
        reinterpret_cast<const uint4*>(lm + (long)r * K);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;

#pragma unroll 4
    for (int i = lane; i < nv; i += 32) {
      uint4 q = wv[i], xq = xv[i];
      const bf16* hh = reinterpret_cast<const bf16*>(&q);
      const bf16* hx = reinterpret_cast<const bf16*>(&xq);
      a0 += ar_tof(hh[0]) * ar_tof(hx[0]);
      a1 += ar_tof(hh[1]) * ar_tof(hx[1]);
      a2 += ar_tof(hh[2]) * ar_tof(hx[2]);
      a3 += ar_tof(hh[3]) * ar_tof(hx[3]);
      a0 += ar_tof(hh[4]) * ar_tof(hx[4]);
      a1 += ar_tof(hh[5]) * ar_tof(hx[5]);
      a2 += ar_tof(hh[6]) * ar_tof(hx[6]);
      a3 += ar_tof(hh[7]) * ar_tof(hx[7]);
    }

    float acc = (a0 + a1) + (a2 + a3);
    acc = ar_warp_reduce_sum(acc);
    if (lane == 0 && acc > best) { 
      best = acc; bi = r; 
    }
  } // r

  __shared__ float sb[BLOCK / 32];
  __shared__ int si[BLOCK / 32];
  if (lane == 0) { 
    sb[warp] = best; si[warp] = bi; 
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float b = SS_NEG_INF; int idx = 0;
    for (int w = 0; w < nwarp; ++w) { 
      if (sb[w] > b) { 
        b = sb[w]; idx = si[w];
      }
    }
    pbest[blockIdx.x] = b; pidx[blockIdx.x] = idx;
  }
}


template <int BLOCK>
__global__ void ss_gemv_ln_kernel(const bf16* __restrict__ w,
                                  const bf16* __restrict__ b,
                                  bf16* __restrict__ h,
                                  const bf16* __restrict__ res,
                                  const bf16* __restrict__ lnw,
                                  const bf16* __restrict__ lnb,
                                  bf16* __restrict__ y,
                                  int N, int K, float eps, int do_gelu) {
  extern __shared__ bf16 sxb[];              // [K] normalised x (bf16)
  __shared__ float sred[BLOCK / 32];

  float s = 0.f;
  for (int i = threadIdx.x; i < K; i += BLOCK) {
    s += ar_tof(h[i]) + (res != nullptr ? ar_tof(res[i]) : 0.f);
  }

  s = velox::sheetsage::block_sum<BLOCK>(s, sred);

  const float mean = s / K;
  float sq = 0.f;
  for (int i = threadIdx.x; i < K; i += BLOCK) {
    const float d = ar_tof(h[i]) + (res != nullptr ? ar_tof(res[i]) : 0.f) - mean;
    sq += d * d;
  }

  sq = velox::sheetsage::block_sum<BLOCK>(sq, sred);

  float rstd = rsqrtf(sq / K + eps);
  for (int i = threadIdx.x; i < K; i += BLOCK) {
    float v = ar_tof(h[i]) + (res != nullptr ? ar_tof(res[i]) : 0.f);
    sxb[i] = ar_tob((v - mean) * rstd * ar_tof(lnw[i]) + ar_tof(lnb[i]));
  }
  __syncthreads();

  if (blockIdx.x == 0)
    for (int i = threadIdx.x; i < K; i += BLOCK) { h[i] = sxb[i];
   }

  // TODO (yiakwy) : reuse the device code
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5, nwarp = BLOCK >> 5;

  const int nv = K / 8;
  const uint4* __restrict__ xv = reinterpret_cast<const uint4*>(sxb);

  for (int r = blockIdx.x * nwarp + warp; r < N; r += gridDim.x * nwarp) {
    const uint4* __restrict__ wv =
        reinterpret_cast<const uint4*>(w + (long)r * K);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;

#pragma unroll 4
    for (int i = lane; i < nv; i += 32) {
      uint4 q = wv[i], xq = xv[i];
      const bf16* hh = reinterpret_cast<const bf16*>(&q);
      const bf16* hx = reinterpret_cast<const bf16*>(&xq);
      a0 += ar_tof(hh[0]) * ar_tof(hx[0]);
      a1 += ar_tof(hh[1]) * ar_tof(hx[1]);
      a2 += ar_tof(hh[2]) * ar_tof(hx[2]);
      a3 += ar_tof(hh[3]) * ar_tof(hx[3]);
      a0 += ar_tof(hh[4]) * ar_tof(hx[4]);
      a1 += ar_tof(hh[5]) * ar_tof(hx[5]);
      a2 += ar_tof(hh[6]) * ar_tof(hx[6]);
      a3 += ar_tof(hh[7]) * ar_tof(hx[7]);
    }

    float acc = (a0 + a1) + (a2 + a3);
    acc = ar_warp_reduce_sum(acc);

    if (lane == 0) {
      float o = acc + (b != nullptr ? ar_tof(b[r]) : 0.f);
      if (do_gelu)
        o = 0.5f * o * (1.f + erff(o * 0.70710678118654752f));
      y[r] = ar_tob(o);
    }
  } // r
}


template <int NWARP>
__global__ void ss_self_attn_split_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ K,
    const bf16* __restrict__ V, float* __restrict__ pm,
    float* __restrict__ pl, float* __restrict__ pacc,
    int S, const int* __restrict__ pos_dev, int split, float scale) {
  const int h = blockIdx.x, sp = blockIdx.y;
  const int rv = *pos_dev + 1;
  const int chunk = (rv + split - 1) / split;
  const int j0 = sp * chunk, j1 = min(rv, j0 + chunk);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  const bf16* qr = q + (long)h * kD;
  const bf16* Kh = K + (long)h * S * kD;
  const bf16* Vh = V + (long)h * S * kD;

  const float q0 = ar_tof(qr[2 * lane]), q1 = ar_tof(qr[2 * lane + 1]);
  float m = -1e30f, l = 0.f, a0 = 0.f, a1 = 0.f;

  for (int j = j0 + warp; j < j1; j += NWARP) {
    const bf16* kr = Kh + (long)j * kD + 2 * lane;
    float part = q0 * ar_tof(kr[0]) + q1 * ar_tof(kr[1]);

    float s = ar_warp_reduce_sum(part) * scale;

    s = __shfl_sync(0xffffffffu, s, 0);

    if (s > m) { 
      float r = __expf(m - s); 
      l *= r; 
      a0 *= r; 
      a1 *= r; 
      m = s;
    }

    float p = __expf(s - m);
    l += p;
    const bf16* vr = Vh + (long)j * kD + 2 * lane;
    a0 += p * ar_tof(vr[0]);
    a1 += p * ar_tof(vr[1]);
  }

  __shared__ float sm[NWARP], sl[NWARP], sa0[NWARP][32], sa1[NWARP][32];

  if (lane == 0) { 
    sm[warp] = m; sl[warp] = l; 
  }

  sa0[warp][lane] = a0;
  sa1[warp][lane] = a1;
  __syncthreads();

  if (warp == 0) {
    float M = sm[0];

#pragma unroll
    for (int w = 1; w < NWARP; ++w) M = fmaxf(M, sm[w]);
    float L = 0.f, A0 = 0.f, A1 = 0.f;

#pragma unroll
    for (int w = 0; w < NWARP; ++w) {
      float c = __expf(sm[w] - M);
      L += sl[w] * c;
      A0 += sa0[w][lane] * c;
      A1 += sa1[w][lane] * c;
    }

    if (lane == 0) { 
      pm[h * split + sp] = M; 
      pl[h * split + sp] = L; 
    }

    const long base = (long)h * kD * split;
    pacc[base + (long)(2 * lane) * split + sp] = A0;
    pacc[base + (long)(2 * lane + 1) * split + sp] = A1;
  }
}


// merge the SPLIT partials of every head: out[H, D]
template <int BLOCK>
__global__ void ss_attn_merge_kernel(const float* __restrict__ pm,
                                     const float* __restrict__ pl,
                                     const float* __restrict__ pacc,
                                     bf16* __restrict__ out, int split, int H_) {
  const int t = blockIdx.x * BLOCK + threadIdx.x;
  const int h = t / kD, d = t % kD;

  if (h >= H_) return;

  const float* pa = pacc + ((long)h * kD + d) * split;
  const float* pmo = pm + h * split;
  const float* plo = pl + h * split;

  float m = -1e30f, l = 0.f, acc = 0.f;
  int sp = 0;
  for (; sp + 4 <= split; sp += 4) {
    float4 a4 = *reinterpret_cast<const float4*>(pa + sp);
    float ms0 = pmo[sp], ls0 = plo[sp];
    float mn = fmaxf(m, ms0);
    float c = __expf(m - mn), c2 = __expf(ms0 - mn);

    l = l * c + ls0 * c2; 
    acc = acc * c + a4.x * c2; 
    m = mn;

    // call 0
    ms0 = pmo[sp + 1]; ls0 = plo[sp + 1];
    mn = fmaxf(m, ms0); c = __expf(m - mn); c2 = __expf(ms0 - mn);

    l = l * c + ls0 * c2; 
    acc = acc * c + a4.y * c2; 
    m = mn;

    // call 1
    ms0 = pmo[sp + 2]; ls0 = plo[sp + 2];
    mn = fmaxf(m, ms0); 
    c = __expf(m - mn); 
    c2 = __expf(ms0 - mn);

    l = l * c + 
    ls0 * c2; 
    acc = acc * c + a4.z * c2; 
    m = mn;

    // call 2
    ms0 = pmo[sp + 3]; ls0 = plo[sp + 3];
    mn = fmaxf(m, ms0); 
    c = __expf(m - mn); 
    c2 = __expf(ms0 - mn);
    l = l * c + ls0 * c2; acc = acc * c + a4.w * c2; m = mn;
  }

  for (; sp < split; ++sp) {
    float ms = pmo[sp], ls = plo[sp], a = pa[sp];
    float mn = fmaxf(m, ms);
    float c = __expf(m - mn), c2 = __expf(ms - mn);
    
    l = l * c + ls * c2; 
    acc = acc * c + a * c2; 
    m = mn;
  }
  out[(long)h * kD + d] = ar_tob(acc / fmaxf(l, 1e-30f));
}


template <int BLOCK>
__global__ void ss_gemv_row_kernel(const bf16* __restrict__ w,
                                   const bf16* __restrict__ b,
                                   const bf16* __restrict__ x,
                                   bf16* __restrict__ y, int N, int K) {
  const int r = blockIdx.x;

  if (r >= N) return;

  const int nv = K / 8;
  const uint4* __restrict__ xv = reinterpret_cast<const uint4*>(x);
  const uint4* __restrict__ wv =
      reinterpret_cast<const uint4*>(w + (long)r * K);

  float acc = 0.f;

  for (int i = threadIdx.x; i < nv; i += BLOCK) {
    uint4 q = wv[i], xq = xv[i];
    const bf16* h = reinterpret_cast<const bf16*>(&q);
    const bf16* hx = reinterpret_cast<const bf16*>(&xq);
#pragma unroll
    for (int j = 0; j < 8; ++j) { 
      acc += ar_tof(h[j]) * ar_tof(hx[j]);
    }
  }

  __shared__ float sred[BLOCK / 32];

  float s = velox::sheetsage::block_sum<BLOCK>(acc, sred);
  if (threadIdx.x == 0) {
    y[r] = ar_tob(s + (b != nullptr ? ar_tof(b[r]) : 0.f));
  }
}

}  // namespace sheetsage_ar
}  // namespace hopper
}  // namespace velox