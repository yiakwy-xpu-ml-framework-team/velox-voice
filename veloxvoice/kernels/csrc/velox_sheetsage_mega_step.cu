/* Copyright 2026 veloxVoice authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

// velox_sheetsage_mega_step.cu
//
// NOTE (yiakwy) : this baseline mega kernel
//
// Experimental impl alternative to cudagraph with pdl (0.5ms/step) for workload analsis
//
// All per-layer weights are packed into ONE bf16 blob :
//
// w_blob layout :
//   [      0] qkv_w 1536x512   [ 786432] qkv_b 1536
//   [ 787968] out_w 512x512    [1050240] out_b 512
//   [1050752] xq_w  512x512    [1313024] xq_b  512
//   [1313536] fc1_w 2048x512   [2359808] fc1_b 2048
//   [2361856] fc2_w 512x2048   [3410304] fc2_b 512
//   [3410816] ln1_w 512        [3411328] ln1_b 512
//   [3411840] ln2_w 512        [3412352] ln2_b 512
//   [3412864] ln3_w 512        [3413376] ln3_b 512
//
// scratch blob (bf16): h[512] | qkv[1536] | ffn[2048] | attn[512] | tmp[512]

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdlib.h>

#include "hopper/arch/thread/pdl_sm90.h"

#include "hopper/fragment/sheetsage_ln.h"
#include "hopper/fragment/sheetsage_gemv.h"
#include "hopper/fragment/sheetsage_attn.h"

namespace cg = cooperative_groups;
using namespace velox::sheetsage;

namespace {

// SS_NEG_INF is defined in sheetsage_attn.h as SS_SS_NEG_INF

using bf16 = __nv_bfloat16;

constexpr int kLayers = 6;
constexpr int kHeads = 8;
constexpr int kD = 64;
constexpr int kModel = 512;
constexpr int kFFN = 2048;
constexpr int kQKV = 3 * kModel;
constexpr int kThreads = 512;
constexpr int kWarp = kThreads / 32;  // 16

constexpr int kGridBlocks = 8;
constexpr int kMaxBlocks = 64;

constexpr long long FULL_MASK = 0xffffffffu;

// w_blob offsets :
constexpr long kQkvWOff = 0;
constexpr long kQkvBOff = kQkvWOff + (long)kQKV * kModel;

constexpr long kOutWOff = kQkvBOff + kQKV;
constexpr long kOutBOff = kOutWOff + (long)kModel * kModel;

constexpr long kXqWOff = kOutBOff + kModel;
constexpr long kXqBOff = kXqWOff + (long)kModel * kModel;

constexpr long kFc1WOff = kXqBOff + kModel;
constexpr long kFc1BOff = kFc1WOff + (long)kFFN * kModel;

constexpr long kFc2WOff = kFc1BOff + kFFN;
constexpr long kFc2BOff = kFc2WOff + (long)kModel * kFFN;

constexpr long kLn1WOff = kFc2BOff + kModel;
constexpr long kLn1BOff = kLn1WOff + kModel;

constexpr long kLn2WOff = kLn1BOff + kModel;
constexpr long kLn2BOff = kLn2WOff + kModel;

constexpr long kLn3WOff = kLn2BOff + kModel;
constexpr long kLn3BOff = kLn3WOff + kModel;

constexpr long kLayerStride = kLn3BOff + kModel;

constexpr long kLnEmbWOff = kLayerStride * kLayers;
constexpr long kLnEmbBOff = kLnEmbWOff + kModel;

// scratch blob offsets in bf16 :
constexpr long kScrH = 0;
constexpr long kScrQkv = kScrH + kModel;
constexpr long kScrFfn = kScrQkv + kQKV;
constexpr long kScrAttn = kScrFfn + kFFN;
constexpr long kScrTmp = kScrAttn + kModel;

// NOTE (yiakwy) : warps 0-7, GEMV + attention compute
constexpr int kComputeWarps = 8;

// NOTE (yiakwy) : warps 8-15, weight prefetch into shared
constexpr int kLoadWarps = 8; 
constexpr int kLoadOffset = kComputeWarps * 32;

constexpr int kPrefetchRows = 16;
constexpr int kPrefetchSmem = kPrefetchRows * kModel * 2;  // 16KB bf16

// red blob offsets (fp32 elements; sized for kMaxBlocks blocks)
constexpr long kRedPm2 = 0;
constexpr long kRedPacc2 = kRedPm2 + (long)kMaxBlocks * kHeads * 2;
constexpr long kRedCmax = kRedPacc2 + (long)kMaxBlocks * kHeads * kD;
constexpr long kRedCidx = kRedCmax + kMaxBlocks;

struct StepParams {
  const bf16* __restrict__ w_blob;     // packed weights (layout above)
  const bf16* __restrict__ token_emb;  // [V, 512]
  const bf16* __restrict__ pos_table;  // [S+2, 512]
  const bf16* __restrict__ k_cross;    // [L, H, T, D]
  const bf16* __restrict__ v_cross;    // [L, H, T, D]
  const bf16* __restrict__ lm_head;    // [V, 512]
  const unsigned char* __restrict__ allowed;  // [V]
  const long long* __restrict__ x_in;  // [1]
  const int* __restrict__ pos_ptr;     // [1] device int
  bf16* __restrict__ k_self;           // [L, H, S, D]
  bf16* __restrict__ v_self;           // [L, H, S, D]
  bf16* __restrict__ scratch;          // h|qkv|ffn|attn|tmp
  float* __restrict__ red;             // pm2|pacc2|cmax|cidx
  bf16* __restrict__ logits;           // [V]
  int* __restrict__ token_out;         // [1]
  float* __restrict__ dbg;             // [8]: [0..5] stage clocks (as fp32 bits via reinterpret), [6] mean, [7] rstd
  int S;
  int T;
  int V;
  int stage_mask;  // debug: bit0 embed, bit1 qkv, bit2 self-attn, bit3 L3,
                   // bit4 cross-attn, bit5 ffn, bit6 lm_head
  float eps;
  float scale;
}; // StepParams 


// SIMT impl of online-softmax attention for decoding.
template <int BLOCK>
__device__ __forceinline__ void attn_stage(
    const bf16* __restrict__ q, const bf16* __restrict__ K,
    const bf16* __restrict__ V, int R, int rows_valid, float scale, int gbid,
    float* sm_m,     // [kWarp][kHeads]
    float* sm_l,     // [kWarp][kHeads]
    float* sm_acc,   // [kWarp][kHeads][kD]
    float* pm2,      // [G, H, 2]
    float* pacc2) {  // [G, H, D]
  constexpr int kW = BLOCK / 32;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  const long npairs = (long)kHeads * rows_valid;
  const long stride = (long)gridDim.x * kW;

  const long me = (long)gbid * kW + warp;

  for (int wh = threadIdx.x; wh < kW * kHeads; wh += BLOCK) {
    sm_m[wh] = SS_NEG_INF;
    sm_l[wh] = 0.f;
  }

  for (int i = threadIdx.x; i < kW * kHeads * kD; i += BLOCK) { 
    sm_acc[i] = 0.f; 
  }
  __syncthreads();

  for (long hj = me; hj < npairs; hj += stride) {
    int head = (int)(hj / rows_valid);
    int j = (int)(hj % rows_valid);

    const bf16* qrow = q + head * kD;
    const bf16* krow = K + ((long)head * R + j) * kD;

    float s = 0.f;
#pragma unroll
    for (int d = lane; d < kD; d += 32) { 
      s += tof(krow[d]) * tof(qrow[d]); 
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      s += __shfl_down_sync(0xffffffffu, s, off);
    }
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
          pw * tof(V[((long)head * R + j) * kD + d]);
    if (lane == 0)
      sm_l[warp * kHeads + head] = sm_l[warp * kHeads + head] * rescale + pw;
  }
  __syncthreads();

  for (int head = 0; head < kHeads; ++head) {

    float mstar = SS_NEG_INF;

    for (int w = 0; w < kW; ++w) { 
      mstar = fmaxf(mstar, sm_m[w * kHeads + head]); 
    }

    for (int d = threadIdx.x; d < kD; d += BLOCK) {
      float num = 0.f, den = 0.f;

      for (int w = 0; w < kW; ++w) {
        float mw = sm_m[w * kHeads + head];

        if (mw == SS_NEG_INF) { 
          continue; 
        }

        float r = __expf(mw - mstar);
        den += sm_l[w * kHeads + head] * r;
        num += sm_acc[(w * kHeads + head) * kD + d] * r;
      } // w

      if (threadIdx.x == 0) {
        float* mp2 = pm2 + ((long)gbid * kHeads + head) * 2;
        mp2[0] = mstar;
        mp2[1] = den;
      }
      pacc2[((long)gbid * kHeads + head) * kD + d] = num / fmaxf(den, 1e-30f);
    } // d
    __syncthreads();

  } // head

} // attn_stage<BLOCK> device func impl


__device__ __forceinline__ void attn_merge(
    const float* __restrict__ pm2, const float* __restrict__ pacc2,
    int nblocks, bf16* __restrict__ attn_out) {
  for (int hd = blockIdx.x * blockDim.x + threadIdx.x; hd < kHeads * kD; hd += gridDim.x * blockDim.x) {
    int head = hd / kD, d = hd % kD;
    float mstar = SS_NEG_INF;

    for (int g = 0; g < nblocks; ++g) {
      mstar = fmaxf(mstar, pm2[((long)g * kHeads + head) * 2]);
    }

    float num = 0.f, den = 0.f;
    for (int g = 0; g < nblocks; ++g) {
      const long idx = (long)g * kHeads + head;
      float m = pm2[idx * 2];

      if (m == SS_NEG_INF) { 
        continue; 
      }

      float r = __expf(m - mstar);
      den += pm2[idx * 2 + 1] * r;
      num += pacc2[idx * kD + d] * r;
    }
    attn_out[hd] = tob(num / fmaxf(den, 1e-30f));
  }
  __syncthreads();

} // attn_merge device func impl


// bf16 gemv (y[n] = W[n,:].x + b[n])
__device__ __forceinline__ void gemv_grid(const bf16* __restrict__ w,
                                          const bf16* __restrict__ b,
                                          const bf16* __restrict__ x,
                                          bf16* __restrict__ y, int N) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  const long gw = (long)blockIdx.x * kWarp + warp;
  const long gs = (long)gridDim.x * kWarp;
  for (long n = gw; n < N; n += gs) {
    const bf16* wrow = w + n * kModel;
    float acc = (b != nullptr) ? tof(b[n]) : 0.f;

#pragma unroll 4
    for (int d = lane; d < kModel; d += 32) { 
      acc += tof(wrow[d]) * tof(x[d]); 
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    if (lane == 0) { 
      y[n] = tob(acc); 
    }
  }
  __syncthreads();

} // gemv_grid device func impl


__device__ StepParams g_step_params;


template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
SheetsageMegaDecodeStep(int nblocks) {
  StepParams p = g_step_params;

  cg::grid_group grid = cg::this_grid();

  __shared__ float sred[BLOCK / 32];
  __shared__ float sm_m[(BLOCK / 32) * kHeads];      // [warp][head]
  __shared__ float sm_l[(BLOCK / 32) * kHeads];      // [warp][head]
  __shared__ float sm_acc[(BLOCK / 32) * kHeads * kD];

  bf16* h = p.scratch + kScrH;
  bf16* qkv_buf = p.scratch + kScrQkv;
  bf16* ffn_buf = p.scratch + kScrFfn;
  bf16* attn_buf = p.scratch + kScrAttn;
  bf16* tmp_buf = p.scratch + kScrTmp;

  float* pm2 = p.red + kRedPm2;
  float* pacc2 = p.red + kRedPacc2;
  float* cmax = p.red + kRedCmax;
  float* cidx = p.red + kRedCidx;

  const int post = *p.pos_ptr;
  const int x = (int)(*p.x_in);

  // tracing statics
  long long t0 = (p.dbg != nullptr) ? clock64() : 0;

  long long acc_t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

#define SS_TIC(i) long long tic_##i = (p.dbg != nullptr) ? clock64() : 0
#define SS_TOC(i) acc_t[i] += (p.dbg != nullptr) ? (clock64() - tic_##i) : 0

  // step 1 : embed + pos row + LN
  if (p.stage_mask & 1) {
    const bf16* emb = p.token_emb + (long)x * kModel;
    const bf16* prow = p.pos_table + (long)(post + 2) * kModel;
    for (int i = threadIdx.x; i < kModel; i += BLOCK)
      h[i] = tob(tof(emb[i]) + tof(prow[i]));
    __syncthreads();
    layernorm_frag<BLOCK, 512>(h, p.w_blob + kLnEmbWOff, p.w_blob + kLnEmbBOff,
                               p.eps, sred);
  }
  grid.sync();

  for (int l = 0; l < kLayers; ++l) {
    const bf16* wb = p.w_blob + (long)l * kLayerStride;
    const bf16* w_qkv = wb + kQkvWOff;
    const bf16* b_qkv = wb + kQkvBOff;
    const bf16* w_out = wb + kOutWOff;
    const bf16* b_out = wb + kOutBOff;
    const bf16* w_xq = wb + kXqWOff;
    const bf16* b_xq = wb + kXqBOff;
    const bf16* w_f1 = wb + kFc1WOff;
    const bf16* b_f1 = wb + kFc1BOff;
    const bf16* w_f2 = wb + kFc2WOff;
    const bf16* b_f2 = wb + kFc2BOff;
    const bf16* w_l1 = wb + kLn1WOff;
    const bf16* b_l1 = wb + kLn1BOff;
    const bf16* w_l2 = wb + kLn2WOff;
    const bf16* b_l2 = wb + kLn2BOff;
    const bf16* w_l3 = wb + kLn3WOff;
    const bf16* b_l3 = wb + kLn3BOff;
    bf16* Ks = p.k_self + (long)l * kHeads * p.S * kD;
    bf16* Vs = p.v_self + (long)l * kHeads * p.S * kD;
    const bf16* Kc = p.k_cross + (long)l * kHeads * p.T * kD;
    const bf16* Vc = p.v_cross + (long)l * kHeads * p.T * kD;

    // step 2 : packed qkv GEMM
    SS_TIC(1);
    {
      const int lane = threadIdx.x & 31;
      const int warp = threadIdx.x >> 5;

      const long gw = (long)blockIdx.x * kWarp + warp;
      const long gw_stride = (long)gridDim.x * kWarp;
      for (long n = (p.stage_mask & 2) ? gw : kQKV; n < kQKV; n += gw_stride) {
        const bf16* wrow = w_qkv + n * kModel;
        float acc = tof(b_qkv[n]);
#pragma unroll 4
        for (int d = lane; d < kModel; d += 32)
          acc += tof(wrow[d]) * tof(h[d]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          acc += __shfl_down_sync(0xffffffffu, acc, off);
        acc = __shfl_sync(0xffffffffu, acc, 0);
        qkv_buf[n] = tob(acc);
        if (n >= kModel) {  // K [512,1024), V [1024,1536)
          int rem = (int)n - kModel;
          int r = rem % kModel;
          long off = ((long)(r / kD) * p.S + post) * kD + (r % kD);
          if (rem < kModel) Ks[off] = tob(acc); else Vs[off] = tob(acc);
        }
      }
    } // SS_TIC(1);
    __syncthreads();
    SS_TOC(1);

    grid.sync();

    // step 3 : self-attn partials over rows [0, post]
    SS_TIC(2);
    if (p.stage_mask & 4) {
      attn_stage<BLOCK>(qkv_buf /* q */, Ks, Vs, p.S, post + 1, p.scale,
                        blockIdx.x, sm_m, sm_l, sm_acc, pm2, pacc2);
    }
    SS_TOC(2);
    grid.sync();

    // step 4 : self-attn merge + out_proj; residual+LN
    SS_TIC(3);
    if (p.stage_mask & 8) {
      attn_merge(pm2, pacc2, nblocks, attn_buf);
      grid.sync();

      gemv_grid(w_out, b_out, attn_buf, tmp_buf, kModel);
      grid.sync();

      for (int i = threadIdx.x; i < kModel; i += BLOCK) {
        h[i] = tob(tof(h[i]) + tof(tmp_buf[i]));
      }
      __syncthreads();

      layernorm_frag<BLOCK, 512>(h, w_l1, b_l1, p.eps, sred);
      __syncthreads();

      gemv_grid(w_xq, b_xq, h, qkv_buf, kModel);  // cross q -> qkv_buf
    } // SS_TIC(3);
    SS_TOC(3);

    grid.sync();

    // stage 5 : cross-attn partials over all T rows
    SS_TIC(4);
    if (p.stage_mask & 16) {
      attn_stage<BLOCK>(qkv_buf, Kc, Vc, p.T, p.T, p.scale, blockIdx.x,
                        sm_m, sm_l, sm_acc, pm2, pacc2);
    } //  SS_TIC(4)
    SS_TOC(4);
    grid.sync();

    // step 6 : attn merge + out_proj + ffn; residual+LN
    SS_TIC(5);
    if (p.stage_mask & 32) {
      attn_merge(pm2, pacc2, nblocks, attn_buf);
      grid.sync();

      gemv_grid(w_out, b_out, attn_buf, tmp_buf, kModel);
      grid.sync();

      for (int i = threadIdx.x; i < kModel; i += BLOCK) {
        h[i] = tob(tof(h[i]) + tof(tmp_buf[i]));
      }
      __syncthreads();

      layernorm_frag<BLOCK, 512>(h, w_l2, b_l2, p.eps, sred);
      __syncthreads();

      gemv_grid(w_f1, b_f1, h, ffn_buf, kFFN);
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kFFN;
           i += gridDim.x * blockDim.x) {
        float v = tof(ffn_buf[i]);
        ffn_buf[i] = tob(0.5f * v * (1.f + erff(v * 0.70710678118654752f)));
      }

      grid.sync();

      gemv_grid(w_f2, b_f2, ffn_buf, tmp_buf, kModel);

      grid.sync();

      for (int i = threadIdx.x; i < kModel; i += BLOCK) {
        h[i] = tob(tof(h[i]) + tof(tmp_buf[i]));
      }
      __syncthreads();

      layernorm_frag<BLOCK, 512>(h, w_l3, b_l3, p.eps, sred);
      __syncthreads();
    } //  SS_TIC(5)
    SS_TOC(5);

    grid.sync();
  }

  // lm_head : logits slice + per-block masked argmax candidate
  if (p.stage_mask & 64) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const long gw = (long)blockIdx.x * kWarp + warp;
    const long gw_stride = (long)gridDim.x * kWarp;

    float best = SS_NEG_INF;
    int bi = 0;

    for (long v = gw; v < p.V; v += gw_stride) {
      const bf16* wrow = p.lm_head + v * kModel;
      float acc = 0.f;

#pragma unroll 4
      for (int d = lane; d < kModel; d += 32) { 
        acc += tof(wrow[d]) * tof(h[d]); 
      }

#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
      }
      acc = __shfl_sync(0xffffffffu, acc, 0);

      p.logits[v] = tob(acc);

      float lb = tof(p.logits[v]);
      if (p.allowed[v] && lb > best) {
        best = lb;
        bi = (int)v;
      }
    }
    __syncthreads();

    __shared__ float sb[BLOCK / 32];
    __shared__ int ib[BLOCK / 32];

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      float ob = __shfl_down_sync(0xffffffffu, best, off);
      int oi = __shfl_down_sync(0xffffffffu, bi, off);
      if (ob > best || (ob == best && oi < bi)) {
        best = ob;
        bi = oi;
      }
    }

    if (lane == 0) {
      sb[warp] = best;
      ib[warp] = bi;
    }
    __syncthreads();

    if (warp == 0) {
      best = (lane < BLOCK / 32) ? sb[lane] : SS_NEG_INF;
      bi = (lane < BLOCK / 32) ? ib[lane] : 0;

#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        float ob = __shfl_down_sync(0xffffffffu, best, off);
        int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ob > best || (ob == best && oi < bi)) {
          best = ob;
          bi = oi;
        }
      } // off

      if (lane == 0) {
        cmax[blockIdx.x] = best;
        cidx[blockIdx.x] = (float)bi;  // exact below 2^24
      }
    } // warp == 0
  } // lm_head 

  grid.sync();

  if (p.dbg != nullptr && blockIdx.x == 0 && threadIdx.x == 0) {
    p.dbg[0] = (float)(clock64() - t0);
  }

  // global masked argmax
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    float best = SS_NEG_INF;
    int bi = 0;
    for (int g = 0; g < nblocks; ++g) {
      if (cmax[g] > best || (cmax[g] == best && (int)cidx[g] < bi)) {
        best = cmax[g];
        bi = (int)cidx[g];
      }
    }
    *p.token_out = bi;
  }
} // SheetsageMegaDecodeStep device func impl


// TODO (yiakwy) : remove
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
SheetsageProbeKernel(int* out) {
  cg::grid_group grid = cg::this_grid();
  if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = 0 /*pre-sync*/;
  grid.sync();
  if (threadIdx.x == 0 && blockIdx.x == 0) out[1] = 1 /*post-sync*/;
}


// ---------------------------------------------------------------------------
// Integration to Harnessing Agent Orchestrator (HAO) 
//  - (https://github.com/yiakwy-xpu-ml-framework-team/flash-float-jit-kernels/blob/main/tools/KERNEL_AGENT.md)
// ---------------------------------------------------------------------------


// TODO (yiakwy) : remove
void velox_ss_probe_impl(tvm::ffi::TensorView out) {
  int device = 0;
  cudaGetDevice(&device);
  int sms = 0, per_sm = 0;

#define kTestThreads 128

  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm, SheetsageProbeKernel<kTestThreads>, kTestThreads, 0);

  int grid_mn = sms * (per_sm > 0 ? per_sm : 1);

  dim3 grid(grid_mn, 1, 1);
  dim3 block(kTestThreads, 1, 1);

  int* outp = static_cast<int*>(out.data_ptr());

  void* args[] = {&outp};
  cudaError_t err = cudaLaunchCooperativeKernel(
      (void*)SheetsageProbeKernel<kTestThreads>, grid, block, args, 0,
      static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(
          out.device().device_type, out.device().device_id
        )
      )
  );
  
  if (err != cudaSuccess) {
    fprintf(stderr, "[ss-probe-error] %s (grid=%d)\n",
            cudaGetErrorString(err), grid);
  }
}

// block_sum isolation probe: x[i]=i+1; every thread must see the same sum.
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
SheetsageSumProbeKernel(float* out) {
  __shared__ float red[BLOCK / 32];
  float v = (float)(threadIdx.x + 1);
  v = block_sum<BLOCK>(v, red);
  out[threadIdx.x] = v;
}

void velox_ss_sum_probe_impl(tvm::ffi::TensorView out) {
  SheetsageSumProbeKernel<512><<<1, 512>>>(
      static_cast<float*>(out.data_ptr()));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(velox_ss_sum_probe, velox_ss_sum_probe_impl);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(velox_ss_probe, velox_ss_probe_impl);

}  // anonymous namespace


// NOTE (yiakwy) : kernel entry
void velox_sheetsage_mega_decode_step_impl(
    tvm::ffi::TensorView x_in,        // [1,1] int64
    tvm::ffi::TensorView pos,         // [1] int32 (device)
    tvm::ffi::TensorView allowed,     // [V] bool
    tvm::ffi::TensorView w_blob,      // bf16, layout in the header comment
    tvm::ffi::TensorView token_emb,   // [V, 512] bf16
    tvm::ffi::TensorView pos_table,   // [S+2, 512] bf16
    tvm::ffi::TensorView k_self,      // [L, H, S, D] bf16
    tvm::ffi::TensorView v_self,      // [L, H, S, D] bf16
    tvm::ffi::TensorView k_cross,     // [L, H, T, D] bf16
    tvm::ffi::TensorView v_cross,     // [L, H, T, D] bf16
    tvm::ffi::TensorView lm_head,     // [V, 512] bf16
    tvm::ffi::TensorView scratch,     // bf16: h|qkv|ffn|attn|tmp
    tvm::ffi::TensorView red,         // fp32: pm2|pacc2|cmax|cidx
    tvm::ffi::TensorView logits,      // [V] bf16
    tvm::ffi::TensorView token_out,   // [1] int32
    tvm::ffi::TensorView dbg,         // [8] int64 host mirror (or empty)
    int64_t S, int64_t T) {

  StepParams p;
  p.w_blob = static_cast<const bf16*>(w_blob.data_ptr());
  p.token_emb = static_cast<const bf16*>(token_emb.data_ptr());
  p.pos_table = static_cast<const bf16*>(pos_table.data_ptr());
  p.k_cross = static_cast<const bf16*>(k_cross.data_ptr());
  p.v_cross = static_cast<const bf16*>(v_cross.data_ptr());
  p.lm_head = static_cast<const bf16*>(lm_head.data_ptr());
  p.allowed = static_cast<const unsigned char*>(allowed.data_ptr());
  p.x_in = static_cast<const long long*>(x_in.data_ptr());
  p.pos_ptr = static_cast<const int*>(pos.data_ptr());
  p.k_self = static_cast<bf16*>(k_self.data_ptr());
  p.v_self = static_cast<bf16*>(v_self.data_ptr());
  p.scratch = static_cast<bf16*>(scratch.data_ptr());
  p.red = static_cast<float*>(red.data_ptr());
  p.logits = static_cast<bf16*>(logits.data_ptr());
  p.token_out = static_cast<int*>(token_out.data_ptr());
  p.dbg = dbg.size(0) >= 8 ? static_cast<float*>(dbg.data_ptr()) : nullptr;
  p.S = (int)S;
  p.T = (int)T;
  p.V = (int)token_emb.size(0);
  p.eps = 1e-5f;
  p.scale = 0.125f;  // 1/sqrt(64), the sdpa default scale
  {
    const char* env = getenv("VELOX_SSMASK");
    p.stage_mask = env ? atoi(env) : 127;
  }

  int device = 0;
  cudaGetDevice(&device);

  int sms = 0, per_sm = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm, SheetsageMegaDecodeStep<kThreads>, kThreads, 0);
  
  int grid = sms * (per_sm > 0 ? per_sm : 1);
  if (grid > kMaxBlocks) grid = kMaxBlocks;

  if (getenv("VELOX_SSDBG")) {
    fprintf(stderr, "[ss-host] x_in=%p pos=%p allowed=%p w=%p tok=%p "
            "ptab=%p ks=%p vs=%p kc=%p vc=%p lm=%p scr=%p red=%p "
            "lg=%p to=%p S=%ld T=%ld\n",
            (void*)p.x_in, (void*)p.pos_ptr, (void*)p.allowed,
            (void*)p.w_blob, (void*)p.token_emb, (void*)p.pos_table,
            (void*)p.k_self, (void*)p.v_self, (void*)p.k_cross,
            (void*)p.v_cross, (void*)p.lm_head, (void*)p.scratch,
            (void*)p.red, (void*)p.logits, (void*)p.token_out,
            (long)S, (long)T);
  }

  int nb = grid;
  void* args[] = {&nb};
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x_in.device().device_type, x_in.device().device_id));

  cudaError_t err = cudaMemcpyToSymbolAsync(
      g_step_params, &p, sizeof(StepParams), 0, cudaMemcpyHostToDevice, stream);
  
  if (err != cudaSuccess) {
    fprintf(stderr, "[ss-memcpy-error] %s\n", cudaGetErrorString(err));
    return;
  }

  err = cudaLaunchCooperativeKernel(
      (void*)SheetsageMegaDecodeStep<kThreads>, dim3(grid), dim3(kThreads),
      args, 0, stream);
  
  if (err != cudaSuccess) {
    fprintf(stderr, "[ss-launch-error] %s (grid=%d)\n",
            cudaGetErrorString(err), grid);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(velox_sheetsage_mega_decode_step, velox_sheetsage_mega_decode_step_impl);
