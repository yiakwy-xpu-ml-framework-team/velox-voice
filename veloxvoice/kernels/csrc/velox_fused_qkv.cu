/* Copyright 2026 veloxVoice authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

// velox_fused_qkv.cu — fused q/k/v projection for small-M streaming shapes in 2-stages voices (Yue2) streaming API:
// Arguments :
//   - x [M, K] fp32;
//   - Wqkv [3xN, K] fp32 (concat q|k|v rows);
//   - bias [3xN].
// out :
//   - [M, 3xN] fp32.
//
// sglang style simple but effecively fast-gemv for M <= WARP_SIZE and small K (512) in voices application.

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cuda_runtime.h>

#ifndef FQKV_SIMT_WARPS_PER_BLOCK
#define FQKV_SIMT_WARPS_PER_BLOCK 4 // 8 for dgx spark
#endif

#ifndef FQKV_SIMT_STREAMING_W
#define FQKV_SIMT_STREAMING_W 1
#endif


#ifndef CEILDIV
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#endif

#define WARP_SIZE 32

namespace veloxvoice {
namespace dgx {

constexpr int MAX_WARPS_PER_BLOCK = FQKV_SIMT_WARPS_PER_BLOCK;

// TODO (yiakwy) : support 64, 512, since epilogue cannot be overlapped by cross k-loop wgmma,see
constexpr int MAX_M = 32;

template <int MC/*rows per warp count*/, int KC_4i/*int4 vector size along K dimension per warp count */ = 0>
__global__ void FusedQkvGevm(const float* __restrict__ x, const float* __restrict__ w,
                             const float* __restrict__ b, float* __restrict__ out,
                             int K, int N, int warps_per_block) {

  // TODO (yiakwy) : using quick ptx instruction to fetch warp_id, lane_id
  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;

  int n = blockIdx.x * warps_per_block + warp_id;

  const float4* x_4f = reinterpret_cast<const float4*>(x);
  const float4* w_4f = reinterpret_cast<const float4*>(w + (long)n * K);

#define VEC_SIZE 4
  int K_VEC_SIZES = K / VEC_SIZE;

  float acc[MC];

#pragma unroll
  for (int m = 0; m < MC; ++m) {
    acc[m] = 0.f;
  }

  if constexpr (KC_4i != 0) {

#pragma unroll
    for (int k_4f = lane_id; k_4f < KC_4i; k_4f += WARP_SIZE) {

#if FQKV_SIMT_STREAMING_W
      const float4 wv = __ldcs(&w_4f[k_4f]);
#else
      const float4 wv = w_4f[k_4f];
#endif // FQKV_SIMT_STREAMING_W

#pragma unroll
      for (int m = 0; m < MC; ++m) {
        const float4 xv = x_4f[m * KC_4i + k_4f];
        acc[m] += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
      }
    }

  } else {

    for (int k_4f = lane_id; k_4f < K_VEC_SIZES; k_4f += WARP_SIZE) {

#if FQKV_SIMT_STREAMING_W
      const float4 wv = __ldcs(&w_4f[k_4f]);
#else
      const float4 wv = w_4f[k_4f];
#endif // FQKV_SIMT_STREAMING_W

#pragma unroll
      for (int m = 0; m < MC; ++m) {
        const float4 xv = x_4f[m * K_VEC_SIZES + k_4f];
        acc[m] += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
      }
    }
  } // KC_4i != 0

#pragma unroll
  for (int m = 0; m < MC; ++m) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      // acc[m] += __shfl_down_sync(0xffffffffu, acc[m], off);
      acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], off);
  }

  // NOTE (yiakwy) : using MIN(MC, WARP_SIZE) to write its column for MC > WARP_SIZE
  if (lane_id < MC) {
    float bv = b[n];
    out[(long)lane_id * N + n] = acc[lane_id] + bv;
  }

}

}  // namespace dgx
}  // namespace veloxvoice

void velox_fused_qkv(tvm::ffi::TensorView x, tvm::ffi::TensorView w,
                     tvm::ffi::TensorView bias, tvm::ffi::TensorView out) {

  namespace vx = veloxvoice::dgx;

  int64_t M = x.size(0);
  int64_t N = w.size(0);
  int64_t K = x.size(1);

  TVM_FFI_ICHECK_LE(M, vx::MAX_M);
  TVM_FFI_ICHECK_GT(N, 0);

  // NOTE (yiakwy) : add support non 128bit (float4) coalesced version
  TVM_FFI_ICHECK_EQ(K % 4, 0);

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));

  int warps = vx::MAX_WARPS_PER_BLOCK;

  // NOTE (yiakwy) : adjust warps so that every warp corresponding to a column
  while (warps > 1 && N % warps != 0) {
    warps >>= 1;
  }

  const int blocks = N / warps;

  dim3 grid((unsigned)blocks, 1, 1);
  dim3 block(warps * WARP_SIZE, 1, 1);

#define LAUNCH_QKV(MC)                                                        \
  do {                                                                        \
    if (K == 512) {                                                           \
      vx::FusedQkvGevm<MC, 128><<<grid, block, 0, stream>>>(                     \
          static_cast<const float*>(x.data_ptr()),                            \
          static_cast<const float*>(w.data_ptr()),                            \
          static_cast<const float*>(bias.data_ptr()),                         \
          static_cast<float*>(out.data_ptr()), (int)K, (int)N, warps);        \
    } else if (K == 1024) {                                                   \
      vx::FusedQkvGevm<MC, 256><<<grid, block, 0, stream>>>(                     \
          static_cast<const float*>(x.data_ptr()),                            \
          static_cast<const float*>(w.data_ptr()),                            \
          static_cast<const float*>(bias.data_ptr()),                         \
          static_cast<float*>(out.data_ptr()), (int)K, (int)N, warps);        \
    } else {                                                                  \
      vx::FusedQkvGevm<MC><<<grid, block, 0, stream>>>(                           \
          static_cast<const float*>(x.data_ptr()),                            \
          static_cast<const float*>(w.data_ptr()),                            \
          static_cast<const float*>(bias.data_ptr()),                         \
          static_cast<float*>(out.data_ptr()), (int)K, (int)N, warps);        \
    }                                                                         \
  } while (0)

  if (M == 1) {
    LAUNCH_QKV(1);
  } else if (M <= 2) {
    LAUNCH_QKV(2);
  } else if (M <= 4) {
    LAUNCH_QKV(4);
  } else if (M <= 8) {
    LAUNCH_QKV(8);
  } else if (M <= 16) {
    LAUNCH_QKV(16);
  } else {
    LAUNCH_QKV(32);
  }
#undef LAUNCH_QKV
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_qkv, velox_fused_qkv);
