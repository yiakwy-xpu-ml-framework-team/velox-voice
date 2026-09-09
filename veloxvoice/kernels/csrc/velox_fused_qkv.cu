// velox_fused_qkv.cu — fused q/k/v projection for small-M streaming shapes.
// x [M, K] fp32; Wqkv [3N, K] fp32 (concat q|k|v rows); bias [3N].
// out [M, 3N] fp32.
//
// sgl-style fast-gemv: one WARP per output column n; lanes walk K in coalesced
// float4 steps (lane l reads w4[l], w4[l+32], ...); per-warp shuffle reduction.
// M is a compile-time template constant (dispatch table below) so accumulators
// and the m-loop are fully unrolled with no dead predicated work.

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cuda_runtime.h>

namespace {

constexpr int WARPS_PER_BLOCK = 8;   // 256 threads
constexpr int MAX_M = 32;

template <int MC>
__global__ void FusedQkvGevm(const float* __restrict__ x, const float* __restrict__ w,
                             const float* __restrict__ b, float* __restrict__ out,
                             int K, int N3) {
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int n = blockIdx.x * WARPS_PER_BLOCK + warp;
  if (n >= N3) return;

  const float4* x4 = reinterpret_cast<const float4*>(x);
  const float4* w4 = reinterpret_cast<const float4*>(w + (long)n * K);
  int K4 = K / 4;

  float acc[MC];
#pragma unroll
  for (int m = 0; m < MC; ++m) acc[m] = 0.f;

  for (int k4 = lane; k4 < K4; k4 += 32) {
    float4 wv = w4[k4];  // lanes read consecutive float4 -> coalesced
#pragma unroll
    for (int m = 0; m < MC; ++m) {
      float4 xv = x4[m * K4 + k4];  // coalesced across lanes
      acc[m] += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
    }
  }

#pragma unroll
  for (int m = 0; m < MC; ++m) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc[m] += __shfl_down_sync(0xffffffffu, acc[m], off);
  }

  if (lane == 0) {
    float bv = b[n];
#pragma unroll
    for (int m = 0; m < MC; ++m) out[(long)m * N3 + n] = acc[m] + bv;
  }
}

}  // namespace

void velox_fused_qkv(tvm::ffi::TensorView x, tvm::ffi::TensorView w,
                     tvm::ffi::TensorView bias, tvm::ffi::TensorView out) {
  int64_t M = x.size(0), K = x.size(1), N3 = w.size(0);
  TVM_FFI_ICHECK_LE(M, MAX_M);
  TVM_FFI_ICHECK_EQ(K % 4, 0);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));

  int blocks = (int)((N3 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
  dim3 grid((unsigned)blocks, 1, 1);
  dim3 block(WARPS_PER_BLOCK * 32, 1, 1);

#define LAUNCH_QKV(MC)                                                        \
  FusedQkvGevm<MC><<<grid, block, 0, stream>>>(                               \
      static_cast<const float*>(x.data_ptr()),                                \
      static_cast<const float*>(w.data_ptr()),                                \
      static_cast<const float*>(bias.data_ptr()),                             \
      static_cast<float*>(out.data_ptr()), (int)K, (int)N3)

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
