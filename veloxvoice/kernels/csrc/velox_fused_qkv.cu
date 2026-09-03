// velox_fused_qkv.cu — fused q/k/v projection for small-M streaming shapes.
// x [M, K] fp32; Wqkv [3N, K] fp32 (concat q|k|v rows); bias [3N].
// out [M, 3N] fp32.
//
// sgl-style fast-gemv: one thread per output column n, K coalesced float4 loads
// of W, x prefetched into registers. Designed for M <= 32 (chunk streams).

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 128;
constexpr int MAX_M = 32;

__global__ void FusedQkvGevm(const float* __restrict__ x, const float* __restrict__ w,
                             const float* __restrict__ b, float* __restrict__ out,
                             int M, int K, int N3) {
  int n = blockIdx.x * THREADS + threadIdx.y * THREADS + threadIdx.x;
  if (n >= N3) return;
  const float* wrow = w + (long)n * K;
  float acc[MAX_M];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) acc[m] = 0.f;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  const float4* w4 = reinterpret_cast<const float4*>(wrow);
  int K4 = K / 4;
  for (int k4 = 0; k4 < K4; ++k4) {
    float4 wv = w4[k4];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
      if (m < M) {
        float4 xv = x4[m * K4 + k4];
        acc[m] += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
      }
    }
  }
  float bv = b[n];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    if (m < M) out[(long)m * N3 + n] = acc[m] + bv;
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
  dim3 grid((unsigned)((N3 + THREADS - 1) / THREADS), 1, 1);
  FusedQkvGevm<<<grid, THREADS, 0, stream>>>(
      static_cast<const float*>(x.data_ptr()), static_cast<const float*>(w.data_ptr()),
      static_cast<const float*>(bias.data_ptr()), static_cast<float*>(out.data_ptr()),
      (int)M, (int)K, (int)N3);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_qkv, velox_fused_qkv);
