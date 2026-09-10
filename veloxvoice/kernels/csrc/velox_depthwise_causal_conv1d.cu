// velox_depthwise_causal_conv1d.cu
// Depthwise causal conv1d with rolling left cache.
// x: [T, C] time-major; cache: [K-1, C]; weight: [C, K]; bias: [C]
// out: [T, C]; new_cache: [K-1, C]
// One thread per (c, t); new cache rolled in the same launch.

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cuda_runtime.h>

namespace {

__global__ void DepthwiseCausalConv1dKernel(const float* __restrict__ x,
                                            const float* __restrict__ cache,
                                            const float* __restrict__ weight,
                                            const float* __restrict__ bias,
                                            float* __restrict__ out,
                                            float* __restrict__ new_cache,
                                            int T, int C, int K) {
  int cid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = blockIdx.y;
  if (cid >= C) return;
  float acc = bias[cid];
  for (int j = 0; j < K; ++j) {
    int tt = tid + j - (K - 1);
    float v = (tt < 0) ? cache[(tt + K - 1) * C + cid] : x[(long)tt * C + cid];
    acc += weight[cid * K + j] * v;
  }
  out[(long)tid * C + cid] = acc;
  // roll the left cache: all K-1 rows, strided over the T threads in blockIdx.y
  for (int r = tid; r < K - 1; r += T) {  // row = concat(cache, x)[T + r]
    int src = T + r;
    new_cache[r * C + cid] = (src < K - 1) ? cache[src * C + cid]
                                           : x[(long)(src - (K - 1)) * C + cid];
  }
}

}  // namespace

void velox_dw_conv1d_impl(tvm::ffi::TensorView x, tvm::ffi::TensorView cache,
                          tvm::ffi::TensorView weight, tvm::ffi::TensorView bias,
                          tvm::ffi::TensorView out, tvm::ffi::TensorView new_cache) {
  int64_t T = x.size(0), C = x.size(1), K = weight.size(1);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
  dim3 block(128, 1, 1);
  dim3 grid((unsigned)((C + block.x - 1) / block.x), (unsigned)T, 1);
  DepthwiseCausalConv1dKernel<<<grid, block, 0, stream>>>(
      static_cast<const float*>(x.data_ptr()), static_cast<const float*>(cache.data_ptr()),
      static_cast<const float*>(weight.data_ptr()),
      static_cast<const float*>(bias.data_ptr()), static_cast<float*>(out.data_ptr()),
      static_cast<float*>(new_cache.data_ptr()), (int)T, (int)C, (int)K);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dw_causal_conv1d, velox_dw_conv1d_impl);
