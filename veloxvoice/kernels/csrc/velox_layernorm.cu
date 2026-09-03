// velox_layernorm.cu
// Rowwise fp32 layer-norm: x [T, D] -> y [T, D]; w/b [D]. One threadgroup per row.

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cuda_runtime.h>
#include <math.h>

namespace {

template <int BLOCK>
__global__ void LayerNormKernel(const float* __restrict__ x, const float* __restrict__ w,
                                const float* __restrict__ b, float* __restrict__ y,
                                int D, float eps) {
  int row = blockIdx.x;
  const float* xr = x + (long)row * D;
  float* yr = y + (long)row * D;
  float s = 0.f, sq = 0.f;
  for (int i = threadIdx.x; i < D; i += BLOCK) {
    float v = xr[i];
    s += v;
    sq += v * v;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    s += __shfl_down_sync(0xffffffffu, s, off);
    sq += __shfl_down_sync(0xffffffffu, sq, off);
  }
  __shared__ float sh_s[BLOCK / 32], sh_sq[BLOCK / 32];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) {
    sh_s[warp] = s;
    sh_sq[warp] = sq;
  }
  __syncthreads();
  if (warp == 0) {
    int nw = BLOCK / 32;
    s = lane < nw ? sh_s[lane] : 0.f;
    sq = lane < nw ? sh_sq[lane] : 0.f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      s += __shfl_down_sync(0xffffffffu, s, off);
      sq += __shfl_down_sync(0xffffffffu, sq, off);
    }
    if (lane == 0) {
      sh_s[0] = s;
      sh_sq[0] = sq;
    }
  }
  __syncthreads();
  float mean = sh_s[0] / (float)D;
  float var = fmaxf(sh_sq[0] / (float)D - mean * mean, 0.f);
  float inv = rsqrtf(var + eps);
  for (int i = threadIdx.x; i < D; i += BLOCK) {
    yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
  }
}

}  // namespace

void velox_layernorm_impl(tvm::ffi::TensorView x, tvm::ffi::TensorView w,
                          tvm::ffi::TensorView b, tvm::ffi::TensorView y, double eps) {
  int64_t T = x.size(0), D = x.size(1);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
  constexpr int BLOCK = 256;
  LayerNormKernel<BLOCK><<<(unsigned)T, BLOCK, 0, stream>>>(
      static_cast<const float*>(x.data_ptr()), static_cast<const float*>(w.data_ptr()),
      static_cast<const float*>(b.data_ptr()), static_cast<float*>(y.data_ptr()), (int)D,
      (float)eps);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(layernorm, velox_layernorm_impl);
