// velox_silu_glu.cu
// GLU with silu sigmoid: x [T, 2C] -> y [T, C]; y = x[:, :C] * sigmoid(x[:, C:]).
// (conformer ConvolutionModule pointwise1 activation)

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cuda_runtime.h>

namespace {

__global__ void SiluGluKernel(const float* __restrict__ x, float* __restrict__ y,
                              int T, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * C) return;
  int t = idx / C, c = idx % C;
  float a = x[(long)t * 2 * C + c];
  float b = x[(long)t * 2 * C + C + c];
  y[(long)t * C + c] = a / (1.f + expf(-b));
}

}  // namespace

void velox_silu_glu_impl(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  int64_t T = x.size(0), C = y.size(1);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
  int64_t total = T * C;
  int block = 256;
  int64_t grid = (total + block - 1) / block;
  SiluGluKernel<<<(unsigned)grid, block, 0, stream>>>(
      static_cast<const float*>(x.data_ptr()), static_cast<float*>(y.data_ptr()), (int)T,
      (int)C);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(silu_glu, velox_silu_glu_impl);
