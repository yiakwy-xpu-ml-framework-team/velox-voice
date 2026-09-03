// velox_power_mel_log.cu
// Fused fbank tail: complex rFFT power -> mel matmul -> log(clamp) -> optional CMVN.
// spec: [T, F] complex (interleaved float2 storage, as from torch.fft.rfft);
// mel: [M, F] fp32; out: [T, M] fp32; cmvn_mean/cmvn_istd: [M] fp32 (used iff has_cmvn).
//
// One thread per (t, m): dot product over F (F = n_fft/2+1, e.g. 257).

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cuda_runtime.h>
#include <math.h>

namespace {

__global__ void PowerMelLogKernel(const float2* __restrict__ spec,
                                  const float* __restrict__ mel,
                                  const float* __restrict__ cmvn_mean,
                                  const float* __restrict__ cmvn_istd,
                                  float* __restrict__ out,
                                  int T, int F, int M, int has_cmvn) {
  constexpr float kLogFloor = 1.1920929e-7f;  // kaldi use-log-fbank floor
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * M) return;
  int t = idx / M, m = idx % M;
  const float* melr = mel + (long)m * F;
  const float2* sr = spec + (long)t * F;
  float acc = 0.f;
#pragma unroll 4
  for (int f = 0; f < F; ++f) {
    float2 v = sr[f];
    acc += (v.x * v.x + v.y * v.y) * melr[f];
  }
  float v = logf(fmaxf(acc, kLogFloor));
  if (has_cmvn) v = (v - cmvn_mean[m]) * cmvn_istd[m];
  out[(long)t * M + m] = v;
}

}  // namespace

void velox_power_mel_log(tvm::ffi::TensorView spec, tvm::ffi::TensorView mel,
                         tvm::ffi::TensorView cmvn_mean, tvm::ffi::TensorView cmvn_istd,
                         tvm::ffi::TensorView out, int64_t has_cmvn) {
  int64_t T = spec.size(0), F = spec.size(1), M = mel.size(0);
  TVM_FFI_ICHECK_EQ(mel.size(1), F);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(spec.device().device_type, spec.device().device_id));
  int64_t total = T * M;
  int block = 256;
  int64_t grid = (total + block - 1) / block;
  PowerMelLogKernel<<<(unsigned)grid, block, 0, stream>>>(
      static_cast<const float2*>(spec.data_ptr()),
      static_cast<const float*>(mel.data_ptr()),
      static_cast<const float*>(cmvn_mean.data_ptr()),
      static_cast<const float*>(cmvn_istd.data_ptr()),
      static_cast<float*>(out.data_ptr()), (int)T, (int)F, (int)M, (int)has_cmvn);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(power_mel_log, velox_power_mel_log);
