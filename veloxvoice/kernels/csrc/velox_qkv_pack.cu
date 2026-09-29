// velox_qkv_pack.cu — single-pass pack of the attention fold chain.
//
// Replaces (per conformer layer, full-context lane):
//   q,k,v = qkv.view(..3,h,dk).unbind(2).transpose(1,2)   # strided views
//   q2 = cat([(q+pos_u), (q+pos_v)], -1) * scale
//   k2 = cat([k, p0], -1)
//   v2 = cat([v, 0], -1)
// (~8 small strided kernels, H800-measured 72 us/layer) with ONE contiguous
// pass (bandwidth floor ~8 us/layer). Pure elementwise + gather — no tensor
// cores, no fragment layouts; runs on any CUDA arch (H800 sm_90, GB10 sm_121).
//
// Inputs  qkv [T, 3d] bf16 (t-major: q | k | v, d = h*dk per token)
//         p0  [T, d]  bf16 (lp(pos_emb) raw output, h-major per token)
//         u/v [h, dk] bf16 (pos_bias_u / pos_bias_v)
//         scale fp32
// Outputs q2 [h, T, 2dk] = (q+u, q+v) * scale   (bf16, double-rounded to
//         k2 [h, T, 2dk] = (k, p0)                reproduce torch's bf16
//         v2 [h, T, 2dk] = (v, 0)                 elementwise rounding)

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "hopper/arch/pdl_sm90.h"

namespace {

__device__ __forceinline__ float rnd_bf16(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

// one thread handles a (h, t, j2) pair: two consecutive j of the 2dk outputs
__global__ void QkvPack(const __nv_bfloat16* __restrict__ qkv,   // [T, 3d]
                        const __nv_bfloat16* __restrict__ p0,    // [T, d]
                        const __nv_bfloat16* __restrict__ u,     // [h, dk]
                        const __nv_bfloat16* __restrict__ v,     // [h, dk]
                        __nv_bfloat16* __restrict__ q2,          // [h, T, 2dk]
                        __nv_bfloat16* __restrict__ k2,          // [h, T, 2dk]
                        __nv_bfloat16* __restrict__ v2,          // [h, T, 2dk]
                        int T, int h, int dk, float scale) {
  // PDL: qkv/p0 均由前驱 kernel (GEMM / lp) 写出 —— 读取前等待其可见
  pdl_wait();
  const int d = h * dk;
  const int total = h * T * (dk >> 1);  // bf162 pairs per output
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (; idx < total; idx += stride) {
    const int j2 = idx % (dk >> 1);
    const int t = (idx / (dk >> 1)) % T;
    const int hh = idx / ((dk >> 1) * T);

    const int j = j2 * 2;                 // 0..dk-2
    const long orow = ((long)hh * T + t) * (2 * dk) + j;
    const __nv_bfloat16* qrow = qkv + (long)t * (3 * d);
    const __nv_bfloat16* prow = p0 + (long)t * d;

    // ---- q2: j<dk -> (q+u)*scale ; j>=dk -> (q+v)*scale
    {
      const int c0 = j, c1 = j + 1;                    // cols within q-part
      const __nv_bfloat162 qv0 =
          *reinterpret_cast<const __nv_bfloat162*>(qrow + hh * dk + c0);
      const __nv_bfloat162 uv0 =
          *reinterpret_cast<const __nv_bfloat162*>(u + hh * dk + c0);
      const __nv_bfloat162 qv1 = qv0;  // same q elements, pos_v bias below
      const __nv_bfloat162 uv1 =
          *reinterpret_cast<const __nv_bfloat162*>(v + hh * dk + c0);
      float lo = rnd_bf16(__bfloat162float(qv0.x) + __bfloat162float(uv0.x)) *
                 scale;
      float hi = rnd_bf16(__bfloat162float(qv0.y) + __bfloat162float(uv0.y)) *
                 scale;
      *reinterpret_cast<__nv_bfloat162*>(q2 + orow) =
          __floats2bfloat162_rn(lo, hi);
      lo = rnd_bf16(__bfloat162float(qv1.x) + __bfloat162float(uv1.x)) * scale;
      hi = rnd_bf16(__bfloat162float(qv1.y) + __bfloat162float(uv1.y)) * scale;
      *reinterpret_cast<__nv_bfloat162*>(q2 + orow + dk) =
          __floats2bfloat162_rn(lo, hi);
    }
    // ---- k2: j<dk -> k ; j>=dk -> p0[t, h*dk + (j-dk)]
    {
      const __nv_bfloat162 kv =
          *reinterpret_cast<const __nv_bfloat162*>(qrow + d + hh * dk + j);
      const __nv_bfloat162 pv =
          *reinterpret_cast<const __nv_bfloat162*>(prow + hh * dk + j);
      *reinterpret_cast<__nv_bfloat162*>(k2 + orow) = kv;
      *reinterpret_cast<__nv_bfloat162*>(k2 + orow + dk) = pv;
    }
    // ---- v2: j<dk -> v ; j>=dk -> 0
    {
      const __nv_bfloat162 vv =
          *reinterpret_cast<const __nv_bfloat162*>(qrow + 2 * d + hh * dk + j);
      *reinterpret_cast<__nv_bfloat162*>(v2 + orow) = vv;
      *reinterpret_cast<__nv_bfloat162*>(v2 + orow + dk) =
          __float2bfloat162_rn(0.f);
    }
  }
}

}  // namespace

void velox_qkv_pack(tvm::ffi::TensorView qkv, tvm::ffi::TensorView p0,
                    tvm::ffi::TensorView u, tvm::ffi::TensorView v,
                    tvm::ffi::TensorView q2, tvm::ffi::TensorView k2,
                    tvm::ffi::TensorView v2, double scale) {
  int64_t T = qkv.size(0), d = qkv.size(1) / 3;   // d = h * dk
  int64_t hh = u.size(0);
  int64_t dkk = d / hh;
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(qkv.device().device_type, qkv.device().device_id));

  const long total = (long)hh * T * (dkk >> 1);
  const int block = 256;
  const int grid = (int)((total + block - 1) / block);
  // PDL 启动: prologue 与前驱 (cublasLt GEMM / lp) 尾部重叠
  const bool ok = velox::hopper::launch_pdl(
      QkvPack, dim3(grid), dim3(block), 0, stream,
      static_cast<const __nv_bfloat16*>(qkv.data_ptr()),
      static_cast<const __nv_bfloat16*>(p0.data_ptr()),
      static_cast<const __nv_bfloat16*>(u.data_ptr()),
      static_cast<const __nv_bfloat16*>(v.data_ptr()),
      static_cast<__nv_bfloat16*>(q2.data_ptr()),
      static_cast<__nv_bfloat16*>(k2.data_ptr()),
      static_cast<__nv_bfloat16*>(v2.data_ptr()), (int)T, (int)hh, (int)dkk,
      (float)scale);
  TVM_FFI_ICHECK(ok) << "qkv_pack: PDL launch failed";
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(qkv_pack, velox_qkv_pack);
