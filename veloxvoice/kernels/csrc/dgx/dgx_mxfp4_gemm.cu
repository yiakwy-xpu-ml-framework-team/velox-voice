/* VeloxVoice — dgx_mxfp4_gemm.cu — packed nvfp4/mxfp4 GEMM for DGX-Spark (sm_121a). */
#include <cuda.h>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 1200

#include "nv_block_wasp_mxfp4.h"

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace {

constexpr int K_BM = 128;
constexpr int K_BN = 128;
constexpr int K_BK = 64;
constexpr int K_STAGES = 4;
constexpr int CLUSTER_M = 2;
constexpr int THREADS = (8 + 1) * 32;  // producer warp + 8 consumer warps

extern "C" __global__ void __cluster_dims__(CLUSTER_M, 1, 1)
__launch_bounds__(THREADS)
mxfp4_gemm_kernel_entry(const __grid_constant__ CUtensorMap tma_A,
                        const __grid_constant__ CUtensorMap tma_B,
                        const uint8_t* __restrict__ scale_A,
                        const uint8_t* __restrict__ scale_B,
                        float* __restrict__ Out,
                        int M, int N, int K, int num_blocks_m, int num_blocks_n) {
  extern __shared__ __align__(128) uint8_t smem_buffer[];
  xpu::HopperPersistentMxfp4Pipeline<K_BM, K_BN, K_BK, K_STAGES, 16, CLUSTER_M>
      ::run_persistent(&tma_A, &tma_B, scale_A, scale_B, Out, M, N, K,
                       num_blocks_m, num_blocks_n, smem_buffer);
}

}  // namespace

void velox_dgx_mxfp4_gemm(tvm::ffi::TensorView packA, tvm::ffi::TensorView packB,
                          tvm::ffi::TensorView scaleA, tvm::ffi::TensorView scaleB,
                          tvm::ffi::TensorView out) {
  int64_t M = packA.size(0), K2 = packA.size(1), K = K2 * 2;
  int64_t N = packB.size(0);
  TVM_FFI_ICHECK_EQ(K % K_BK, 0) << "K must be divisible by BK=" << K_BK;
  TVM_FFI_ICHECK_EQ(M % K_BM, 0);
  TVM_FFI_ICHECK_EQ(N % K_BN, 0);

  static CUtensorMap desc_A, desc_B;
  static int64_t cached_M = -1, cached_N = -1, cached_K = -1;
  if (M != cached_M || N != cached_N || K != cached_K) {
    auto rA = nvgpu::arch::make_2d_u8_desc(&desc_A, packA.data_ptr(), K2, M, K_BK / 2,
                                           K_BM, K2);
    auto rB = nvgpu::arch::make_2d_u8_desc(&desc_B, packB.data_ptr(), K2, N, K_BK / 2,
                                           K_BN, K2);
    TVM_FFI_ICHECK(rA == CUDA_SUCCESS && rB == CUDA_SUCCESS);
    cached_M, cached_N, cached_K = M, N, K;
  }
  int nb_m = (M + K_BM - 1) / K_BM, nb_n = (N + K_BN - 1) / K_BN;
  int nb = nb_m * nb_n;
  nb = (nb + CLUSTER_M - 1) / CLUSTER_M * CLUSTER_M;  // grid multiple of cluster dims
  size_t smem = sizeof(xpu::HopperPersistentMxfp4Pipeline<K_BM, K_BN, K_BK, K_STAGES,
                                                          16, CLUSTER_M>::SmemLayout);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(packA.device().device_type, packA.device().device_id));
  cudaFuncSetAttribute(mxfp4_gemm_kernel_entry,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  mxfp4_gemm_kernel_entry<<<nb, THREADS, smem, stream>>>(
      desc_A, desc_B, (const uint8_t*)scaleA.data_ptr(), (const uint8_t*)scaleB.data_ptr(),
      (float*)out.data_ptr(), (int)M, (int)N, (int)K, nb_m, nb_n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dgx_mxfp4_gemm, velox_dgx_mxfp4_gemm);

#else  // fallback so the module-building path stays guard-compatible

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

void velox_dgx_mxfp4_gemm_stub(tvm::ffi::TensorView packA, tvm::ffi::TensorView packB,
                               tvm::ffi::TensorView scaleA, tvm::ffi::TensorView scaleB,
                               tvm::ffi::TensorView out) {
  TVM_FFI_THROW("dgx_mxfp4_gemm requires CUDA_VERSION >= 1200");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dgx_mxfp4_gemm, velox_dgx_mxfp4_gemm_stub);

#endif
