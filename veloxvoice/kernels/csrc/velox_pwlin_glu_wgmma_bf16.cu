/* Copyright 2026 veloxVoice authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

// NOTE (yiakwy) : velox_pwlin_glu_wgmma_bf16.cu — sm_90a entry for the fused pwlin + silu-GLU
#include <tvm/ffi/tvm_ffi.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "hopper/arch/pdl_sm90.h"
#include "hopper/block/pwlin_glu_block_1p2c.h"

namespace velox {
namespace hopper {
namespace pwlin_glu {

#define PWLIN_KERNEL_CONCAT_(name, tile) name##_##tile
#define PWLIN_KERNEL_CONCAT(name, tile) PWLIN_KERNEL_CONCAT_(name, tile)
#define PWLIN_TILE_CFG_CONCAT_(m, n) m##x##n
#define PWLIN_TILE_CFG_CONCAT(m, n) PWLIN_TILE_CFG_CONCAT_(m, n)
#define PWLIN_TILE_CFG PWLIN_TILE_CFG_CONCAT(PWLIN_TILE_M, PWLIN_TILE_N)
#define PWLIN_KERNEL_NAME PWLIN_KERNEL_CONCAT(PwlinGluPersistent, PWLIN_TILE_CFG)

__global__ void __launch_bounds__(N_THREADS, 1)
    PWLIN_KERNEL_NAME(const __nv_bfloat16* __restrict__ x,
                       const __nv_bfloat16* __restrict__ w,
                       const __nv_bfloat16* __restrict__ b,
                       __nv_bfloat16* __restrict__ out, int M,
                       const __grid_constant__ CUtensorMap tmap_x,
                       const __grid_constant__ CUtensorMap tmap_w) {
  run_pwlin_glu(x, w, b, out, M, &tmap_x, &tmap_w);
}

inline int smem_bytes() {
  return NSTAGES * STAGE_BYTES + 2 * NSTAGES * 8 + 1024;
}

inline bool launch_pwlin_glu_bf16(const void* x, const void* w, const void* b,
                                  void* out, int M, int K, cudaStream_t stream) {
  if (M < 64 || K != K_BLOCK) return false;

  // TMA tensor maps: innermost dim = K (contiguous), box = {BK, BM/BN}.
  CUtensorMap tmap_x, tmap_w;
  if (!make_tma_desc_2d_bf16(&tmap_x, x, K_BLOCK, (uint64_t)M,
                             (uint64_t)K_BLOCK * 2, BK, BM))
    return false;
  if (!make_tma_desc_2d_bf16(&tmap_w, w, K_BLOCK, 2 * N_HALF,
                             (uint64_t)K_BLOCK * 2, BK, BN))
    return false;

  const int n_tiles = N_HALF / BN;
  const int m_tiles = (M + BM - 1) / BM;

  const int total_tiles = m_tiles * n_tiles;

  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

  const int grid = total_tiles < sms ? total_tiles : sms;

  const int smem = smem_bytes();
  cudaError_t e = cudaFuncSetAttribute(
      PWLIN_KERNEL_NAME, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return false;
  return hopper::launch_pdl(
      PWLIN_KERNEL_NAME, dim3(grid), dim3(N_THREADS), smem, stream,
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(w),
      static_cast<const __nv_bfloat16*>(b), static_cast<__nv_bfloat16*>(out),
      M, tmap_x, tmap_w);
}

}  // namespace pwlin_glu
}  // namespace hopper
}  // namespace velox

void velox_pwlin_glu(tvm::ffi::TensorView x, tvm::ffi::TensorView w,
                     tvm::ffi::TensorView b, tvm::ffi::TensorView out) {
  int64_t M = x.size(0), K = x.size(1);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));

  if (x.dtype().bits == 16) {
    if (velox::hopper::pwlin_glu::launch_pwlin_glu_bf16(
            x.data_ptr(), w.data_ptr(), b.data_ptr(), out.data_ptr(), (int)M,
            (int)K, stream)) {
      return;
    }
  }
  TVM_FFI_ICHECK(false) << "pwlin_glu: unsupported arch/shape — python "
                           "wrapper should fall back to F.linear + glu";
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(pwlin_glu, velox_pwlin_glu);
