/* Copyright 2026 veloxVoice authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "hopper/block/fused_qkv_block_1p2c.h"

namespace velox {
namespace hopper {
namespace fused_qkv {

__global__ void __launch_bounds__(N_THREADS, 1)
    hopper_fused_qkv_kernel_entry(const __nv_bfloat16* __restrict__ x,
                       const __nv_bfloat16* __restrict__ w,
                       const __nv_bfloat16* __restrict__ b,
                       __nv_bfloat16* __restrict__ out, int M,
                       const __grid_constant__ CUtensorMap tmap_x,
                       const __grid_constant__ CUtensorMap tmap_w,
                       const __grid_constant__ CUtensorMap tmap_out,
                       uint64_t* trace, int split_k, int split_start) {
  run_fused_qkv_persistent_splitk_pipeline(x, w, b, out, M, &tmap_x, &tmap_w,
                                           &tmap_out, trace, split_k,
                                           split_start);
}

inline bool launch_impl(
    const void* x, const void* w, const void* b, void* out, int M,
    cudaStream_t stream, int split_k, uint64_t* trace, int trace_captures) {
  if (M < 64) return false;

  CUtensorMap tma_x_desc, tma_w_desc, tma_o_desc;
  if (!make_tma_desc_2d_bf16(&tma_x_desc, x, K_BLOCK, static_cast<uint64_t>(M),
                             static_cast<uint64_t>(K_BLOCK) * 2, BK, BM))
    return false;
  if (!make_tma_desc_2d_bf16(&tma_w_desc, w, K_BLOCK,
                             static_cast<uint64_t>(N_OUT),
                             static_cast<uint64_t>(K_BLOCK) * 2, BK, BN))
    return false;
  if (!make_tma_desc_2d_bf16_noswizzle(
          &tma_o_desc, out, static_cast<uint64_t>(N_OUT), static_cast<uint64_t>(M),
          static_cast<uint64_t>(N_OUT) * 2, 32, 64))
    return false;

  const int m_tiles = (M + BM - 1) / BM;
  const int n_tiles = (N_OUT + BN - 1) / BN;
  const int logical_tiles = m_tiles * n_tiles;

  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);

  const int smem =
      NSTAGES * STAGE_BYTES + EPI_BYTES + 2 * NSTAGES * 8 + 1024;
  cudaFuncSetAttribute(hopper_fused_qkv_kernel_entry,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(hopper_fused_qkv_kernel_entry,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  if constexpr (FQKV_STREAM_K) {

    int grid_mn = min(logical_tiles, sms);

    if (trace_captures > 0 && grid_mn > trace_captures)
      grid_mn = trace_captures;

    int stream_split_k = 1;
    if (logical_tiles < sms) {
      const int max_split_by_occupancy = max(1, sms / grid_mn);
      stream_split_k = min(split_k, max_split_by_occupancy);
    }

    const dim3 grid(stream_split_k, grid_mn);
    const dim3 block(N_THREADS);
    const dim3 cluster(stream_split_k, 1, 1);

    return launch_pdl_cluster(
        hopper_fused_qkv_kernel_entry, grid, block, cluster,
        smem, stream,
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(w),
        static_cast<const __nv_bfloat16*>(b),
        static_cast<__nv_bfloat16*>(out), M,
        tma_x_desc, tma_w_desc, tma_o_desc,
        trace, stream_split_k, 0);

  } else {
    // launching w/o clusters
    int grid_mn = min(logical_tiles, sms);

    if (trace_captures > 0 && grid_mn > trace_captures)
        grid_mn = trace_captures;

    const dim3 grid(grid_mn);
    const dim3 block(N_THREADS);

    return launch_pdl(
        hopper_fused_qkv_kernel_entry, grid, block, smem, stream,
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(w),
        static_cast<const __nv_bfloat16*>(b),
        static_cast<__nv_bfloat16*>(out), M,
        tma_x_desc, tma_w_desc, tma_o_desc,
        trace, split_k, 0);
  }
}

inline bool launch(const void* x, const void* w, const void* b, void* out,
                   int M, cudaStream_t stream, int split_k = 1,
                   uint64_t* trace = nullptr) {
  return launch_impl(x, w, b, out, M, stream, split_k, trace, 0);
}

inline bool launch_trace(const void* x, const void* w, const void* b,
                         void* out, int M, cudaStream_t stream,
                         uint64_t* trace, int trace_captures) {
  return launch_impl(x, w, b, out, M, stream, 1, trace, trace_captures);
}

}  // namespace fused_qkv
}  // namespace hopper
}  // namespace velox

void velox_fused_qkv_wgmma(tvm::ffi::TensorView x, tvm::ffi::TensorView w,
                           tvm::ffi::TensorView b, tvm::ffi::TensorView out) {
  namespace vx = velox;

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));

  if (x.dtype().bits == 16) {
    if (vx::hopper::fused_qkv::launch(
            x.data_ptr(), w.data_ptr(), b.data_ptr(), out.data_ptr(),
            static_cast<int>(x.size(0)), stream))
      return;
  }
  TVM_FFI_ICHECK(false) << "fused_qkv_wgmma: unsupported";
}


void velox_fused_qkv_wgmma_streamk(
    tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView b,
    tvm::ffi::TensorView out, int64_t split_k) {
  namespace vx = velox;

#if FQKV_STREAM_K
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
  if (x.dtype().bits == 16 &&
      vx::hopper::fused_qkv::launch(
          x.data_ptr(), w.data_ptr(), b.data_ptr(), out.data_ptr(),
          static_cast<int>(x.size(0)), stream, static_cast<int>(split_k)))
    return;
#endif
  TVM_FFI_ICHECK(false) << "fused_qkv_wgmma_streamk: unsupported";
}

// NOTE (yiakwy) : Do Not use it in production code
void velox_fused_qkv_wgmma_phase_trace(
    tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView b,
    tvm::ffi::TensorView out, tvm::ffi::TensorView trace,
    int64_t trace_captures) {
  namespace vx = velox;

#if FQKV_PHASE_TRACE
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
  if (x.dtype().bits == 16 &&
      vx::hopper::fused_qkv::launch_trace(
          x.data_ptr(), w.data_ptr(), b.data_ptr(), out.data_ptr(),
          static_cast<int>(x.size(0)), stream,
          reinterpret_cast<uint64_t*>(trace.data_ptr()),
          static_cast<int>(trace_captures)))
    return;
#endif
  TVM_FFI_ICHECK(false) << "fused_qkv_wgmma_phase_trace: unsupported";
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_qkv_wgmma, velox_fused_qkv_wgmma);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_qkv_wgmma_streamk,
                              velox_fused_qkv_wgmma_streamk);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_qkv_wgmma_phase_trace,
                              velox_fused_qkv_wgmma_phase_trace);
