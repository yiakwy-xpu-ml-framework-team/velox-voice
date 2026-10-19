/* Copyright 2026 veloxVoice authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

// NOTE (yiakwy) : instead of one MegaKernel for whole decode step, based on our analysis
// we split the whole graph into K1 (cross attn), K2 (GEMV + LN) and K3 (self-attn).
//
// The spliting is based on our proposed Rempping Technology in early years. It is apparent
// the a cross-attn shares different workloads, and need more blocks than other operations
// due to expensive cross blocks synchronization.
//
// See the pattn we profiled in 
//   benchmark/visualize_gemm_pipelining/hopper/sheetsage_decode_step_v2/sheetsage_decode_step_v2_phase_timeline.png

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "hopper/block/sheetsage_ar_block.h"

namespace {

using namespace velox::hopper::sheetsage_ar;

constexpr int kGemvBlock = 256;

void ss_gemv_impl(tvm::ffi::TensorView w, tvm::ffi::TensorView b,
                  tvm::ffi::TensorView x, tvm::ffi::TensorView y,
                  int64_t N, int64_t K, int64_t blocks) {
  cudaStream_t st = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(w.device().device_type, w.device().device_id));
  const bf16* bp = b.data_ptr() ? static_cast<const bf16*>(b.data_ptr())
                                : nullptr;
  ss_gemv_kernel<kGemvBlock><<<(unsigned)blocks, kGemvBlock, 0, st>>>(
      static_cast<const bf16*>(w.data_ptr()), bp,
      static_cast<const bf16*>(x.data_ptr()),
      static_cast<bf16*>(y.data_ptr()), (int)N, (int)K);
}

void ss_ln_res_impl(tvm::ffi::TensorView h, tvm::ffi::TensorView res,
                    tvm::ffi::TensorView w, tvm::ffi::TensorView b,
                    double eps) {
  cudaStream_t st = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(w.device().device_type, w.device().device_id));
  const bf16* rp = res.data_ptr() ? static_cast<const bf16*>(res.data_ptr())
                                  : nullptr;
  ss_ln_res_kernel<256, 512><<<1, 256, 0, st>>>(
      static_cast<bf16*>(h.data_ptr()), rp,
      static_cast<const bf16*>(w.data_ptr()),
      static_cast<const bf16*>(b.data_ptr()), (float)eps);
}

void ss_self_attn_impl(tvm::ffi::TensorView q, tvm::ffi::TensorView K,
                       tvm::ffi::TensorView V, tvm::ffi::TensorView out,
                       tvm::ffi::TensorView rows_dev, double scale,
                       int64_t heads, int64_t S_) {
  cudaStream_t st = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(q.device().device_type, q.device().device_id));
  ss_self_attn_kernel<32><<<(unsigned)heads, 32, 0, st>>>(
      static_cast<const bf16*>(q.data_ptr()),
      static_cast<const bf16*>(K.data_ptr()),
      static_cast<const bf16*>(V.data_ptr()),
      static_cast<bf16*>(out.data_ptr()),
      (int)S_, static_cast<const int*>(rows_dev.data_ptr()), (float)scale);
}

void ss_kvstore_impl(tvm::ffi::TensorView qkv, tvm::ffi::TensorView K,
                     tvm::ffi::TensorView V, int64_t S, tvm::ffi::TensorView pos_dev) {
  cudaStream_t st = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(qkv.device().device_type, qkv.device().device_id));
  ss_kvstore_kernel<256><<<(512 + 255) / 256, 256, 0, st>>>(
      static_cast<const bf16*>(qkv.data_ptr()),
      static_cast<bf16*>(K.data_ptr()), static_cast<bf16*>(V.data_ptr()),
      (int)S, static_cast<const int*>(pos_dev.data_ptr()));
}

void ss_lmhead_impl(tvm::ffi::TensorView lm, tvm::ffi::TensorView h,
                    tvm::ffi::TensorView allowed, tvm::ffi::TensorView pbest,
                    tvm::ffi::TensorView pidx, int64_t V, int64_t K,
                    int64_t blocks) {
  cudaStream_t st = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(lm.device().device_type, lm.device().device_id));
  ss_lmhead_kernel<kGemvBlock><<<(unsigned)blocks, kGemvBlock, 0, st>>>(
      static_cast<const bf16*>(lm.data_ptr()),
      static_cast<const bf16*>(h.data_ptr()),
      static_cast<const unsigned char*>(allowed.data_ptr()),
      static_cast<float*>(pbest.data_ptr()),
      static_cast<int*>(pidx.data_ptr()), (int)V, (int)K);
}


void ss_gemv_ln_impl(tvm::ffi::TensorView w, tvm::ffi::TensorView b,
                     tvm::ffi::TensorView h, tvm::ffi::TensorView res,
                     tvm::ffi::TensorView lnw, tvm::ffi::TensorView lnb,
                     tvm::ffi::TensorView y, int64_t N, int64_t K,
                     double eps, int64_t do_gelu, int64_t blocks) {
  cudaStream_t st = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(w.device().device_type, w.device().device_id));
  size_t smem = (size_t)K * 2;
  ss_gemv_ln_kernel<kGemvBlock><<<(unsigned)blocks, kGemvBlock, smem, st>>>(
      static_cast<const bf16*>(w.data_ptr()),
      static_cast<const bf16*>(b.data_ptr()),
      static_cast<bf16*>(h.data_ptr()),
      res.data_ptr() ? static_cast<const bf16*>(res.data_ptr()) : nullptr,
      static_cast<const bf16*>(lnw.data_ptr()),
      static_cast<const bf16*>(lnb.data_ptr()),
      static_cast<bf16*>(y.data_ptr()), (int)N, (int)K, (float)eps,
      (int)do_gelu);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(ss_gemv_ln, ss_gemv_ln_impl);

void ss_self_attn_split_impl(tvm::ffi::TensorView q, tvm::ffi::TensorView K,
                             tvm::ffi::TensorView V, tvm::ffi::TensorView pm,
                             tvm::ffi::TensorView pl, tvm::ffi::TensorView pacc,
                             tvm::ffi::TensorView out,
                             tvm::ffi::TensorView pos_dev, int64_t S, int64_t split,
                             int64_t heads, double scale) {
  cudaStream_t st = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(q.device().device_type, q.device().device_id));
  dim3 g((unsigned)heads, (unsigned)split);
  constexpr int kAttnWarps = 32;             // BLOCK = 1024 threads
  ss_self_attn_split_kernel<kAttnWarps><<<g, kAttnWarps * 32, 0, st>>>(
      static_cast<const bf16*>(q.data_ptr()),
      static_cast<const bf16*>(K.data_ptr()),
      static_cast<const bf16*>(V.data_ptr()),
      static_cast<float*>(pm.data_ptr()), static_cast<float*>(pl.data_ptr()),
      static_cast<float*>(pacc.data_ptr()), (int)S,
      static_cast<const int*>(pos_dev.data_ptr()), (int)split, (float)scale);
  int total = (int)heads * kD;
  ss_attn_merge_kernel<256><<<(total + 255) / 256, 256, 0, st>>>(
      static_cast<const float*>(pm.data_ptr()),
      static_cast<const float*>(pl.data_ptr()),
      static_cast<const float*>(pacc.data_ptr()),
      static_cast<bf16*>(out.data_ptr()), (int)split, (int)heads);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(ss_self_attn_split, ss_self_attn_split_impl);

void ss_gemv_row_impl(tvm::ffi::TensorView w, tvm::ffi::TensorView b,
                      tvm::ffi::TensorView x, tvm::ffi::TensorView y,
                      int64_t N, int64_t K) {
  cudaStream_t st = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(w.device().device_type, w.device().device_id));
  ss_gemv_row_kernel<256><<<(unsigned)N, 256, 0, st>>>(
      static_cast<const bf16*>(w.data_ptr()),
      static_cast<const bf16*>(b.data_ptr()),
      static_cast<const bf16*>(x.data_ptr()),
      static_cast<bf16*>(y.data_ptr()), (int)N, (int)K);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(ss_gemv_row, ss_gemv_row_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ss_gemv, ss_gemv_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ss_ln_res, ss_ln_res_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ss_self_attn, ss_self_attn_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ss_kvstore, ss_kvstore_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ss_lmhead, ss_lmhead_impl);

}  // anonymous namespace
