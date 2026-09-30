#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "mbarrier_sm90.h"

namespace velox {
namespace hopper {

inline bool make_tma_desc_2d_bf16(CUtensorMap* tmap, const void* base,
                                  uint64_t inner, uint64_t outer,
                                  uint64_t stride_bytes, uint32_t box_inner,
                                  uint32_t box_outer) {
  uint64_t gdim[2] = {inner, outer};
  uint64_t gstride[1] = {stride_bytes};
  uint32_t box[2] = {box_inner, box_outer};
  uint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), gdim,
      gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS;
}

inline bool make_tma_desc_2d_bf16_noswizzle(CUtensorMap* tmap, const void* base,
                                             uint64_t inner, uint64_t outer,
                                             uint64_t stride_bytes,
                                             uint32_t box_inner,
                                             uint32_t box_outer) {
  uint64_t gdim[2] = {inner, outer};
  uint64_t gstride[1] = {stride_bytes};
  uint32_t box[2] = {box_inner, box_outer};
  uint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), gdim,
      gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS;
}

inline bool make_tma_desc_2d_bf16_l2(CUtensorMap* tmap, const void* base,
                                    uint64_t inner, uint64_t outer,
                                    uint64_t stride_bytes, uint32_t box_inner,
                                    uint32_t box_outer) {
  uint64_t gdim[2] = {inner, outer};
  uint64_t gstride[1] = {stride_bytes};
  uint32_t box[2] = {box_inner, box_outer};
  uint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), gdim,
      gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS;
}

inline bool make_tma_desc_2d_bf16_noswizzle_l2(
    CUtensorMap* tmap, const void* base, uint64_t inner, uint64_t outer,
    uint64_t stride_bytes, uint32_t box_inner, uint32_t box_outer) {
  uint64_t gdim[2] = {inner, outer};
  uint64_t gstride[1] = {stride_bytes};
  uint32_t box[2] = {box_inner, box_outer};
  uint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), gdim,
      gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS;
}

inline bool make_tma_desc_3d_bf16_stmatrix(CUtensorMap* tmap, const void* base,
                                            uint64_t inner, uint64_t outer,
                                            uint64_t outer_tiles) {
  uint64_t gdim[3] = {inner, outer, outer_tiles};
  uint64_t gstride[2] = {outer_tiles * inner * 2, inner * 2};
  uint32_t box[3] = {static_cast<uint32_t>(inner), 64, 2};
  uint32_t estride[3] = {1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(base), gdim,
      gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS;
}

__device__ __forceinline__ void tma_expect_bytes(uint64_t* bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
          smem_u32(bar)),
      "r"(bytes));
}

__device__ __forceinline__ void tma2d_load_async(const void* smem,
                                                 const CUtensorMap* tmap,
                                                 uint64_t* bar, int32_t c0,
                                                 int32_t c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(smem)),
      "l"(reinterpret_cast<uint64_t>(tmap)), "r"(c0), "r"(c1),
      "r"(smem_u32(bar))
      : "memory");
}

__device__ __forceinline__ void tma2d_load_async_hint(
    const void* smem, const CUtensorMap* tmap, uint64_t* bar, int32_t c0,
    int32_t c1, uint64_t cache_hint) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;" ::
          "r"(smem_u32(smem)), "l"(reinterpret_cast<uint64_t>(tmap)),
          "r"(smem_u32(bar)), "r"(c0), "r"(c1), "l"(cache_hint)
      : "memory");
}

__device__ __forceinline__ void tma2d_load_async_multicast2(
    const void* smem, const CUtensorMap* tmap, uint64_t* bar, int32_t c0,
    int32_t c1) {
  const uint64_t cache_hint = 0x1000000000000000ull;
  const uint16_t mask = 3;
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, "
      "%6;" ::
          "r"(smem_u32(smem)), "l"(reinterpret_cast<uint64_t>(tmap)), "r"(
              smem_u32(bar)),
          "h"(mask), "r"(c0), "r"(c1), "l"(cache_hint)
      : "memory");
}

__device__ __forceinline__ void tma2d_store_async(const CUtensorMap* tmap,
                                                  const void* smem, int32_t c0,
                                                  int32_t c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
      " [%0, {%2, %3}], [%1];" ::"l"(reinterpret_cast<uint64_t>(tmap)),
      "r"(smem_u32(smem)), "r"(c0), "r"(c1)
      : "memory");
}

__device__ __forceinline__ void tma2d_store_async_bulk(
    const CUtensorMap* tmap, const void* smem, int32_t c0, int32_t c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
      " [%0, {%2, %3}], [%1];" ::"l"(reinterpret_cast<uint64_t>(tmap)),
      "r"(smem_u32(smem)), "r"(c0), "r"(c1)
      : "memory");
}

__device__ __forceinline__ void tma3d_store_async_stmatrix(
    const CUtensorMap* tmap, const void* smem, int32_t c1, int32_t c2) {
  asm volatile(
      "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
      " [%0, {%2, %3, %4}], [%1];" ::"l"(reinterpret_cast<uint64_t>(tmap)),
      "r"(smem_u32(smem)), "n"(0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

}  // namespace hopper
}  // namespace velox
