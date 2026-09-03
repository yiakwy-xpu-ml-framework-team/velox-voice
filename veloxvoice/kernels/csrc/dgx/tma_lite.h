/* VeloxVoice — minimal TMA helpers for DGX-Spark sm_121a (Apache-2.0). */
#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

namespace nvgpu { namespace arch {

inline CUresult make_2d_u8_desc(CUtensorMap* desc, const void* base,
                                uint64_t inner, uint64_t outer,
                                uint32_t box_inner, uint32_t box_outer,
                                uint64_t global_stride_inner_bytes) {
  uint64_t shape[2] = {inner, outer};
  uint64_t stride[1] = {global_stride_inner_bytes};
  uint32_t box[2] = {box_inner, box_outer};
  uint32_t step[2] = {1, 1};
  return cuTensorMapEncodeTiled(
      desc, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base),
      shape, stride, box, step, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

__device__ inline void tma_load_2d_bytes(const CUtensorMap* desc, void* dst,
                                         int32_t off_i, int32_t off_o, uint64_t* bar) {
  const uint32_t dst_smem = (uint32_t)__cvta_generic_to_shared(dst);
  const uint64_t bar_smem = (uint64_t)__cvta_generic_to_shared(bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n"
      :: "r"(dst_smem), "l"(desc), "r"(off_i), "r"(off_o), "r"((uint32_t)bar_smem)
      : "memory");
}

}}  // nvgpu::arch
