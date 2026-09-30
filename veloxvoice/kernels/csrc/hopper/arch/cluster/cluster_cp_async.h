#pragma once

#include "cluster.h"

namespace velox {
namespace hopper {

__device__ __forceinline__ void cluster_cp_async_bulk(
    void* dst_local_smem, const void* src_remote_smem, uint32_t bytes,
    uint64_t* mbar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"r"(
          static_cast<uint32_t>(__cvta_generic_to_shared(dst_local_smem))),
      "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src_remote_smem))),
      "r"(bytes),
      "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar)))
      : "memory");
}

}  // namespace hopper
}  // namespace velox
