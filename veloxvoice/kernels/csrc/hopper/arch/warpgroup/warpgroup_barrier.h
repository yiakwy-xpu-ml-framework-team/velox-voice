#pragma once

namespace velox {
namespace hopper {

__device__ __forceinline__ void warpgroup_barrier(int barrier_id) {
  asm volatile("barrier.sync %0, 128;" ::"r"(barrier_id) : "memory");
}

}  // namespace hopper
}  // namespace velox
