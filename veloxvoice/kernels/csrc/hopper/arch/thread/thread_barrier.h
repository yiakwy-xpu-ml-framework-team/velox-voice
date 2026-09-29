#pragma once

namespace velox {
namespace hopper {

__device__ __forceinline__ void thread_barrier(int barrier_id, int count) {
  asm volatile("barrier.sync %0, %1;" ::"r"(barrier_id), "r"(count) : "memory");
}

}  // namespace hopper
}  // namespace velox
