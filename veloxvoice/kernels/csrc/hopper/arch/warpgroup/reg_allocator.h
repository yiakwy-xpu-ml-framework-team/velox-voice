#pragma once

namespace velox {
namespace hopper {

template <int N>
__device__ __forceinline__ void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N));
}

template <int N>
__device__ __forceinline__ void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N));
}

}  // namespace hopper
}  // namespace velox
