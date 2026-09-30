#pragma once

#include <cstdint>

namespace velox {
namespace hopper {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)));
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n"
      ".reg .pred P;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
      "@!P bra WAIT_%=;\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(phase));
}

__device__ __forceinline__ void cp_async16(void* sdst, const void* gsrc) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(sdst)),
               "l"(gsrc));
}

__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* bar) {
  asm volatile(
      "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(
          smem_u32(bar)));
}

__device__ __forceinline__ void cluster_barrier() {
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

__device__ __forceinline__ void mbar_init_release_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

}  // namespace hopper
}  // namespace velox
