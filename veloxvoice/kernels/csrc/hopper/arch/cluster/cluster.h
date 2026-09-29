#pragma once

#include <cstdint>

namespace velox {
namespace hopper {

__device__ __forceinline__ uint32_t cluster_map_shared(uint32_t address,
                                                        uint32_t cta_rank) {
  uint32_t mapped;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(mapped)
               : "r"(address), "r"(cta_rank));
  return mapped;
}

}  // namespace hopper
}  // namespace velox
