#pragma once

#include <cuda_runtime.h>
#include <cstdlib>

#ifndef VELOXVOICE_ENABLE_PDL
#define VELOXVOICE_ENABLE_PDL 0
#endif

#if VELOXVOICE_ENABLE_PDL && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void pdl_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;");
}
#else
__device__ __forceinline__ void pdl_wait() {}
__device__ __forceinline__ void pdl_launch_dependents() {}
#endif

namespace velox {
namespace hopper {

template <typename KernelT, typename... Args>
inline bool launch_pdl_cluster(KernelT kernel, dim3 grid, dim3 block,
                               dim3 cluster, size_t smem, cudaStream_t stream,
                               Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr = {};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {cluster.x, cluster.y, cluster.z};
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, kernel, args...) == cudaSuccess;
}

template <typename KernelT, typename... Args>
inline bool launch_pdl(KernelT kernel, dim3 grid, dim3 block, size_t smem,
                       cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  static const bool pdl_enabled = [] {
#if VELOXVOICE_ENABLE_PDL
    const char* v = ::getenv("VELOXVOICE_DISABLE_PDL");
    return !(v && v[0] == '1');
#else
    return false;
#endif
  }();
  if (pdl_enabled) {
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
  }
  return cudaLaunchKernelEx(&cfg, kernel, args...) == cudaSuccess;
}

}  // namespace hopper
}  // namespace velox
