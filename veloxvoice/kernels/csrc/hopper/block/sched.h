/* Copyright 2026 veloxVoice authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

namespace velox {
namespace hopper {

template <int GROUP_SIZE_M>
__host__ __device__ __forceinline__ void swizzle2d(
    int tile, int m_tiles, int n_tiles, int* m_tile, int* n_tile) {
  const int group = GROUP_SIZE_M * n_tiles;
  const int gid = tile / group;
  const int first_m = gid * GROUP_SIZE_M;
  const int gm = min(m_tiles - first_m, GROUP_SIZE_M);
  const int in_group = tile % group;
  *m_tile = first_m + in_group % gm;
  *n_tile = in_group / gm;
}

}  // namespace hopper
}  // namespace velox
