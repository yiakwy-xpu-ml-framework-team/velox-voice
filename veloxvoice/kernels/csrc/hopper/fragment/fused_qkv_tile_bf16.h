#pragma once

// NOTE (yiakwy) : tested for kernels other than fused_qkv

#include "wgmma_accumulator_bf16.h"

#ifndef FQKV_TILE_N
#define FQKV_TILE_N 192
#endif

namespace velox {
namespace hopper {
namespace fused_qkv {

template <int TILE_N>
struct FusedQkvTile;

template <>
struct FusedQkvTile<192> {
  static constexpr int bn = 192;
  static constexpr int nstages = 4;
  static constexpr int producer_threads = 128;
  using Accum = WgmmaM64N192Frag;
};

template <>
struct FusedQkvTile<256> {
  static constexpr int bn = 256;
  static constexpr int nstages = 3;
  static constexpr int producer_threads = 128;
  using Accum = WgmmaM64N256Frag;
};

#if FQKV_TILE_N == 192
using FusedQkvTileConfig = FusedQkvTile<192>;
#elif FQKV_TILE_N == 256
using FusedQkvTileConfig = FusedQkvTile<256>;
#else
#error "fused-qkv supports only FQKV_TILE_N=192 or 256"
#endif

using FqkvAccum = typename FusedQkvTileConfig::Accum;
constexpr int FQKV_TILE_BN = FusedQkvTileConfig::bn;
constexpr int FQKV_TILE_NSTAGES = FusedQkvTileConfig::nstages;
constexpr int FQKV_TILE_PRODUCER_THREADS =
    FusedQkvTileConfig::producer_threads;

}  // namespace fused_qkv
}  // namespace hopper
}  // namespace velox
