#pragma once

#include "../arch/wgmma/gmma_sm90.h"

namespace velox {
namespace hopper {

struct WgmmaM64N64Frag {
  float r[32];

  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int i = 0; i < 32; ++i) r[i] = 0.f;
  }

  __device__ __forceinline__ void mma(uint64_t desc_a, uint64_t desc_b,
                                      uint32_t scale_d) {
    wgmma_m64n64k16_bf16(r, desc_a, desc_b, scale_d);
  }

  __device__ __forceinline__ void get_mn(int i, int warp_in_wg, int lane,
                                         int* row, int* col) const {
    const int group = lane >> 2;
    const int tig = lane & 3;
    const int c = i >> 2;
    const int rem = i & 3;
    const int h = rem >> 1;
    const int cs = rem & 1;
    *row = warp_in_wg * 16 + group + 8 * h;
    *col = c * 8 + tig * 2 + cs;
  }
};

struct WgmmaM64N128Frag {
  float r[64];

  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int i = 0; i < 64; ++i) r[i] = 0.f;
  }

  __device__ __forceinline__ void mma(uint64_t desc_a, uint64_t desc_b,
                                      uint32_t scale_d) {
    wgmma_m64n128k16_bf16(r, desc_a, desc_b, scale_d);
  }

  __device__ __forceinline__ void get_mn(int i, int warp_in_wg, int lane,
                                         int* row, int* col) const {
    const int group = lane >> 2;
    const int tig = lane & 3;
    const int c = i >> 2;
    const int rem = i & 3;
    const int h = rem >> 1;
    const int cs = rem & 1;
    *row = warp_in_wg * 16 + group + 8 * h;
    *col = c * 8 + tig * 2 + cs;
  }
};

struct WgmmaM64N192Frag {
  float r[96];

  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int i = 0; i < 96; ++i) r[i] = 0.f;
  }

  __device__ __forceinline__ void mma(uint64_t desc_a, uint64_t desc_b,
                                      uint32_t scale_d) {
    wgmma_m64n192k16_bf16(r, desc_a, desc_b, scale_d);
  }

  __device__ __forceinline__ void get_mn(int i, int warp_in_wg, int lane,
                                         int* row, int* col) const {
    const int group = lane >> 2;
    const int tig = lane & 3;
    const int c = i >> 2;
    const int rem = i & 3;
    const int h = rem >> 1;
    const int cs = rem & 1;
    *row = warp_in_wg * 16 + group + 8 * h;
    *col = c * 8 + tig * 2 + cs;
  }
};

struct WgmmaM64N224Frag {
  float r[112];

  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int i = 0; i < 112; ++i) r[i] = 0.f;
  }

  __device__ __forceinline__ void mma(uint64_t desc_a, uint64_t desc_b,
                                      uint32_t scale_d) {
    wgmma_m64n224k16_bf16(r, desc_a, desc_b, scale_d);
  }

  __device__ __forceinline__ void get_mn(int i, int warp_in_wg, int lane,
                                         int* row, int* col) const {
    const int group = lane >> 2;
    const int tig = lane & 3;
    const int c = i >> 2;
    const int rem = i & 3;
    const int h = rem >> 1;
    const int cs = rem & 1;
    *row = warp_in_wg * 16 + group + 8 * h;
    *col = c * 8 + tig * 2 + cs;
  }
};

struct WgmmaM64N256Frag {
  float r[128];

  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int i = 0; i < 128; ++i) r[i] = 0.f;
  }

  __device__ __forceinline__ void mma(uint64_t desc_a, uint64_t desc_b,
                                      uint32_t scale_d) {
    wgmma_m64n256k16_bf16(r, desc_a, desc_b, scale_d);
  }

  __device__ __forceinline__ void get_mn(int i, int warp_in_wg, int lane,
                                         int* row, int* col) const {
    const int group = lane >> 2;
    const int tig = lane & 3;
    const int c = i >> 2;
    const int rem = i & 3;
    const int h = rem >> 1;
    const int cs = rem & 1;
    *row = warp_in_wg * 16 + group + 8 * h;
    *col = c * 8 + tig * 2 + cs;
  }
};

__device__ __forceinline__ void wgmma_frag_mn(int i, int warp_in_wg, int lane,
                                               int* row, int* col) {
  const int group = lane >> 2;
  const int tig = lane & 3;
  const int c = i >> 2;
  const int rem = i & 3;
  const int h = rem >> 1;
  const int cs = rem & 1;
  *row = warp_in_wg * 16 + group + 8 * h;
  *col = c * 8 + tig * 2 + cs;
}

__device__ __forceinline__ void wgmma_frag_m128n(int i, int warp_in_wg,
                                                  int lane, int* row, int* col) {
  const int group = lane >> 2;
  const int tig = lane & 3;
  const int c = i >> 2;
  const int rem = i & 3;
  const int h = rem >> 1;
  const int cs = rem & 1;
  *row = warp_in_wg * 16 + group + 8 * h;
  *col = c * 8 + tig * 2 + cs;
}

}  // namespace hopper
}  // namespace velox
