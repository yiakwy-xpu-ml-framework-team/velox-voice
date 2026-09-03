/* VeloxVoice — mxfp4/e2m1 & ue8m0 codec helpers and params (Apache-2.0).
 *
 * Host/device reference decode tables for the native packed mxf4nvf4 path:
 * 2 FP4 codes per byte (lo nibble = even-k), unit scale e8m0 = 2^(int8).
 *
 * The mma.sync.aligned.kind::mxf4nvf4.block_scale path is sm_120+/121a hardware:
 * see CUDA PTX ISA 8.5 "Matrix multiply-accumulate instructions with block scale".
 */
#pragma once

#include <cstdint>

namespace velox_dgx {

struct Fp4 {
  static __host__ __device__ inline float decode(uint8_t c) {
    const float t[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return t[c & 7];
  }

  static __host__ __device__ inline uint8_t encode(float x) {
    float ax = fabsf(x);
    uint8_t c;
    if (ax <= .25f) c = 0;
    else if (ax <= .75f) c = 1;
    else if (ax <= 1.25f) c = 2;
    else if (ax <= 1.75f) c = 3;
    else if (ax <= 2.5f) c = 4;
    else if (ax <= 3.5f) c = 5;
    else if (ax <= 5.f) c = 6;
    else c = 7;
    return c | (x < 0 ? 8 : 0);
  }
};

// ue8m0 scale = exponent field of an fp32-like byte (unit scale 0x7F => 1.0)
constexpr uint8_t UE8M0_UNIT = 0x7F;
constexpr uint32_t SCALE_VEC_2X_UNIT = 0x00007F7F;

}  // namespace velox_dgx
