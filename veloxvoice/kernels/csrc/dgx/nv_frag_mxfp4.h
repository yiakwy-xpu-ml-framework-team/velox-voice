/* VeloxVoice — DGX-Spark (sm_121a) warp-level nvfp4/mxfp4 fragments (Apache-2.0).
 *
 * Lineage: flash-float-jit-kernels fragment/nv_frag_gemm_scaled_impl.h (Hopper
 * symmetric fragment semantics) + secYOUre/nvfp4bench gemm_warp_mxf4.cu for the
 * m16n8k64 mxf4nvf4.block_scale row.col register mapping:
 *   g=lane>>2, t=lane&3 ; A row=2g+((p>>3)&1), k=16t+8*((p>>3)>>1)+(p&7)
 *   B col=g, k=16t+p ; D m=2g+(dreg>>1), n=2t+(dreg&1)
 */
#pragma once

#include <cstdint>

namespace xpu {

// one m16n8k64 packed mxf4nvf4 MMA: A = 4x uint32 packed e2m1 (16 codes), B = 2x uint32
// packed e2m1 (16 codes), D = 4x float. Scales: ue8m0 2X unit (legacy 1.0 default).
__device__ inline void mxfp4_mma(float acc[4], const uint32_t a[4], const uint32_t b[2],
                                 uint32_t scale_a = 0x00007F7Fu,
                                 uint32_t scale_b = 0x00007F7Fu) {
  float c0 = acc[0], c1 = acc[1], c2 = acc[2], c3 = acc[3];
  const uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col."
      "f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
      "{%14},{%15,%16},{%17},{%18,%19};\n"
      : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(scale_a), "h"(z), "h"(z), "r"(scale_b), "h"(z), "h"(z));
}

}  // namespace xpu
