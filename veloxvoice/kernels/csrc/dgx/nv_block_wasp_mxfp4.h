/* VeloxVoice — DGX-Spark WASP-1p2c persistent split-K pipeline for mxfp4/nvfp4 (Apache-2.0).
 *
 * Structure lineage: flash-float-jit-kernels block/nv_block_1p2c_gemm_scaled_impl.h
 * (Hopper persistent producer/consumer split-K pipeline, mbarrier stage ring,
 * TMA cooperative tile loads, cluster sync). Our pull-request-delta for the
 * DGX-Spark sm_121a:
 *   * A/B tiles are PACKED nvfp4 uint8 (2 codes/byte) — fp8 unpacking dropped
 *   * warp m16n8k64 mxf4nvf4.block_scale.row.col per nv_frag_mxfp4.h
 *   * e8m0 scale tensors pluggable (v1: unit scale sf=0x7F7F registers)
 *   * cluster(1..2) sync per tile before the fp32 DSMEM split-K reduction
 *
 * Ring parity: producers skip waiting on the first STAGES slots; consumers toggle
 * parity per slot use; consumers arrive empty as one unit (warp leader).
 */
#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include "nv_frag_mxfp4.h"
#include "tma_lite.h"

namespace xpu {

template <int BM, int BN, int BK, int STAGES, int GROUP_SIZE_M, int CLUSTER_SIZE_M>
struct HopperPersistentMxfp4Pipeline {
  static constexpr int WARP = 32;
  static constexpr int TM = 2, TN = 8, WN = 2;
  static constexpr int KWP = BK / 8;
  static constexpr int NUM_WARPS = 9;        // 1 producer warp + 8 consumer warps

  struct SmemLayout {
    uint32_t As[STAGES][BM * KWP];
    uint32_t Bs[STAGES][BN * KWP];
    uint64_t full[STAGES];
    uint64_t empty[STAGES];
    float acc_smem[TM * TN * 4];
  };

  static __device__ inline void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
                 "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count) : "memory");
  }
  static __device__ inline void mbar_expect_tx(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::
                 "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx) : "memory");
  }
  static __device__ inline void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], 1;" ::
                 "r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
  }
  static __device__ inline void mbar_wait(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{.reg .pred p; WAIT: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra WAIT;}"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(parity) : "memory");
  }

  static __device__ inline void run_persistent(
      const CUtensorMap* tma_desc_A, const CUtensorMap* tma_desc_B,
      const uint8_t* scale_A, const uint8_t* scale_B,
      float* Out, int M, int N, int K,
      int num_blocks_m, int num_blocks_n, uint8_t* smem_buffer) {
    SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_buffer);
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP;
    const int lane_id = tid % WARP;
    const bool is_producer = (warp_id == 0);

    if (tid < STAGES) {
      mbar_init(&smem->full[tid], 1);              // 1 producer warp-leader arrives
      mbar_init(&smem->empty[tid], NUM_WARPS - 1); // 8 consumer warp-leaders arrive
    }
    if constexpr (CLUSTER_SIZE_M > 1) {
      cg::this_cluster().sync();
    } else {
      __syncthreads();
    }
    (void)scale_A; (void)scale_B;  // v1: unit scale registers per MMA operand

    for (int tile = blockIdx.x; tile < num_blocks_m * num_blocks_n; tile += gridDim.x * CLUSTER_SIZE_M) {
      const int bm = tile / num_blocks_n;
      const int bn = tile % num_blocks_n;
      const int baseM = bm * BM, baseN = bn * BN;
      const int ns = K / BK;

      if (is_producer) {
        if (lane_id == 0) {
          for (int s = 0; s < ns; ++s) {
            const int buf = s % STAGES;
            if (s >= STAGES) {
              mbar_wait(&smem->empty[buf], ((s / STAGES) & 1) ^ 1);
            }
            mbar_expect_tx(&smem->full[buf], BM * KWP * 4 + BN * KWP * 4);
            nvgpu::arch::tma_load_2d_bytes(tma_desc_A, smem->As[buf], s * (BK / 2), baseM,
                                           &smem->full[buf]);
            nvgpu::arch::tma_load_2d_bytes(tma_desc_B, smem->Bs[buf], s * (BK / 2), baseN,
                                           &smem->full[buf]);
          }
        }
        continue;
      }

      // -- consumers (warps 1..8) --
      const int wm = warp_id - 1;
      const int wm2 = wm / WN;
      const int wn = wm % WN;
      const int warpM = wm2 * (TM * 16);
      const int warpN = wn * (TN * 8);
      const int g = lane_id / 4, t = lane_id % 4;

      float acc[TM * TN][4];
#pragma unroll
      for (int i = 0; i < TM * TN; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

      for (int s = 0; s < ns; ++s) {
        const int buf = s % STAGES;
        mbar_wait(&smem->full[buf], (s / STAGES) & 1);

        uint32_t af[TM][4], bf[TN][2];
#pragma unroll
        for (int mi = 0; mi < TM; ++mi) {
          int rE = warpM + mi * 16 + 2 * g, rO = rE + 1;
          af[mi][0] = smem->As[buf][rE * KWP + t * 2];
          af[mi][1] = smem->As[buf][rO * KWP + t * 2];
          af[mi][2] = smem->As[buf][rE * KWP + t * 2 + 1];
          af[mi][3] = smem->As[buf][rO * KWP + t * 2 + 1];
        }
#pragma unroll
        for (int ni = 0; ni < TN; ++ni) {
          int col = warpN + ni * 8 + g;
          bf[ni][0] = smem->Bs[buf][col * KWP + t * 2];
          bf[ni][1] = smem->Bs[buf][col * KWP + t * 2 + 1];
        }
#pragma unroll
        for (int mi = 0; mi < TM; ++mi)
          for (int ni = 0; ni < TN; ++ni)
            mxfp4_mma(acc[mi * TN + ni], af[mi], bf[ni]);

        if (lane_id == 0) mbar_arrive(&smem->empty[buf]);
      }

      if constexpr (CLUSTER_SIZE_M > 1) {
        cg::this_cluster().sync();
      }
#pragma unroll
      for (int mi = 0; mi < TM; ++mi)
#pragma unroll
        for (int ni = 0; ni < TN; ++ni) {
          int row0 = baseM + warpM + mi * 16 + 2 * g;
          int col0 = baseN + warpN + ni * 8 + 2 * t;
          float* a = acc[mi * TN + ni];
          if (row0 + 1 < M && col0 + 1 < N) {
            Out[(row0 + 0) * N + col0 + 0] = a[0];
            Out[(row0 + 0) * N + col0 + 1] = a[1];
            Out[(row0 + 1) * N + col0 + 0] = a[2];
            Out[(row0 + 1) * N + col0 + 1] = a[3];
          }
        }
    }
  }
};

}  // namespace xpu
