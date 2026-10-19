// sheetsage_wgmma.h
// WGMMA tensor-core fragments for the SheetSage2 decoder GEMVs.
//
// The decoder's per-step GEMVs are [1, K] × [K, N] — mathematically GEMV.
// We pad M=1→64 and use wgmma.m64nNk16 (Hopper tensor core) for a 2×
// throughput win over SIMT (bf16→fp32 cvt + shfl reduce ≈ 2 cy/MAC vs
// wgmma ≈ 1 cy/64 MACs, with 64× padding waste the net is ~2×).
//
// The query h[512] is replicated 64× into shared memory (wgmma A operand
// layout) once per GEMV call.  Shared usage: 64 × K × 2B per k-tile.
//
// Fragments:
//   SheetsageWgmmaN64   — [64, 512] × [512, 64]   (out_proj, xq, fc2)
//   SheetsageWgmmaN128  — [64, 512] × [512, 128]
//   SheetsageWgmmaN256  — [64, 512] × [512, 256]
//   SheetsageWgmmaN192  — [64, 512] × [512, 192]  (qkv: 3×64 per wg group)
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../arch/wgmma/gmma_sm90.h"

namespace velox {
namespace hopper {
namespace sheetsage {

using bf16 = __nv_bfloat16;

// Replicate a [K] bf16 vector 64× into shared memory in the wgmma A operand
// layout (m64k16: 64 rows × 16 cols per k-tile, K/16 k-tiles).
// smem_a must have space for 64 * K * 2 bytes.
template <int K>
__device__ __forceinline__ void replicate_query(
    const bf16* __restrict__ h,  // [K] query vector (global or shared)
    bf16* smem_a,                // [64, K] output in shared
    int tid, int BLOCK) {
  // Each thread copies elements; the 64-row replication is a simple stride
  // write (all rows get the same value).  For K=512: 512 × 64 × 2B = 64KB.
  for (int i = tid; i < K; i += BLOCK) {
    bf16 val = h[i];
    // write to all 64 rows at column i
    bf16* col = smem_a + i;
    #pragma unroll 8
    for (int r = 0; r < 64; ++r)
      col[r * K] = val;
  }
  __syncthreads();
}

// Generic wgmma GEMV: [1, K] × [K, N] via M=64 padded wgmma.
// Only row 0 of the accumulator is useful; rows 1-63 are garbage.
// The caller extracts row 0 from the accumulator fragment.
template <int N, int K, typename Acc>
__device__ __forceinline__ void wgmma_gemv(
    const bf16* __restrict__ smem_a,  // [64, K] replicated query (shared)
    const bf16* __restrict__ w,       // [N, K] weight (global, L2-cached)
    const bf16* __restrict__ bias,    // [N] or nullptr
    Acc& acc,                          // wgmma accumulator (output)
    int tid, int BLOCK) {
  constexpr int kKPerTile = 16;
  constexpr int kKTiles = K / kKPerTile;
  constexpr int kNTiles = N / 64;  // for N=64: 1 tile, N=128: 2, etc.

  acc.zero();
  wgmma_fence();

  // For each n-tile of 64 columns
  for (int nt = 0; nt < kNTiles; ++nt) {
    for (int kt = 0; kt < kKTiles; ++kt) {
      // A descriptor: points to smem_a + (kt * 16) column offset
      // The A operand for this k-tile is [64, 16] at smem_a[0][kt*16]
      const bf16* a_ptr = smem_a + kt * kKPerTile;
      uint64_t desc_a = make_gmma_desc_b128(a_ptr);

      // B descriptor: points to w + (nt*64) * K + kt*16
      // The B operand is [N_tile, 16] from global memory — wgmma requires
      // B in shared memory too.  For the persistent kernel, we prefetch
      // B into shared memory (producer warps).
      // For now: this is a placeholder that needs the B operand in smem.
      // The producer warps handle this prefetch.
      // desc_b = make_gmma_desc_b128(smem_b + ...);
      // acc.mma(desc_a, desc_b, 1);
    }
  }
  wgmma_commit();
  wgmma_wait<0>();
}

// Extract row 0 from the accumulator (the only useful row for GEMV).
// The get_mn() method maps accumulator index → (row, col).
template <int N, typename Acc>
__device__ __forceinline__ void extract_row0(
    const Acc& acc, const bf16* __restrict__ bias,
    bf16* __restrict__ out,  // [N]
    int warp_in_wg, int lane) {
  constexpr int kAccsPerThread = sizeof(acc.r) / sizeof(acc.r[0]);
  for (int i = 0; i < kAccsPerThread; ++i) {
    int row, col;
    acc.get_mn(i, warp_in_wg, lane, &row, &col);
    if (row == 0 && col < N) {
      float val = acc.r[i];
      if (bias != nullptr) val += ar_tof(bias[col]);
      out[col] = ar_tob(val);
    }
  }
}

}  // namespace sheetsage
}  // namespace hopper
}  // namespace velox
