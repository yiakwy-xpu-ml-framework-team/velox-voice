// velox_chunk_rel_pos_attn.cu — bounded-cache chunked rel-pos attention.
// Two kernels, matching models/wenet/torch_conformer.py math exactly:
//
// A) scores: per (h, i, j): m_ac[h,i,j] = dot(q_u[h,i,:], k[h,j,:]);
//    m_bd[h,i,j] = dot(q_v[h,i,:], P[idx[h,i,j], :]);  scores = (m_ac + m_bd)/sqrt(DK)
//    then masked softmax over L (per h, i) -> probs [H, Tq, L]
// B) ctx:    out[h,i,:] = sum_j probs[h,i,j] * v[h,j,:]
//
// q_u/q_v: [H, Tq, DK] precomputed (linear_q(x) + pos_bias_{u,v});
// k/v: [H, L, DK] post-roll; pe: [SPAN, H*DK]; idx/h2i tables; L <= 1024, DK <= 128.

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int BLOCK_X = 256;
constexpr int MAX_L = 1024;  // bounded caches; launcher asserts L <= MAX_L

__global__ void RelPosScoresKernel(const float* __restrict__ q_u,
                                   const float* __restrict__ q_v,
                                   const float* __restrict__ k,
                                   const float* __restrict__ pe,
                                   const int* __restrict__ idx,
                                   const int* __restrict__ valid,
                                   float* __restrict__ probs,
                                   int H, int Tq, int L, int DK, int SPAN,
                                   float inv_sqrt_dk) {
  // one block per (h, i); one WARP per j (lanes walk d coalesced, shfl reduce)
  int h = blockIdx.y;
  int i = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int nwarp = blockDim.x >> 5;  // BLOCK_X / 32
  const float* q_u_i = q_u + ((long)(h * Tq + i)) * DK;
  const float* q_v_i = q_v + ((long)(h * Tq + i)) * DK;
  int vcnt = valid[h * Tq + i];       // number of usable k slots for this (h, i)
  __shared__ float sh_scores[MAX_L];  // L <= MAX_L asserted by launcher
  __shared__ float sh_red[BLOCK_X / 32];

  for (int j = warp; j < L; j += nwarp) {
    float s = -1e30f;
    if (j >= L - vcnt) {  // usable = the LAST vcnt slots of the L-window
      const float* k_j = k + ((long)(h * L + j)) * DK;
      const float* p_ij = pe + ((long)idx[i * L + j] * (H * DK)) + (long)h * DK;
      float ac = 0.f, bd = 0.f;
      for (int d = lane; d < DK; d += 32) {  // lanes contiguous -> coalesced
        ac += q_u_i[d] * k_j[d];
        bd += q_v_i[d] * p_ij[d];
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        ac += __shfl_down_sync(0xffffffffu, ac, off);
        bd += __shfl_down_sync(0xffffffffu, bd, off);
      }
      s = (ac + bd) * inv_sqrt_dk;
    }
    if (lane == 0) sh_scores[j] = s;
  }
  __syncthreads();
  float best = -1e30f;
  for (int j = threadIdx.x; j < L; j += BLOCK_X) best = fmaxf(best, sh_scores[j]);
  // block reduce max
  for (int off = 16; off > 0; off >>= 1) best = fmaxf(best, __shfl_down_sync(0xffffffffu, best, off));
  if (lane == 0) sh_red[warp] = best;
  __syncthreads();
  if (warp == 0) {
    int nw = BLOCK_X / 32;
    best = lane < nw ? sh_red[lane] : -1e30f;
    for (int off = 16; off > 0; off >>= 1) best = fmaxf(best, __shfl_down_sync(0xffffffffu, best, off));
    if (lane == 0) sh_red[0] = best;
  }
  __syncthreads();
  best = sh_red[0];
  // exp & sum
  float sum = 0.f;
  for (int j = threadIdx.x; j < L; j += BLOCK_X) {
    float e = expf(sh_scores[j] - best);
    sh_scores[j] = e;
    sum += e;
  }
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
  if (lane == 0) sh_red[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    int nw = BLOCK_X / 32;
    sum = lane < nw ? sh_red[lane] : 0.f;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) sh_red[0] = sum;
  }
  __syncthreads();
  float inv = 1.f / fmaxf(sh_red[0], 1e-30f);
  for (int j = threadIdx.x; j < L; j += BLOCK_X)
    probs[(long)(h * Tq + i) * L + j] = sh_scores[j] * inv;
}

__global__ void RelPosCtxKernel(const float* __restrict__ probs,
                                const float* __restrict__ v,
                                float* __restrict__ out,
                                int H, int Tq, int L, int DK) {
  // thread per (h, i, d): dot over L
  int idx_ = blockIdx.x * BLOCK_X + threadIdx.x;
  if (idx_ >= H * Tq * DK) return;
  int d = idx_ % DK;
  int i = (idx_ / DK) % Tq;
  int h = idx_ / (DK * Tq);
  const float* p_i = probs + ((long)(h * Tq + i) * L);
  float acc = 0.f;
  for (int j = 0; j < L; ++j) acc += p_i[j] * v[((h * L + j) * DK) + d];
  out[idx_] = acc;
}

}  // namespace

void velox_chunk_rel_pos_attn(tvm::ffi::TensorView q_u, tvm::ffi::TensorView q_v,
                              tvm::ffi::TensorView k, tvm::ffi::TensorView pe,
                              tvm::ffi::TensorView idx, tvm::ffi::TensorView valid,
                              tvm::ffi::TensorView probs, tvm::ffi::TensorView v,
                              tvm::ffi::TensorView out) {
  int64_t H = q_u.size(0), Tq = q_u.size(1), DK = q_u.size(2);
  int64_t L = k.size(1), SPAN = pe.size(0);
  TVM_FFI_ICHECK_LE(L, MAX_L);
  TVM_FFI_ICHECK_LE(DK, 128);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(q_u.device().device_type, q_u.device().device_id));
  dim3 scores_grid((unsigned)Tq, (unsigned)H, 1);
  RelPosScoresKernel<<<scores_grid, BLOCK_X, 0, stream>>>(
      static_cast<const float*>(q_u.data_ptr()), static_cast<const float*>(q_v.data_ptr()),
      static_cast<const float*>(k.data_ptr()), static_cast<const float*>(pe.data_ptr()),
      static_cast<const int*>(idx.data_ptr()), static_cast<const int*>(valid.data_ptr()),
      static_cast<float*>(probs.data_ptr()),
      (int)H, (int)Tq, (int)L, (int)DK, (int)SPAN, (float)(1.0 / sqrt((double)DK)));
  int64_t total = H * Tq * DK;
  RelPosCtxKernel<<<(unsigned)((total + BLOCK_X - 1) / BLOCK_X), BLOCK_X, 0, stream>>>(
      static_cast<const float*>(probs.data_ptr()), static_cast<const float*>(v.data_ptr()),
      static_cast<float*>(out.data_ptr()), (int)H, (int)Tq, (int)L, (int)DK);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(chunk_rel_pos_attn, velox_chunk_rel_pos_attn);
