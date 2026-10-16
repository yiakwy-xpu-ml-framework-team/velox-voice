// velox_prefix_beam_search.cu
// Multi-block CLUSTER CTC prefix beam search (EXPERIMENT).
//
// This variant launches C blocks thread-block cluster :
//   1. the cross-block barrier (cluster.sync() — no global atomics), and
//   2. cross-block SHARING OF THE BEAM STATE (s, ns, ln, ...): block 0 reads
//      each peer's final beam state directly from the peer's dynamic shared
//      memory via cooperative_groups::cluster::map_shared_rank, and uses those
//      states to score the stitched N-best.
//
// Algorithm: 
//   - [0, T) splits into C chunks; each block runs the SAME per-frame
// beam update over its chunk; 
//   - after one cluster.sync(), block 0 stitches hypothesis 
//       j = concat(chunk_0 rank-j, ..., chunk_{C-1} rank-j)
//     scoring each chunk's rank-j candidate by the peer's logadd(s, ns) read over
//     DSMEM. 
// 
// C=1 degenerates to the exact baseline.

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math.h>

#define ENABLE_HOPPER 1

namespace cg = cooperative_groups;

namespace {

// prob 0
#define NEG (-1.0e30f)

// MAX beams
#define MAX_B 8

// MAX tokens per frame
#define MAX_K 8

// MAX candidate paths (ended with blank or non-blank)
#define MAX_C (2 * MAX_B * MAX_K)

#define PRIME 1000003ULL

// MAX frames per chunk
#define MAX_SMEM_CF 1024

#define THREADS_PER_BLOCK 128

struct __align__(16) SmemLayout {
  // beam state shared cross-blocks (read over DSMEM by block 0)
  float s[MAX_B];
  float ns[MAX_B];
  float lobsh[MAX_B];
  int ln[MAX_B];
  int lasttok[MAX_B];
  int ln_old[MAX_B];
  int lasttok_old[MAX_B];
  unsigned long long h[MAX_B];

  // per-frame candidate scratch
  float sa[MAX_C], na[MAX_C], csv[MAX_C], cnv[MAX_C], total[MAX_C];

  unsigned long long ch[MAX_C];
  int parent[MAX_C], app[MAX_C], rep[MAX_C];
  int selb[MAX_B];

  // merge scratch
  float m_sc[MAX_B];
  int m_len[MAX_B], m_rank[MAX_B];

  // per-chunk token frame tracking (for ownership-based exact merge)
  int tok_frm[MAX_B * MAX_SMEM_CF];
};

constexpr size_t kSmemSize = sizeof(SmemLayout);

__device__ __forceinline__ float logadd(float a, float b) {
  if (a < -1.0e29f) return b;
  if (b < -1.0e29f) return a;
  float m = fmaxf(a, b);
  return m + __logf(1.0f + __expf(-fabsf(a - b)));
}

__global__ void PrefixBeamChunksKernel(
    const long long* __restrict__ ti, const float* __restrict__ tv,
    int* __restrict__ hist_par, int* __restrict__ hist_app,
    long long* __restrict__ c_tok, int* __restrict__ c_tok_len, int* __restrict__ done,
    long long* __restrict__ tokens_out, int* __restrict__ lens_out,
    float* __restrict__ scores_out,
    const int T, const int K, const int B, const int Lmax, const int blank,
    const int C, const int cf, const int ov) {
  extern __shared__ char smem_raw[];
  SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_raw);
  cg::cluster_group cluster = cg::this_cluster();

  float* s = smem->s;
  float* ns = smem->ns;
  float* lobsh = smem->lobsh;
  float* sa = smem->sa;
  float* na = smem->na;
  float* csv = smem->csv;
  float* cnv = smem->cnv;
  float* total = smem->total;
  int* ln = smem->ln;
  int* lasttok = smem->lasttok;
  int* ln_old = smem->ln_old;
  int* lasttok_old = smem->lasttok_old;
  unsigned long long* h = smem->h;
  unsigned long long* ch = smem->ch;
  int* parent = smem->parent;
  int* app = smem->app;
  int* rep = smem->rep;
  int* selb = smem->selb;
  float* m_sc = smem->m_sc;
  int* m_len = smem->m_len;
  int* m_rank = smem->m_rank;

  const int tid = threadIdx.x;
  const int C2 = 2 * B * K;
  const int bid = blockIdx.x;

  const int t0 = bid * cf;
  const int t1 = min(t0 + cf, T);
  const int ps = max(0, t0 - ov);   // processing start
  const int pe = min(t1 + ov, T);   // processing end

  if (tid < B) {
    s[tid] = (tid == 0) ? 0.0f : NEG;
    ns[tid] = NEG;
    ln[tid] = 0;
    lasttok[tid] = -1;
    h[tid] = 0ULL;
  }
  __syncthreads();

  // Step 1 : per-chunk beam
  for (int t = ps; t < pe; ++t) {
    if (tid < B) {
      lobsh[tid] = logadd(s[tid], ns[tid]);
      lasttok_old[tid] = (ln[tid] > 0) ? lasttok[tid] : -1;
      ln_old[tid] = ln[tid];
      selb[tid] = -1;
    }
    __syncthreads();

    for (int c = tid; c < C2; c += blockDim.x) {
      int ext = (c >= B * K);
      int cc = ext ? (c - B * K) : c;
      int b = cc / K;
      int k = cc - b * K;
      int u = (int)ti[(long)t * K + k];
      float pu = tv[(long)t * K + k];
      int last = lasttok_old[b];
      bool is_blank = (u == blank);
      bool is_eq = (!is_blank) && (u == last);
      bool is_new = (!is_blank) && (!is_eq);
      parent[c] = b;
      app[c] = ext ? u : -1;
      ch[c] = ext ? (h[b] * PRIME + (unsigned long long)(unsigned)u) : h[b];
      if (ext) {
        csv[c] = NEG;
        cnv[c] = is_eq ? (s[b] + pu) : (is_new ? (lobsh[b] + pu) : NEG);
      } else {
        csv[c] = is_blank ? (lobsh[b] + pu) : NEG;
        cnv[c] = is_eq ? (ns[b] + pu) : NEG;
      }
    }
    __syncthreads();

    for (int c = tid; c < C2; c += blockDim.x) {
      unsigned long long hc = ch[c];
      float ms = NEG, accs = 0.f, mn = NEG, accn = 0.f;
      bool r = true;
      for (int d = 0; d < C2; ++d) {
        if (ch[d] == hc) {
          float v = csv[d];
          if (v > -1.0e29f) {
            if (v > ms) { accs *= __expf(ms - v); ms = v; }
            accs += __expf(v - ms);
          }
          float w = cnv[d];
          if (w > -1.0e29f) {
            if (w > mn) { accn *= __expf(mn - w); mn = w; }
            accn += __expf(w - mn);
          }
          if (d < c) r = false;
        }
      }
      sa[c] = (ms < -1.0e29f) ? NEG : (ms + __logf(accs));
      na[c] = (mn < -1.0e29f) ? NEG : (mn + __logf(accn));
      rep[c] = r ? 1 : 0;
      total[c] = r ? logadd(sa[c], na[c]) : NEG;
    }
    __syncthreads();

    if (tid < C2) {
      if (rep[tid] && total[tid] > -1.0e29f) {
        int rank = 0;
        for (int d = 0; d < C2; ++d) {
          if (rep[d] && total[d] > -1.0e29f) {
            if (total[d] > total[tid] ||
                (total[d] == total[tid] && ch[d] < ch[tid]))
              ++rank;
          }
        }
        if (rank < B) selb[rank] = tid;
      }
    }
    __syncthreads();

    if (tid < B) {
      int c = selb[tid];
      if (c < 0) {
        s[tid] = NEG; ns[tid] = NEG; ln[tid] = 0; lasttok[tid] = -1; h[tid] = 0ULL;
        hist_par[((long)bid * T + t) * B + tid] = 0;
        hist_app[((long)bid * T + t) * B + tid] = -1;
      } else {
        int b = parent[c];
        int a = app[c];
        hist_par[((long)bid * T + t) * B + tid] = b;
        hist_app[((long)bid * T + t) * B + tid] = a;
        s[tid] = sa[c];
        ns[tid] = na[c];
        h[tid] = ch[c];
        ln[tid] = (a >= 0) ? (ln_old[b] + 1) : ln_old[b];
        lasttok[tid] = (a >= 0) ? a : lasttok_old[b];
      }
    }
    __syncthreads();
  }

  // Step 2 : reconstruct owned tokens per chunk
  if (tid < B) {
    int cur = tid;

    int owned = 0;
    for (int t = t1 - 1; t >= t0; --t) {
      int a = hist_app[((long)bid * T + t) * B + cur];
      if (a >= 0) ++owned;
      cur = hist_par[((long)bid * T + t) * B + cur];
    }

    long long* dst = c_tok + ((long)bid * B + tid) * cf;
    int pos = owned;
    cur = tid;
    for (int t = t1 - 1; t >= t0 && pos > 0; --t) {
      int a = hist_app[((long)bid * T + t) * B + cur];
      if (a >= 0) dst[--pos] = (long long)a;
      cur = hist_par[((long)bid * T + t) * B + cur];
    }

    c_tok_len[bid * B + tid] = owned;
  }

  cluster.sync();

  // Step 3 : Cross-block Stitching to recover N-best 
  if (bid != 0) {
    // Non leading blocks wait for completion
    if (tid == 0) { while (atomicAdd(done, 0) == 0) {} }
    __syncthreads();
    return;
  }

  if (tid < B) {
    float sc = 0.f;
    int len = 0;

    for (int c = 0; c < C; ++c) {
      // the beam (tid) owned token count from differnt chunks
      int l_c = (int)c_tok_len[c * B + tid];

      const SmemLayout* peer =
          (c == bid) ? smem : cluster.map_shared_rank(smem, c);
      
      // accumulate scores for this beam
      float s_c = logadd(peer->s[tid], peer->ns[tid]);
      len += l_c;
      if (l_c > 0 && s_c > -1.0e29f) sc += s_c;
    }

    m_sc[tid] = sc;
    m_len[tid] = min(len, Lmax);
  }
  __syncthreads();

  // NOTE (yiakwy) : beams ranking, we tested a lot of times and find beams=5 is the best
  if (tid == 0) {
    for (int j = 0; j < B; ++j) {
      int r = 0;
      for (int d = 0; d < B; ++d) {
        if (m_sc[d] > m_sc[j] || (m_sc[d] == m_sc[j] && d < j)) ++r;
      }
      m_rank[j] = r;
    }
  }
  __syncthreads();

  if (tid < B) {
    int r = m_rank[tid];
    int off = 0;
    for (int c = 0; c < C; ++c) {
      int l_c = (int)c_tok_len[c * B + tid];
      const long long* src = c_tok + ((long)c * B + tid) * cf;
      for (int i = 0; i < l_c && off < Lmax; ++i)
        tokens_out[(long)r * Lmax + off++] = src[i];
    }
    for (int i = off; i < Lmax; ++i) tokens_out[(long)r * Lmax + i] = 0;
    lens_out[r] = off;
    scores_out[r] = m_sc[tid];
  }
  __syncthreads();
  if (tid == 0) { 
    __threadfence(); 
    atomicExch(done, 1); 
  }
}

}  // namespace

void velox_beam_mp_impl(tvm::ffi::TensorView ti /*topK token_ids*/, tvm::ffi::TensorView tv /*topK candidates log probs*/,
                        tvm::ffi::TensorView hist_par /*[C, T, B], parent per beam per frame*/, 
                        tvm::ffi::TensorView hist_app /*[C, T, B], appended tokens per beam per frame*/,
                        tvm::ffi::TensorView c_tok /*[C, B, cf], owned tokens per beam per frame*/, 
                        tvm::ffi::TensorView c_tok_len  /*[C, B], count of owend tokens per beam per frame*/,
                        tvm::ffi::TensorView done,
                        tvm::ffi::TensorView tokens_out /*[B, Lmax], N-best token sequences*/, 
                        tvm::ffi::TensorView lens_out   /*[B,], length per N-best token sequences*/,
                        tvm::ffi::TensorView scores_out /*[B,], scores per N-best token sequences*/,
                        int64_t nchunks, int64_t chunk_frames, int64_t blank,
                        int64_t ov) {
  const int T = ti.size(0);
  const int K = ti.size(1);

  // NOTE (yiakwy) : beam size
  const int B = tokens_out.size(0);

  const int Lmax = tokens_out.size(1);

  // 
  const int C = nchunks;

  // NOTE (yiakwy) : frames per chunk
  const int cf = chunk_frames;

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(ti.device().device_type, ti.device().device_id));

  auto topk_ids_ptr = static_cast<const long long*>(ti.data_ptr());
  auto topk_probs_ptr = static_cast<const float*>(tv.data_ptr());

  // NOTE (yiakwy) : backtrace table
  auto hist_par_ptr = static_cast<int*>(hist_par.data_ptr());
  auto hist_app_ptr = static_cast<int*>(hist_app.data_ptr());

  auto c_tok_ptr = static_cast<long long*>(c_tok.data_ptr());
  auto c_tok_len_ptr = static_cast<int*>(c_tok_len.data_ptr());

  auto done_ptr = static_cast<int*>(done.data_ptr());

  auto tokens_out_ptr = static_cast<long long*>(tokens_out.data_ptr());
  auto lens_out_ptr = static_cast<int*>(lens_out.data_ptr());
  auto scores_out_ptr = static_cast<float*>(scores_out.data_ptr());

  dim3 grid(C, 1, 1);
  dim3 block(THREADS_PER_BLOCK); 

#if ENABLE_HOPPER // Hopper cluster launch (host code; built for sm_90a)
  cudaLaunchConfig_t launch_config = {0};
  launch_config.gridDim = grid;
  launch_config.blockDim = block;
  launch_config.dynamicSmemBytes = kSmemSize;
  launch_config.stream = stream;

  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {(unsigned int)C, 1u, 1u};

  launch_config.attrs = &attr;
  launch_config.numAttrs = 1;

  cudaError_t state = cudaLaunchKernelEx(
      &launch_config, PrefixBeamChunksKernel,
      topk_ids_ptr, topk_probs_ptr, hist_par_ptr, hist_app_ptr, c_tok_ptr, c_tok_len_ptr, done_ptr, tokens_out_ptr, lens_out_ptr, scores_out_ptr, T, K, B, Lmax, blank, C, cf, ov);

  if (state != cudaSuccess) {
    (void)cudaGetLastError();
  }
#else
  cudaLaunchConfig_t cfg = {0};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = kSmemSize;
  cfg.stream = stream;
  cfg.attrs = nullptr;
  cfg.numAttrs = 0;

  cudaError_t state = cudaLaunchKernelEx(&cfg, PrefixBeamChunksKernel,
      topk_ids_ptr, topk_probs_ptr, hist_par_ptr, hist_app_ptr, c_tok_ptr, c_tok_len_ptr, done_ptr, tokens_out_ptr, lens_out_ptr, scores_out_ptr, T, K, B, Lmax, blank, C, cf, ov);
  if (state != cudaSuccess) {
    (void)cudaGetLastError();
  }
#endif // Hopper 900+ GPU with cluster/NoC support
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(beam_search, velox_beam_mp_impl);
