uint tid = thread_position_in_threadgroup.x;
uint nthreads = threads_per_threadgroup.x;
uint best_key = 0u;
int best_idx = 0;
for (uint i = tid; i < P; i += nthreads) {
    uint k = partial_key[i];
    int j = partial_idx[i];
    if (k > best_key || (k == best_key && j < best_idx)) {
        best_key = k;
        best_idx = j;
    }
}
threadgroup uint s_key[1024];
threadgroup int s_idx[1024];
s_key[tid] = best_key;
s_idx[tid] = best_idx;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    uint bk = 0u;
    int bi = 0;
    for (uint i = 0; i < nthreads; i++) {
        uint k = s_key[i];
        int j = s_idx[i];
        if (k > bk || (k == bk && j < bi)) { bk = k; bi = j; }
    }
    out[0] = bi;
}
