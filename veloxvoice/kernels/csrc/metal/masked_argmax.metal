uint tid = thread_position_in_threadgroup.x;
uint nthreads = threads_per_threadgroup.x;
float best = -INFINITY;
int best_i = 0;
for (uint i = tid; i < V; i += nthreads) {
    if (!(bool)allowed[i]) continue;
    float v = (float)logits[i];
    if (v > best || (v == best && (int)i < best_i)) {
        best = v;
        best_i = (int)i;
    }
}
threadgroup float s_val[1024];
threadgroup int s_idx[1024];
s_val[tid] = best;
s_idx[tid] = best_i;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    float bv = -INFINITY;
    int bi = 0;
    for (uint i = 0; i < nthreads; i++) {
        float v = s_val[i];
        int j = s_idx[i];
        if (v > bv || (v == bv && j < bi)) { bv = v; bi = j; }
    }
    out[0] = bi;
}
