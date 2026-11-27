uint tgid = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint lane = tid % 32;
uint warp = tid / 32;
uint row = tgid * WARPS + warp;
uint slot = tgid * WARPS + warp;

if (row >= N) {
    if (lane == 0) { partial_key[slot] = 0u; partial_idx[slot] = 0; }
    return;
}

float acc = 0.0f;
for (uint k = lane; k < K; k += 32) {
    acc += (float)x[k] * (float)w[row * K + k];
}
acc = simd_sum(acc);

if (lane == 0) {
    uint f = as_type<uint>(acc);
    uint key = (f & 0x80000000u) ? ~f : (f | 0x80000000u);
    if (!(bool)allowed[row]) key = 0u;
    partial_key[slot] = key;
    partial_idx[slot] = (int)row;
}
