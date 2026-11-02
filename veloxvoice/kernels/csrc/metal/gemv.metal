uint tid = thread_position_in_threadgroup.x;
uint tgid = threadgroup_position_in_grid.x;
uint lane = tid % 32;
uint sg = tid / 32;
uint row = tgid * ROWS_PER_TG + sg;
if (row >= N) return;

float acc = 0.0f;
for (uint k = lane; k < K; k += 32) {
    acc += (float)x[k] * (float)w[row * K + k];
}
acc = simd_sum(acc);
if (lane == 0) out[row] = (T)acc;
