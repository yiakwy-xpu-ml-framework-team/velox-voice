uint tgid = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint lane = tid % 32;
uint warp = tid / 32;
uint row = tgid * WARPS + warp;
if (row >= N) return;

float sum = 0.0f, sumsq = 0.0f;
for (uint i = lane; i < D; i += 32) {
    float v = (float)x[row * D + i];
    sum += v;
    sumsq += v * v;
}
sum = simd_sum(sum);
sumsq = simd_sum(sumsq);

float mean = sum / (float)D;
float var = sumsq / (float)D - mean * mean;
float rstd = metal::rsqrt(var + eps[0]);
for (uint i = lane; i < D; i += 32) {
    out[row * D + i] = (T)(((float)x[row * D + i] - mean) * rstd *
                           (float)w[i] + (float)b[i]);
}
