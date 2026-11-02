uint tid = thread_position_in_threadgroup.x;
uint tgid = threadgroup_position_in_grid.x;
uint lane = tid % 32;
uint warp = tid / 32;
uint row = tgid * WARPS + warp;
if (row >= N) return;

float acc[M];
#pragma unroll
for (uint m = 0; m < M; m++) acc[m] = 0.0f;

uint k4 = K / 4;
if (k4 == 0) return;
device const ushort4* wp =
    (device const ushort4*)(w + (size_t)row * K);
uint i = lane;
ushort4 wv = wp[i < k4 ? i : 0];
for (; i < k4; i += 32) {
    uint ni = i + 32;
    ushort4 nw = (ni < k4) ? wp[ni] : wv;   // double-buffer / prefetch
    float w0 = float(as_type<bfloat16_t>(wv.x));
    float w1 = float(as_type<bfloat16_t>(wv.y));
    float w2 = float(as_type<bfloat16_t>(wv.z));
    float w3 = float(as_type<bfloat16_t>(wv.w));
#pragma unroll
    for (uint m = 0; m < M; m++) {
        ushort4 xv = ((device const ushort4*)(x + (size_t)m * K))[i];
        acc[m] += w0 * float(as_type<bfloat16_t>(xv.x))
                + w1 * float(as_type<bfloat16_t>(xv.y))
                + w2 * float(as_type<bfloat16_t>(xv.z))
                + w3 * float(as_type<bfloat16_t>(xv.w));
    }
    wv = nw;
}
#pragma unroll
for (uint m = 0; m < M; m++) acc[m] = simd_sum(acc[m]);
if (lane == 0) {
#pragma unroll
    for (uint m = 0; m < M; m++) out[(size_t)m * N + row] = (T)acc[m];
}
