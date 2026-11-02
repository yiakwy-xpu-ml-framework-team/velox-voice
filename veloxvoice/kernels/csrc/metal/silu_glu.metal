uint i = thread_position_in_grid.x;
uint c0 = (i * VEC) % C;
uint n = (i * VEC) / C;
if (n >= N) return;
#pragma unroll
for (uint j = 0; j < VEC; j++) {
    uint c = c0 + j;
    if (c >= C) break;
    float a = (float)x[n * 2 * C + c];
    float b = (float)x[n * 2 * C + C + c];
    out[n * C + c] = (T)(a / (1.0f + metal::exp(-b)));
}
