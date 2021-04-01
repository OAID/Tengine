#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel
void mat_mul(__global const float* x,
             __global const float* y,
             __global float* out,
             const int M, const int N, const int K) {
 // const int col = get_global_id(0); // [0, M) columns of out == m
 // const int row = get_global_id(1); // [0, N) rows of out == n

  const int index = get_global_id(0);

  int col = index / N;
  int row = index % N;

  if ((col >= M) || (row >= N)) {
    return;
  }

  float x0, y0, out0 = 0;

  for (int p = 0; p < K; ++p) {
    x0 = *(x + row * K + p);
    y0 = *(y + col * K + p);
    out0 += x0 * y0;
  }

  out[index] = out0;
}