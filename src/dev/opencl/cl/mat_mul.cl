#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel
void mat_mul(__global const float* x,
             __global const float* y,
             __global float* out,
             const int M, const int N, const int K) {
 // const int col = get_global_id(0); // [0, M) columns of out == m
 // const int row = get_global_id(1); // [0, N) rows of out == n

  const int index = get_global_id(0);

//  int bM = ((M+3)/4*4-4);
//  int bN = ((N+3)/4*4-4);

  const int bM = M/4*4;
  const int bN = N/4*4;

  int i = (index / N);
  int j = (index % N);

  int col = (index / N) << 2;
  int row = (index % N) << 2;
  if (index < bM*bN)
  {
    if (col+3 < M && row+3 < N)
    {
        float c00 = 0, c01 = 0, c02 = 0, c03 = 0,
                      c10 = 0, c11 = 0, c12 = 0, c13 = 0,
                      c20 = 0, c21 = 0, c22 = 0, c23 = 0,
                      c30 = 0, c31 = 0, c32 = 0, c33 = 0;

        for (int p = 0; p < K; p++)
        {
            float a00 = *(x + p * N + row),
                  a10 = *(x + p * N + (row+1) ),
                  a20 = *(x + p * N + (row+2) ),
                  a30 = *(x + p * N + (row+3) ),

                  b00 = *(y + col * K + p),
                  b01 = *(y + (col+1) * K + p),
                  b02 = *(y + (col+2) * K + p),
                  b03 = *(y + (col+3) * K + p);

            c00 += a00 * b00; c01 += a00 * b01; c02 += a00 * b02; c03 += a00 * b03;
            c10 += a10 * b00; c11 += a10 * b01; c12 += a10 * b02; c13 += a10 * b03;
            c20 += a20 * b00; c21 += a20 * b01; c22 += a20 * b02; c23 += a20 * b03;
            c30 += a30 * b00; c31 += a30 * b01; c32 += a30 * b02; c33 += a30 * b03;
        }
        out[col*N+row] = c00;     out[(col+1)*N+row] = c01;     out[(col+2)*N+row] = c02;     out[(col+3)*N+row] = c03;
        out[col*N+(row+1)] = c10;     out[(col+1)*N+(row+1)] = c11;     out[(col+2)*N+(row+1)] = c12;     out[(col+3)*N+(row+1)] = c13;
        out[col*N+(row+2)] = c20;     out[(col+1)*N+(row+2)] = c21;     out[(col+2)*N+(row+2)] = c22;     out[(col+3)*N+(row+2)] = c23;
        out[col*N+(row+3)] = c30;     out[(col+1)*N+(row+3)] = c31;     out[(col+2)*N+(row+3)] = c32;     out[(col+3)*N+(row+3)] = c33;
    }
  }
  else if ( index < M*N )
  {
      float a0, b0, c0 = 0;
      for (int p = 0; p < K; ++p)
      {
          a0 = *(x + p * N + j);
          b0 = *(y + i * K + p);
          c0 += a0 * b0;
      }
      out[i * N + j] = c0;
  }

//        float a0, b0, c0 = 0;
//        for(int i = col; i < M; ++i)
//        {
//            for (int j = row; j < N; ++j)
//            {
//                c0 = 0;
//                for (int p = 0; p < K; ++p)
//                {
//                    a0 = *(x + p * N + j);
//                   b0 = *(y + i * K + p),
//                    c0 += a0 * b0;
//                }
//                out[i * N + j] = c0;
//            }
//        }



//  float x0, y0, out0 = 0;

//  for (int p = 0; p < K; ++p) {
//    x0 = *(x + p * N + row);
//    y0 = *(y + col * K + p);
//    out0 += x0 * y0;
//  }

//  out[index] = out0;

}