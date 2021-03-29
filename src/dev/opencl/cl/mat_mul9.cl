#pragma OPENCL EXTENSION cl_amd_printf : enable

#define TS 16

__kernel
void mat_mul(__global const float* x,
             __global const float* y,
             __global float* out,
             const int M, const int N, const int K) {

    int local_idx = get_local_id(0);
    int local_group = get_group_id(0);

//    int row_local = get_local_id(0); // Local row ID (max: TS)
//    int col_local = get_local_id(1); // Local col ID (max: TS)

    int row_local = local_idx / TS; // Local row ID (max: TS)
    int col_local = local_idx % TS; // Local col ID (max: TS)

//    int i = TS*get_group_id(0) + row_local; // Row ID of C (0..M)
//    int j = TS*get_group_id(1) + col_local; // Col ID of C (0..N)

    int NCol = (N + (TS-1))/TS;
    int Mgroup = local_group / NCol;
    int Ngroup = local_group % NCol;
    int i = TS * Mgroup + row_local; // Row ID of C (0..M)
    int j = TS * Ngroup + col_local; // Col ID of C (0..N)

    if ((i >= M) || (j >= N)) {
        return;
    }

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    const int bM = M/TS*TS;
    const int bN = N/TS*TS;

//    int col = i << 2;
//    int row = j << 2;

    float x0, y0, out0 = 0;
    float a0, b0, c0 = 0;

    const int numTiles = K/TS;
    const int TilesBreak = K/TS * TS;

    if (i < bM && j < bN)
    {
        for (int t=0; t<numTiles; t++)
        {
            const int tiledRow = TS*t + row_local;
            const int tiledCol = TS*t + col_local;

            Asub[row_local][col_local] = x[tiledRow*N + j];
            Bsub[row_local][col_local] = y[i*K + tiledCol];

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int k=0; k<TS; k++)
            {
                //x0 = *(x + tiledRow*N + j);
                //y0 = *(y + i*K + tiledCol);
                //out0 += x0 * y0;
                out0 += Asub[k][col_local] * Bsub[row_local][k];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        for (int k=TilesBreak; k<K; k++)
        {
            x0 = *(x + k * N + j);
            y0 = *(y + i * K + k);
            out0 += x0 * y0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        out[i * N + j] = out0;
    }
    else if (i < M && j < N)
    {
      for (int p = 0; p < K; ++p)
      {
          a0 = *(x + p * N + j);
          b0 = *(y + i * K + p);
          c0 += a0 * b0;
      }
      out[i * N + j] = c0;
    }


}