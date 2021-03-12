
__kernel
void concat2(__global float* y,
                  __global const float* x0,
                  __global const float* x1,
                  const int N,
                  const int axis_size,
                  const int pre_size,
                  const int post_size)
{
    int idx = get_global_id(0);

    int col = idx / axis_size;
    int row = idx % axis_size;

    if (idx < N)
    {
      if (row < pre_size)
      {
         y[idx] = x0[col * pre_size + row];
      }
      else
      {
         y[idx] = x1[col * post_size + row - pre_size];
      }
    }
}
