
__kernel
void eltwise_add(__global float* y,
                  __global const float* x0,
                  __global const float* x1,
                  const int N)
{
    int idx = get_global_id(0);

    if (idx < N)
    {
      y[idx] = x0[idx] + x1[idx];
    }
}






