
__kernel
void relu(__global float* y,
          __global const float* x,
             const int N)
{
    const int idx = get_global_id(0);
    if (idx < N)
    {
        y[idx] = x[idx] > 0 ? x[idx] : 0;
    }
}

__kernel
void relu6(__global float* y,
          __global const float* x,
             const int N)
{
    const int idx = get_global_id(0);
    if (idx < N)
    {
        y[idx] = x[idx] > 0 ? x[idx] : 0;
        y[idx] = y[idx] > 6 ? 6 : y[idx];
    }
}