
__kernel
void slice(__global float* y,
             __global const float* x,
                const int N,
                int res)
{
    const int idx = get_global_id(0);
    int idx_new = idx + res;

    if (idx < N)
    {
        y[idx] = x[idx_new];
    }
}