
__kernel
void slice(__global float* y,
             __global const float* x,
                const int N,
                int res)
{
    const int idx = get_global_id(0);

    if (idx < N)
    {
        const int idx_new = idx + res;
        y[idx] = x[idx_new];
    }
}