
__kernel
void reshape(__global float* y,
             __global const float* x,
                const int N)
{
    const int idx = get_global_id(0);
    if (idx < N)
    {
        y[idx] = x[idx];
    }
}