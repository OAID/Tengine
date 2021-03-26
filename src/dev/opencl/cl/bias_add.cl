
__kernel
void bias_add(__global float* y,
                   __global const float* x,
                   const int elem_num_perimg, const int elem_perchannel, const int N)
{
    const int idx = get_global_id(0);
    if (idx < N)
    {
        const int x_idx = idx % elem_num_perimg / elem_perchannel;
        y[idx] += x[x_idx];
    }
}