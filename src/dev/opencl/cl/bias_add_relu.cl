
__kernel
void bias_add_relu(__global float* y,
                   __global const float* x,
                   const int elem_num_perimg, const int elem_perchannel, const int N)
{
    const int idx = get_global_id(0);
    if (idx < N)
    {
        y[idx] += x[idx % elem_num_perimg / elem_perchannel];
        y[idx] = y[idx] > 0 ? y[idx] : 0;
    }
}
