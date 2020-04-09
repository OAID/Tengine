#include <stdio.h>

int ref_hardsigmoid_fp32(float* in_data, float* out_data, int size, int alpha, int beta,float scale, int zero_point)
{
    float lower = -beta / alpha;
    float upper = (1.f/alpha) + lower;
    for(int i = 0; i < size; i++)
    {
        if(in_data[i] < lower)
            out_data[i] = 0.f;
        else if(out_data[i] > upper)
            out_data[i] = 1.f;
        else
            out_data[i] = in_data[i] * alpha + beta;
    }

    return 0;
}