#include <stdio.h>
int ref_threshold_fp32(float* out_data, float* in_data, float threshold, int size, float scale, int zero_point)
{
    for(int i = 0; i < size; i++)
    {
        out_data[i] = in_data[i] > threshold ? 1.f : 0.f;
    }

    return 0;
}