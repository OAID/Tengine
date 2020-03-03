#include<stdio.h>
int ref_bias_fp32(float* in_data, float* out_data, float* bias, int size, int channels, float scale, int zero_point)
{
    for(int c = 0; c < channels; c++){
        float* out_ptr = out_data + c*size;
        float* in_ptr = in_data + c*size;
        for(int i = 0; i < size; i++)
        {
          out_ptr[i] = in_ptr[i] + bias[c];  
        }
    }
    return 0;
}