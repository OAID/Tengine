#include <math.h>
#include <iostream>
static int ref_squared_difference_fp32(float* input1, float* input2, float* output, squared_difference_param* param)
{
    int elements_num = 1;
    for(int i = 0; i < param->in_dim_size; i++){
        elements_num *= param->in_dim[i];
    }

    for(int j = 0; j < elements_num; j++){
        *output++ = pow((*input1++ - *input2++), 2);
    }

    return 0;
}