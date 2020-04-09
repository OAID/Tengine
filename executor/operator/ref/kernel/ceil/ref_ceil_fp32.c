#include <math.h>
static int ref_ceil_fp32(float* input, int* output, ceil_param* param)
{
    int elements_num = 1;
    for(int i = 0; i < param->dim_size; i++){
        elements_num *= param->in_dim[i];
    }

    for(int j = 0; j < elements_num; j++){
        *output++ = ceil((*input++));
    }

    return 0;
}