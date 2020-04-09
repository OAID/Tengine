static int ref_zeros_like_fp32(float* input, int* output, zeros_like_param* param)
{
    int elements_num = 1;
    for(int i = 0; i < param->dim_size; i++){
        elements_num *= param->in_dim[i];
    }

    for(int j = 0; j < elements_num; j++){
        *output++ = 0;
    }

    return 0;
}