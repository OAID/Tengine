static int ref_addn_fp16(__fp16** input, __fp16* output, const ref_addn_param* param)
{
    int input_size = (param->input_size / sizeof(__fp16));
    int in_num = param->in_num;

#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
    float* buff = ( float* )malloc(input_size);
    for(int i = 0; i < in_num; ++i)
    {
        __fp16* input_data = input[i];
        for(int j = 0; j < input_size; ++j)
        {
            float data = fp16_to_fp32(input_data[j]);
            buff[j] += data;
        }
    }
    for(int j = 0; j < input_size; ++j)
    {
        output[j] = fp32_to_fp16(buff[j]);
    }

    free(buff);
#else
    for(int i = 0; i < in_num; ++i)
    {
        __fp16* input_data = input[i];
        for(int j = 0; j < input_size; ++j)
        {
            output[j] += input_data[j];
        }
    }
#endif
    return 0;
}
