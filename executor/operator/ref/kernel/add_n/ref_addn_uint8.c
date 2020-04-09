static int ref_addn_uint8(uint8_t** input, uint8_t* output, const ref_addn_param* param)
{
    int input_size = param->input_size;
    int in_num = param->in_num;
    float* out_f32 = ( float* )malloc(input_size);
    memset(out_f32, 0, input_size);
    for(int i = 0; i < in_num; ++i)
    {
        uint8_t* input_data = input[i];
        float input_scale = param->in_scale[i];
        int zero_point = param->in_zero[i];
        for(int j = 0; j < input_size; ++j)
        {
            out_f32[j] += (input_data[j] * input_scale + zero_point);
        }
    }
    for(int j = 0; j < input_size; ++j)
    {
        int s32_out = round(out_f32[j] / param->out_scale) + param->out_zero;
        if(s32_out > 255)
            s32_out = 255;
        if(s32_out < 0)
            s32_out = 0;
        output[j] = s32_out;
    }
    free(out_f32);
    out_f32 = NULL;

    return 0;
}
