static int ref_addn_int8(int8_t** input, int8_t* output, ref_addn_param* param)
{
    int input_size = param->input_size;
    int in_num = param->in_num;

    float* out_f32 = ( float* )malloc(input_size * sizeof(float));
    memset(out_f32, 0x0, input_size);
    for(int i = 0; i < in_num; ++i)
    {
        int8_t* input_data = input[i];
        float input_scale = param->in_scale[i];
        for(int j = 0; j < input_size; ++j)
        {
            out_f32[j] += input_data[j] * input_scale;
        }
    }
    float output_max = 0.0f;
    for(int i = 0; i < input_size; i++)
    {
        if(output_max < fabs(out_f32[i]))
            output_max = fabs(out_f32[i]);
    }
    param->out_scale = output_max / 127;
    param->out_zero = 0;
    for(int i = 0; i < input_size; i++)
    {
        int s32_out = round(out_f32[i] * 127 / output_max);
        if(s32_out > 127)
            s32_out = 127;
        if(s32_out < -127)
            s32_out = -127;
        output[i] = s32_out;
    }
    free(out_f32);
    out_f32 = NULL;
    return 0;
}
