static int ref_normalize_int8(int8_t* input, int8_t* output, int8_t* scale, ref_normalize_param* param)
{
    int batch_num = param->input_n;
    int in_h = param->input_h;
    int in_w = param->input_w;
    int in_c = param->input_c;
    int in_offset = 0;
    int out_offset = 0;

    int out_size = batch_num * in_h * in_w * in_c;
    float* buff = ( float* )malloc(sizeof(float) * in_h * in_w);
    float* out_f32 = ( float* )malloc(sizeof(float) * out_size);
    int8_t* in_buf = input;
    float* out_f32_tmp = out_f32;
    for(int n = 0; n < batch_num; ++n)
    {
        in_buf = input + n * in_h * in_w * in_c;
        out_f32_tmp = out_f32 + n * in_h * in_w * in_c;
        memset(buff, 0, sizeof(float) * in_h * in_w);
        for(int h = 0; h < in_h; ++h)
        {
            for(int w = 0; w < in_w; ++w)
            {
                int buff_idx = h * in_w + w;
                for(int c = 0; c < in_c; ++c)
                {
                    if(param->layout == 0)    // nchw
                    {
                        in_offset = c * in_h * in_w + h * in_w + w;
                    }
                    else    // nhwc
                    {
                        in_offset = h * in_w * in_c + w * in_c + c;
                    }
                    float data = ( float )param->in_scale * (in_buf[in_offset] - param->in_zero);
                    buff[buff_idx] += data * data;
                }
                buff[buff_idx] = 1.f / sqrt(buff[buff_idx]);
            }
        }
        for(int h = 0; h < in_h; ++h)
        {
            for(int w = 0; w < in_w; ++w)
            {
                int buff_idx = h * in_w + w;
                for(int c = 0; c < in_c; ++c)
                {
                    if(param->layout == 0)    // nchw
                    {
                        out_offset = c * in_h * in_w + h * in_w + w;
                        in_offset = out_offset;
                    }
                    else    // nhwc
                    {
                        out_offset = h * in_w * in_c + w * in_c + c;
                        in_offset = out_offset;
                    }
                    float data = buff[buff_idx];
                    float in_data = ( float )param->in_scale * (in_buf[in_offset] + param->in_zero);
                    float out_data = in_data * data;
                    if(scale)
                    {
                        out_data = out_data * param->scale_scale * (scale[c] + param->scale_zero);
                    }
                    out_f32_tmp[out_offset] = out_data;
                }
            }
        }
    }
    float output_max = 0.0f;
    for(int i = 0; i < out_size; i++)
    {
        if(output_max < fabs(out_f32[i]))
            output_max = fabs(out_f32[i]);
    }
    param->out_scale = output_max / 127;
    param->out_zero = 0;
    for(int i = 0; i < out_size; i++)
    {
        int s32_out = round(out_f32[i] * 127 / output_max);
        if(s32_out > 127)
            s32_out = 127;
        if(s32_out < -127)
            s32_out = -127;
        output[i] = s32_out;
    }
    free(buff);
    free(out_f32);
    out_f32 = NULL;
    buff = NULL;
    return 0;
}
