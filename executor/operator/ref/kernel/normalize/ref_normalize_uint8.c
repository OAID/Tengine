
static int ref_normalize_uint8(uint8_t* input, uint8_t* output, uint8_t* scale, const ref_normalize_param* param)
{
    int batch_num = param->input_n;
    int in_h = param->input_h;
    int in_w = param->input_w;
    int in_c = param->input_c;
    int in_offset = 0;
    int out_offset = 0;

    float* buff = ( float* )malloc(sizeof(float) * in_h * in_w);
    uint8_t* in_buf = input;
    uint8_t* out_buf = output;

    for(int n = 0; n < batch_num; ++n)
    {
        in_buf = input + n * in_h * in_w * in_c;
        out_buf = output + n * in_h * in_w * in_c;
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
                    float data = ( float )param->in_scale * (in_buf[in_offset] + param->in_zero);
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
                    float in_data = ( float )param->in_scale * (in_buf[in_offset] - param->in_zero);
                    float out_data = in_data * data;
                    if(scale)
                    {
                        out_data = out_data * param->scale_scale * (scale[c] - param->scale_zero);
                    }
                    int s32_out = round(out_data / param->out_scale) + param->out_zero;
                    if(s32_out > 255)
                        s32_out = 255;
                    if(s32_out < 0)
                        s32_out = 0;
                    out_buf[out_offset] = s32_out;
                }
            }
        }
    }

    free(buff);
    buff = NULL;

    return 0;
}
