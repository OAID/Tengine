
static int ref_normalize_fp32(float* input, float* output, float* scale, const ref_normalize_param* param)
{
    int batch_num = param->input_n;
    int in_h = param->input_h;
    int in_w = param->input_w;
    int in_c = param->input_c;
    int in_offset = 0;
    int out_offset = 0;

    float* buff = ( float* )malloc(sizeof(float) * in_h * in_w);
    float* in_buf = input;
    float* out_buf = output;

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
                    float data = in_buf[in_offset];
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
                    float in_data = in_buf[in_offset];
                    out_buf[out_offset] = in_data * data;
                    if(scale)
                    {
                        out_buf[out_offset] = out_buf[out_offset] * scale[c];
                    }
                }
            }
        }
    }

    free(buff);
    buff = NULL;
    return 0;
}
