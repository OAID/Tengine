
static int ref_normalize_fp16(__fp16* input, __fp16* output, __fp16* scale, const ref_normalize_param* param)
{
    int batch_num = param->input_n;
    int in_h = param->input_h;
    int in_w = param->input_w;
    int in_c = param->input_c;
    int in_offset = 0;
    int out_offset = 0;

    float* buff = ( float* )malloc(sizeof(float) * in_h * in_w);
    __fp16* in_buf = input;
    __fp16* out_buf = output;

    for(int n = 0; n < batch_num; ++n)
    {
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
                        in_offset = n * in_h * in_w * in_c + c * in_h * in_w + h * in_w + w;
                    }
                    else    // nhwc
                    {
                        in_offset = n * in_h * in_w * in_c + h * in_w * in_c + w * in_c + c;
                    }
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
                    float data = fp16_to_fp32(in_buf[in_offset]);
#else
                    __fp16 data = in_buf[in_offset];
#endif
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
                        out_offset = n * in_h * in_w * in_c + c * in_h * in_w + h * in_w + w;
                        in_offset = out_offset;
                    }
                    else    // nhwc
                    {
                        out_offset = n * in_h * in_w * in_c + h * in_w * in_c + w * in_c + c;
                        in_offset = out_offset;
                    }
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
                    float in_data = fp16_to_fp32(in_buf[in_offset]);
#else
                    __fp16 in_data = in_buf[in_offset];
#endif
                    float data = buff[buff_idx];

                    float out_data = in_data * data;
                    if(scale)
                    {
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
                        float scale_data = fp16_to_fp32(scale[c]);
#else
                        __fp16 scale_data = scale[c];
#endif

                        out_data = out_data * scale_data;
                    }

#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
                    out_buf[out_offset] = fp32_to_fp16(out_data);
#else
                    out_buf[out_offset] = ( __fp16 )out_data;
#endif
                }
            }
        }
    }

    free(buff);
    buff = NULL;
    return 0;
}
