static int ref_batchnorm_fp16(__fp16* input, __fp16* output, const ref_batchnorm_param* param)
{
    float* scale_mean = param->scale_mean;
    float* scale_var_inv = param->scale_var_inv;
    float* gamma = param->gamma;
    float* beta = param->beta;

    int img_size = param->input_c * param->input_h * param->input_w;
    float* out_f32 = ( float* )malloc(sizeof(float) * img_size * param->input_n);
    memset(out_f32, 0, sizeof(float) * img_size);
    for(int n = 0; n < param->input_n; ++n)
    {
        for(int h = 0; h < param->input_h; ++h)
        {
            for(int w = 0; w < param->input_w; ++w)
            {
                for(int c = 0; c < param->input_c; ++c)
                {
                    float s_mean = scale_mean[c];
                    float s_var = scale_var_inv[c];
                    float s_val1 = s_mean;
                    float s_val2 = s_var;
                    if(!param->iscaffe)
                    {
                        float s_gamma = gamma[c];
                        float s_beta = beta[c];
                        s_val1 = s_beta + s_gamma * s_mean;
                        s_val2 = s_gamma * s_var;
                    }
                    int offset = 0;
                    if(TENGINE_LAYOUT_NCHW == param->layout)
                    {
                        offset = n * img_size + c * param->input_h * param->input_w + h * param->input_w + w;
                    }
                    else
                    {
                        offset = n * img_size + h * param->input_w * param->input_c + w * param->input_c + c;
                    }
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
                    float data = fp16_to_fp32(input[offset]);
#else
                    __fp16 data = input[offset];
#endif
                    out_f32[offset] = data * s_val2 + s_val1;
                }
            }
        }
    }
    for(int j = 0; j < img_size * param->input_n; ++j)
    {
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
        output[j] = fp32_to_fp16(out_f32[j]);
#else
        output[j] = ( __fp16 )out_f32[j];
#endif
    }
    free(out_f32);
    out_f32 = NULL;

    return 0;
}
