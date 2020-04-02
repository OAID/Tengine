static int ref_batchnorm_int8(int8_t* input, int8_t* output, ref_batchnorm_param* param)
{
    float* scale_mean = param->scale_mean;
    float* scale_var_inv = param->scale_var_inv;
    float* gamma = param->gamma;
    float* beta = param->beta;

    int img_size = param->input_c * param->input_h * param->input_w;
    float* out_f32 = ( float* )malloc(sizeof(float) * img_size * param->input_n);
    memset(out_f32, 0, sizeof(float) * img_size * param->input_n);
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
                    float data = ( float )param->in_scale * (input[offset] - param->in_zero);
                    out_f32[offset] = data * s_val2 + s_val1;
                }
            }
        }
    }
    float output_max = 0.0f;
    for(int i = 0; i < img_size * param->input_n; i++)
    {
        if(output_max < fabs(out_f32[i]))
            output_max = fabs(out_f32[i]);
    }
    param->out_scale = output_max / 127;
    param->out_zero = 0;
    for(int i = 0; i < img_size * param->input_n; i++)
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
