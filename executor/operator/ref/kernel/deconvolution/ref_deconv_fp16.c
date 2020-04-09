static inline void activation_fp16(__fp16* input, int activation)
{
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
    float tmp = fp16_to_fp32(*input);
#else
    __fp16 tmp = *input;
#endif
    if(activation >= 0)
    {
        if(tmp < 0)
            tmp = 0;
        if(activation == 1 && tmp > 1)
            tmp = 1;
        if(activation == 2 && tmp > 6)
            tmp = 6;
    }

#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
    *input = fp32_to_fp16(tmp);
#else
    *input = tmp;
#endif
}

static int ref_deconv_fp16(const __fp16* input, __fp16* output, const __fp16* kernel, const __fp16* bias,
                           const deconv_ref_param* param)
{
    int batch = param->batch;
    int group = param->group;
    int input_c = param->in_shape[0] / group;
    int input_h = param->in_shape[1];
    int input_w = param->in_shape[2];
    int output_c = param->out_shape[0] / group;
    int output_h = param->out_shape[1];
    int output_w = param->out_shape[2];
    int kernel_h = param->kernels[0];
    int kernel_w = param->kernels[1];
    int pad_h0 = param->pads[0];
    int pad_w0 = param->pads[1];
    int stride_h = param->strides[0];
    int stride_w = param->strides[1];
    int dilation_h = param->dilations[0];
    int dilation_w = param->dilations[1];

    int n, g, c, h, w, kc, k_h, k_w;
    int org_out_x = 0;
    int org_out_y = 0;
    int cur_out_x = 0;
    int cur_out_y = 0;

    int input_offset = 0;
    int kernel_offset = 0;
    int output_offset = 0;

    memset(( void* )output, 0, output_h * output_w * output_c * batch * group * sizeof(__fp16));

    for(n = 0; n < batch; ++n)
    {
        for(g = 0; g < group; ++g)
        {
            for(h = 0; h < input_h; h++)
            {
                for(w = 0; w < input_w; w++)
                {
                    org_out_x = w * stride_w - pad_w0;
                    org_out_y = h * stride_h - pad_h0;
                    for(kc = 0; kc < input_c; kc++)
                    {
                        if(param->layout == 0)
                        {
                            input_offset = n * group * input_c * input_h * input_w + g * input_c * input_h * input_w +
                                           kc * input_h * input_w + h * input_w + w;
                        }
                        else
                        {
                            input_offset = n * group * input_c * input_h * input_w + h * group * input_c * input_w +
                                           w * group * input_c + g * input_c + kc;
                        }
                        for(c = 0; c < output_c; c++)
                        {
                            for(k_h = 0; k_h < kernel_h; k_h++)
                            {
                                for(k_w = 0; k_w < kernel_w; k_w++)
                                {
                                    cur_out_x = org_out_x + k_w * dilation_w;
                                    cur_out_y = org_out_y + k_h * dilation_h;

                                    if(cur_out_x >= 0 && cur_out_x < output_w && cur_out_y >= 0 && cur_out_y < output_h)
                                    {
                                        if(param->layout == 0)
                                        {
                                            kernel_offset = g * output_c * input_c * kernel_h * kernel_w +
                                                            kc * output_c * kernel_h * kernel_w +
                                                            c * kernel_h * kernel_w + k_h * kernel_w + k_w;

                                            output_offset = n * group * output_c * output_w * output_h +
                                                            g * output_c * output_w * output_h +
                                                            c * output_w * output_h + cur_out_y * output_w + cur_out_x;
                                        }
                                        else
                                        {
                                            kernel_offset = g * output_c * input_c * kernel_h * kernel_w +
                                                            k_h * kernel_w * output_c + k_w * output_c + c;
                                            output_offset = n * output_h * output_w * output_c * group +
                                                            cur_out_y * group * output_w * output_c +
                                                            cur_out_x * group * output_c + g * output_c + c;
                                        }
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
                                        float tmp = 0;
                                        tmp = fp16_to_fp32(output[output_offset]);
                                        tmp +=
                                            (fp16_to_fp32(input[input_offset]) * fp16_to_fp32(kernel[kernel_offset]));
                                        output[output_offset] = fp32_to_fp16(tmp);
#else
                                        output[output_offset] += kernel[kernel_offset] * input[input_offset];
#endif
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if(NULL != bias)
    {
        for(n = 0; n < batch; n++)
        {
            for(g = 0; g < group; g++)
            {
                for(c = 0; c < output_c; c++)
                {
                    for(h = 0; h < output_h; h++)
                    {
                        for(w = 0; w < output_w; w++)
                        {
                            if(param->layout == 0)
                            {
                                output_offset = n * output_c * group * output_w * output_h +
                                                g * output_c * output_w * output_h + c * output_h * output_w +
                                                h * output_w + w;
                            }
                            else
                            {
                                output_offset = n * output_c * group * output_w * output_h +
                                                h * output_c * group * output_w + w * output_c * group + c;
                            }
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
                            float tmp = 0;
                            tmp = fp16_to_fp32(output[output_offset]);
                            tmp += (fp16_to_fp32(output[output_offset]) + fp16_to_fp32(bias[g * output_c + c]));
                            ;
                            output[output_offset] = fp32_to_fp16(tmp);
#else
                            output[output_offset] += bias[g * output_c + c];
#endif
                        }
                    }
                }
            }
        }
    }

    // activation
    if(param->activation >= 0)
    {
        for(n = 0; n < batch * group * output_c * output_w * output_h; n++)
        {
            activation_fp16(&output[n], param->activation);
        }
    }

    return 0;
}
