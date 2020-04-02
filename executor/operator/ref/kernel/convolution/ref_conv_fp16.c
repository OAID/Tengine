
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

static int ref_conv_fp16(const __fp16* input, __fp16* output, const __fp16* kernel, const __fp16* bias, op_data* param)
{
    int batch = param->batch;
    int group = param->group;
    int input_c = param->in_shape[0] / group;
    int input_h = param->in_shape[1];
    int input_w = param->in_shape[2];
    int output_c = param->out_shape[0] / group;
    int output_h = param->out_shape[1];
    int output_w = param->out_shape[2];

    int kernel_size = input_c * param->kernels[0] * param->kernels[1];
    int n, g, c, h, w, kc, kh, kw;
    int input_offset = 0;
    int kernel_offset = 0;
    int output_offset = 0;
    for(n = 0; n < batch; ++n)
    {
        for(g = 0; g < group; ++g)
        {
            for(c = 0; c < output_c; ++c)
            {
                for(h = 0; h < output_h; ++h)
                {
                    for(w = 0; w < output_w; ++w)
                    {
                        const int h_start = (h * param->strides[0]) - param->pads[0];
                        const int w_start = (w * param->strides[1]) - param->pads[1];
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
                        float total = bias ? fp16_to_fp32(bias[output_c * g + c]) : 0;
#else
                        __fp16 total = bias ? bias[output_c * g + c] : 0;
#endif
                        if(param->layout == 0)
                        {
                            output_offset = n * group * output_c * output_h * output_w +
                                            g * output_c * output_h * output_w + c * output_h * output_w +
                                            h * output_w + w;
                        }
                        else
                        {
                            output_offset = n * group * output_c * output_h * output_w +
                                            h * output_w * group * output_c + w * group * output_c + output_c * g + c;
                        }
                        for(kc = 0; kc < input_c; ++kc)
                        {
                            for(kh = 0; kh < param->kernels[0]; ++kh)
                            {
                                for(kw = 0; kw < param->kernels[1]; ++kw)
                                {
                                    const int cur_y = h_start + param->dilations[0] * kh;
                                    const int cur_x = w_start + param->dilations[1] * kw;
                                    // If the location is outside the bounds of the input image,
                                    // use zero as a default value.
                                    if((cur_x >= 0) && (cur_x < input_w) && (cur_y >= 0) && (cur_y < input_h))
                                    {
                                        if(param->layout == 0)
                                        {
                                            input_offset = n * group * input_c * input_h * input_w +
                                                           g * input_c * input_h * input_w + kc * input_h * input_w +
                                                           cur_y * input_w + cur_x;
                                            kernel_offset = g * output_c * kernel_size + c * kernel_size +
                                                            kc * param->kernels[0] * param->kernels[1] +
                                                            kh * param->kernels[1] + kw;
                                        }
                                        else
                                        {
                                            input_offset = n * group * input_c * input_h * input_w +
                                                           cur_y * input_w * input_c * group + cur_x * input_c * group +
                                                           g * input_c + kc;
                                            kernel_offset = c * kernel_size * group +
                                                            kh * param->kernels[1] * input_c * group +
                                                            kw * input_c * group + g * input_c + kc;
                                        }

#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
                                        total +=
                                            fp16_to_fp32(input[input_offset]) * fp16_to_fp32(kernel[kernel_offset]);
#else
                                        total += (input[input_offset] * kernel[kernel_offset]);
#endif
                                    }
                                }
                            }
                        }
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
                        total = activation(total, param->activation);
                        output[output_offset] = fp32_to_fp16(total);
#else
                        activation_fp16(&total, param->activation);
                        output[output_offset] = total;
#endif
                    }
                }
            }
        }
    }
    return 0;
}
