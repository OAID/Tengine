
static int ref_conv_int8(const int8_t* input, int8_t* output, const int8_t* kernel, const int32_t* bias, op_data* param)
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
    float *i32_f32_scales = (float*)malloc(sizeof(float) * param->out_shape[0]);

    for(int i = 0; i < param->out_shape[0]; i++)
    {
        i32_f32_scales[i] = (param->scale[0] * param->k_scale[i]);
    }
    /* malloc  output */
    int output_size = group * batch * output_c * output_h * output_w;
    float* output_buf = ( float* )malloc(sizeof(float) * output_size);

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
                        int32_t total = 0.f;
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

                                        total += (input[input_offset] * kernel[kernel_offset]);
                                    }
                                }
                            }
                        }
                        int bias_value = 0;
                        if(bias)
                        {
                            bias_value = bias[output_c * g + c];
                        }
                        total = total + bias_value;
                        float fp32_out_val = total * i32_f32_scales[output_c * g + c];
                        output_buf[output_offset] = activation(fp32_out_val, param->activation);
                    }
                }
            }
        }
    }

    float scale = param->scale[1];

    for(int i = 0; i < output_size; i++)
    {
        int tmp  = round(output_buf[i] / scale);
        if(tmp > 127)
           tmp = 127;
        if(tmp < -127)
           tmp = -127;
        output[i] = (int8_t)tmp;
    }
    free(i32_f32_scales);
    free(output_buf);
    //free(kernel_buf);
    //free(input_buf);
    return 0;
}
