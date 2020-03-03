
static int ref_conv_uint8(const uint8_t* input, uint8_t* output, const uint8_t* kernel, const int* bias, op_data* param)
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

    /* dequant input  */
    int input_size = batch * group * input_c * input_h * input_w;
    float* input_buf = ( float* )malloc(sizeof(float) * input_size);
    for(int i = 0; i < input_size; i++)
        input_buf[i] = (input[i] - param->zero[0]) * param->scale[0];

    /* dequant kernel  */
    int kernel_total = group * output_c * kernel_size;
    float* kernel_buf = ( float* )malloc(sizeof(float) * kernel_total);
    for(int i = 0; i < kernel_total; i++)
        kernel_buf[i] = (kernel[i] - param->zero[1]) * param->k_scale[0];

    /* dequant biases  */
    int bias_size = group * output_c;

    float* bias_buf = NULL;
    if(bias != NULL)
    {
        bias_buf = ( float* )malloc(sizeof(float) * bias_size);
        for(int i = 0; i < bias_size; i++)
            bias_buf[i] = bias[i] * param->scale[0] * param->k_scale[0];
    }

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
                        float total = 0.f;
                        if(param->layout == 0)
                        {
                            output_offset = n * group * output_c * output_h * output_w +
                                            g * output_c * output_h * output_w + c * output_h * output_w +
                                            h * output_w + w;
                        }
                        else
                        {
                            output_offset = n * group * output_c * output_h * output_w +
                                            h * output_w * group * output_c + w * group * output_c + g * output_c + c;
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
                                            if(group == 1)
                                                kernel_offset = c * kernel_size + kh * param->kernels[1] * input_c + kw * input_c + kc;
                                            else
                                                kernel_offset = kh * param->kernels[1] * group * output_c + kw * group * output_c + g * output_c + c;
                                        }
                                        total += (input_buf[input_offset] * kernel_buf[kernel_offset]);
                                    }
                                }
                            }
                        }
                        float bias_value = 0.0f;
                        if(bias != NULL)
                        {
                            bias_value = bias_buf[output_c * g + c];
                        }
                        total = activation(total + bias_value, param->activation);
                        int out = round(total / param->scale[1]) + param->zero[2];
                        if(out > 255)
                            out = 255;
                        if(out < 0)
                            out = 0;
                        output[output_offset] = out;
                    }
                }
            }
        }
    }
    if(bias != NULL)
        free(bias_buf);
    free(kernel_buf);
    free(input_buf);
    return 0;
}
