static int ref_deconv_uint8(const uint8_t* input, uint8_t* output, const uint8_t* kernel, const int* bias,
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

    float input_scale = param->scale[0];
    float weight_scale = param->scale[1];
    float output_scale = param->scale[2];

    float input_val = 0;
    float weight_val = 0;
    float bias_val = 0;
    uint8_t input_zero = param->zero[0];
    uint8_t weight_zero = param->zero[1];
    uint8_t output_zero = param->zero[2];

    float output_size = batch * output_c * group * output_h * output_w;
    float* output_tmp = ( float* )malloc(output_size);
    if(NULL == output_tmp)
    {
        return -1;
    }
    memset(output_tmp, 0, output_size);

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

                                        input_val = input_scale * (input[input_offset] - input_zero);
                                        weight_val = weight_scale * (kernel[kernel_offset] - weight_zero);
                                        output_tmp[output_offset] += input_val * weight_val;
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
                            bias_val = bias[g * output_c + c] * input_scale * weight_scale;
                            output_tmp[output_offset] += bias_val;
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
            output_tmp[n] = activation(output_tmp[n], param->activation);
        }
    }

    // quant the output
    for(n = 0; n < batch * group * output_c * output_w * output_h; n++)
    {
        int output_data = round(output_tmp[n] / output_scale + output_zero);
        if(output_data > 255)
            output[n] = 255;
        else if(output_data < 0)
            output[n] = 0;
        else
            output[n] = ( uint8_t )output_data;
    }

    free(output_tmp);
    return 0;
}
