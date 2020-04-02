void elt_prod_int8(int input_hw, int input_hw_1, int input_count4, int input1_count4, int out_size, float* output_buf,
                   int8_t* input0, int8_t* input1, eltwise_param* param)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = ((*input0++)) * param->scale[0];
            float real_input1 = (input1[0]) * param->scale[1];
            float result = real_input0 * real_input1;
            output_buf[i] = result;
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = (input0[i]) * param->scale[0];
            float real_input1 = (input1[i]) * param->scale[1];
            float result = real_input0 * real_input1;
            output_buf[i] = result;
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            float real_input0 = (input0[0]) * param->scale[0];
            float real_input1 = ((*input1++)) * param->scale[1];
            float result = real_input0 * real_input1;
            output_buf[i] = result;
        }
    }
    else if(param->shape0[1] == input1_count4)
    {
        for(int n = 0; n < param->shape0[0]; n++)
        {
            for(int c = 0; c < param->shape0[1]; c++)
            {
                for(int i = 0; i < input_hw; ++i)
                {
                    int offset = 0;
                    if(param->layout == 0)
                        offset = n * param->shape0[1] * input_hw + c * input_hw + i;
                    else
                        offset = n * param->shape0[1] * input_hw + i * param->shape0[1] + c;
                    float real_input0 = (input0[offset]) * param->scale[0];
                    float real_input1 = (input1[c]) * param->scale[1];
                    float result = real_input0 * real_input1;
                    output_buf[offset] = result;
                }
            }
        }
    }
    else if(param->shape1[1] == input_count4)
    {
        for(int n = 0; n < param->shape1[0]; n++)
        {
            for(int c = 0; c < param->shape1[1]; c++)
            {
                for(int i = 0; i < input_hw_1; ++i)
                {
                    int offset = 0;
                    if(param->layout == 0)
                        offset = n * param->shape1[1] * input_hw_1 + c * input_hw_1 + i;
                    else
                        offset = n * param->shape1[1] * input_hw_1 + i * param->shape1[1] + c;
                    float real_input0 = (input0[c]) * param->scale[0];
                    float real_input1 = (input1[offset]) * param->scale[1];
                    float result = real_input0 * real_input1;
                    output_buf[offset] = result;
                }
            }
        }
    }
    else
    {
        return;
    }

    return;
}

void elt_sum_int8(int input_hw, int input_hw_1, int input_count4, int input1_count4, int out_size, float* output_buf,
                  int8_t* input0, int8_t* input1, eltwise_param* param)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = ((*input0++)) * param->scale[0];
            float real_input1 = (input1[0]) * param->scale[1];
            float result = real_input0 + real_input1;
            output_buf[i] = result;
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = (input0[i]) * param->scale[0];
            float real_input1 = (input1[i]) * param->scale[1];
            float result = real_input0 + real_input1;
            output_buf[i] = result;
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            float real_input0 = (input0[0]) * param->scale[0];
            float real_input1 = ((*input1++)) * param->scale[1];
            float result = real_input0 + real_input1;
            output_buf[i] = result;
        }
    }
    else if(param->shape0[1] == input1_count4)
    {
        for(int n = 0; n < param->shape0[0]; n++)
        {
            for(int c = 0; c < param->shape0[1]; c++)
            {
                for(int i = 0; i < input_hw; ++i)
                {
                    int offset = 0;
                    if(param->layout == 0)
                        offset = n * param->shape0[1] * input_hw + c * input_hw + i;
                    else
                        offset = n * param->shape0[1] * input_hw + i * param->shape0[1] + c;
                    float real_input0 = (input0[offset]) * param->scale[0];
                    float real_input1 = (input1[c]) * param->scale[1];
                    float result = real_input0 + real_input1;
                    output_buf[offset] = result;
                }
            }
        }
    }
    else if(param->shape1[1] == input_count4)
    {
        for(int n = 0; n < param->shape1[0]; n++)
        {
            for(int c = 0; c < param->shape1[1]; c++)
            {
                for(int i = 0; i < input_hw_1; ++i)
                {
                    int offset = 0;
                    if(param->layout == 0)
                        offset = n * param->shape1[1] * input_hw_1 + c * input_hw_1 + i;
                    else
                        offset = n * param->shape1[1] * input_hw_1 + i * param->shape1[1] + c;
                    float real_input0 = (input0[c]) * param->scale[0];
                    float real_input1 = (input1[offset]) * param->scale[1];
                    float result = real_input0 + real_input1;
                    output_buf[offset] = result;
                }
            }
        }
    }
    else
    {
        return;
    }
    return;
}

void elt_sub_int8(int input_hw, int input_hw_1, int input_count4, int input1_count4, int out_size, float* output_buf,
                  int8_t* input0, int8_t* input1, eltwise_param* param)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = ((*input0++)) * param->scale[0];
            float real_input1 = (input1[0]) * param->scale[1];
            float result = real_input0 - real_input1;
            output_buf[i] = result;
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = (input0[i]) * param->scale[0];
            float real_input1 = (input1[i]) * param->scale[1];
            float result = real_input0 - real_input1;
            *output_buf = result;
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            float real_input0 = (input0[0]) * param->scale[0];
            float real_input1 = ((*input1++)) * param->scale[1];
            float result = real_input0 - real_input1;
            output_buf[i] = result;
        }
    }
    else if(param->shape0[1] == input1_count4)
    {
        for(int n = 0; n < param->shape0[0]; n++)
        {
            for(int c = 0; c < param->shape0[1]; c++)
            {
                for(int i = 0; i < input_hw; ++i)
                {
                    int offset = 0;
                    if(param->layout == 0)
                        offset = n * param->shape0[1] * input_hw + c * input_hw + i;
                    else
                        offset = n * param->shape0[1] * input_hw + i * param->shape0[1] + c;
                    float real_input0 = (input0[offset]) * param->scale[0];
                    float real_input1 = (input1[c]) * param->scale[1];
                    float result = real_input0 - real_input1;
                    output_buf[offset] = result;
                }
            }
        }
    }
    else if(param->shape1[1] == input_count4)
    {
        for(int n = 0; n < param->shape1[0]; n++)
        {
            for(int c = 0; c < param->shape1[1]; c++)
            {
                for(int i = 0; i < input_hw_1; ++i)
                {
                    int offset = 0;
                    if(param->layout == 0)
                        offset = n * param->shape1[1] * input_hw_1 + c * input_hw_1 + i;
                    else
                        offset = n * param->shape1[1] * input_hw_1 + i * param->shape1[1] + c;
                    float real_input0 = (input0[c]) * param->scale[0];
                    float real_input1 = (input1[offset]) * param->scale[1];
                    float result = real_input0 - real_input1;
                    output_buf[offset] = result;
                }
            }
        }
    }
    else
    {
        return;
    }
    return;
}
void elt_divid_int8(int input_hw, int input_hw_1, int input_count4, int input1_count4, int out_size, float* output_buf,
                    int8_t* input0, int8_t* input1, eltwise_param* param)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = ((*input0++)) * param->scale[0];
            float real_input1 = (input1[0]) * param->scale[1];
            float result = real_input0 / real_input1;
            output_buf[i] = result;
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = (input0[i]) * param->scale[0];
            float real_input1 = (input1[i]) * param->scale[1];
            float result = real_input0 / real_input1;
            output_buf[i] = result;
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            float real_input0 = (input0[0]) * param->scale[0];
            float real_input1 = ((*input1++)) * param->scale[1];
            float result = real_input0 / real_input1;
            output_buf[i] = result;
        }
    }
    else if(param->shape0[1] == input1_count4)
    {
        for(int n = 0; n < param->shape0[0]; n++)
        {
            for(int c = 0; c < param->shape0[1]; c++)
            {
                for(int i = 0; i < input_hw; ++i)
                {
                    int offset = 0;
                    if(param->layout == 0)
                        offset = n * param->shape0[1] * input_hw + c * input_hw + i;
                    else
                        offset = n * param->shape0[1] * input_hw + i * param->shape0[1] + c;
                    float real_input0 = (input0[offset]) * param->scale[0];
                    float real_input1 = (input1[c]) * param->scale[1];
                    float result = real_input0 / real_input1;
                    output_buf[offset] = result;
                }
            }
        }
    }
    else if(param->shape1[1] == input_count4)
    {
        for(int n = 0; n < param->shape1[0]; n++)
        {
            for(int c = 0; c < param->shape1[1]; c++)
            {
                for(int i = 0; i < input_hw_1; ++i)
                {
                    int offset = 0;
                    if(param->layout == 0)
                        offset = n * param->shape1[1] * input_hw_1 + c * input_hw_1 + i;
                    else
                        offset = n * param->shape1[1] * input_hw_1 + i * param->shape1[1] + c;
                    float real_input0 = (input0[c]) * param->scale[0];
                    float real_input1 = (input1[offset]) * param->scale[1];
                    float result = real_input0 / real_input1;
                    output_buf[offset] = result;
                }
            }
        }
    }
    else
    {
        return;
    }

    return;
}
static int ref_eltwise_int8(int8_t* input0, int8_t* input1, int8_t* output, eltwise_param* param)
{
    int input_hw = param->shape0[2] * param->shape0[3];
    int input_hw_1 = param->shape1[2] * param->shape1[3];
    int input_count4 = param->shape0[0] * param->shape0[1] * input_hw;
    int input1_count4 = param->shape1[0] * param->shape1[1] * input_hw_1;
    int out_size = input_count4 > input1_count4 ? input_count4 : input1_count4;

    float* output_buf = ( float* )malloc(sizeof(float) * out_size);
    switch(param->type)
    {
        case 0:    // ELT_PROD
        {
            elt_prod_int8(input_hw, input_hw_1, input_count4, input1_count4, out_size, output_buf, input0, input1,
                          param);
            break;
        }
        case 2:    // ELT_SUM
        {
            elt_sum_int8(input_hw, input_hw_1, input_count4, input1_count4, out_size, output_buf, input0, input1,
                         param);
            break;
        }

        case 4:    // ELT_SUB
        {
            elt_sub_int8(input_hw, input_hw_1, input_count4, input1_count4, out_size, output_buf, input0, input1,
                         param);
            break;
        }
        case 6:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                float real_input0 = (input0[i]) * param->scale[0];
                float real_input1 = (input1[i]) * param->scale[0];
                output_buf[i] = std::max(real_input0, real_input1);
            }
            break;
        }
        case 10:    // ELT_DIV
        {
            elt_divid_int8(input_hw, input_hw_1, input_count4, input1_count4, out_size, output_buf, input0, input1,
                           param);
            break;
        }
        case 12:    // ELT_EXP
        {
            for(int i = 0; i < input_count4; ++i)
            {
                float real_input0 = (input0[i]) * param->scale[0];
                float result = exp(real_input0);
                output_buf[i] = result;
            }
            break;
        }
        default:
            break;
    }

    for(int i = 0; i < out_size; i++)
    {
        int8_t tmp = round(output_buf[i] / param->scale[2]);
        if (tmp > 127)
            tmp = 127;
        else if (tmp < -127)
            tmp = -127;
        output[i] = tmp;
    }

    return 0;
}
