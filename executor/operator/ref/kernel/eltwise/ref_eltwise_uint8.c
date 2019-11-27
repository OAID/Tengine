void elt_prod_uint8(int input_hw, int input_hw_1, int input_count4, int input1_count4, uint8_t* input0, uint8_t* input1,
                    uint8_t* output, eltwise_param* param)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = ((*input0++) - param->zero[0]) * param->scale[0];
            float real_input1 = (input1[0] - param->zero[1]) * param->scale[1];
            float result = real_input0 * real_input1;
            output[i] = round(result / param->scale[2]) + param->zero[2];
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = (input0[i] - param->zero[0]) * param->scale[0];
            float real_input1 = (input1[i] - param->zero[1]) * param->scale[1];
            float result = real_input0 * real_input1;
            output[i] = round(result / param->scale[2]) + param->zero[2];
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            float real_input0 = (input0[0] - param->zero[0]) * param->scale[0];
            float real_input1 = ((*input1++) - param->zero[1]) * param->scale[1];
            float result = real_input0 * real_input1;
            output[i] = round(result / param->scale[2]) + param->zero[2];
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
                    float real_input0 = (input0[offset] - param->zero[0]) * param->scale[0];
                    float real_input1 = (input1[c] - param->zero[1]) * param->scale[1];
                    float result = real_input0 * real_input1;
                    output[offset] = round(result / param->scale[2]) + param->zero[2];
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
                    float real_input0 = (input0[c] - param->zero[0]) * param->scale[0];
                    float real_input1 = (input1[offset] - param->zero[1]) * param->scale[1];
                    float result = real_input0 * real_input1;
                    output[offset] = round(result / param->scale[2]) + param->zero[2];
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

void elt_sum_uint8(int input_hw, int input_hw_1, int input_count4, int input1_count4, uint8_t* input0, uint8_t* input1,
                   uint8_t* output, eltwise_param* param)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = ((*input0++) - param->zero[0]) * param->scale[0];
            float real_input1 = (input1[0] - param->zero[1]) * param->scale[1];
            float result = real_input0 + real_input1;
            *output++ = round(result / param->scale[2]) + param->zero[2];
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = (input0[i] - param->zero[0]) * param->scale[0];
            float real_input1 = (input1[i] - param->zero[1]) * param->scale[1];
            float result = real_input0 + real_input1;
            *output++ = round(result / param->scale[2]) + param->zero[2];
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            float real_input0 = (input0[0] - param->zero[0]) * param->scale[0];
            float real_input1 = ((*input1++) - param->zero[1]) * param->scale[1];
            float result = real_input0 + real_input1;
            *output++ = round(result / param->scale[2]) + param->zero[2];
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
                    float real_input0 = (input0[offset] - param->zero[0]) * param->scale[0];
                    float real_input1 = (input1[c] - param->zero[1]) * param->scale[1];
                    float result = real_input0 + real_input1;
                    output[offset] = round(result / param->scale[2]) + param->zero[2];
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
                    float real_input0 = (input0[c] - param->zero[0]) * param->scale[0];
                    float real_input1 = (input1[offset] - param->zero[1]) * param->scale[1];
                    float result = real_input0 + real_input1;
                    output[offset] = round(result / param->scale[2]) + param->zero[2];
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

void elt_sub_uint8(int input_hw, int input_hw_1, int input_count4, int input1_count4, uint8_t* input0, uint8_t* input1,
                   uint8_t* output, eltwise_param* param)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = ((*input0++) - param->zero[0]) * param->scale[0];
            float real_input1 = (input1[0] - param->zero[1]) * param->scale[1];
            float result = real_input0 - real_input1;
            *output++ = round(result / param->scale[2]) + param->zero[2];
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = (input0[i]) * param->scale[0];
            float real_input1 = (input1[i]) * param->scale[1];
            float result = real_input0 - real_input1;
            *output = round(result / param->scale[2]);
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            float real_input0 = (input0[0] - param->zero[0]) * param->scale[0];
            float real_input1 = ((*input1++) - param->zero[1]) * param->scale[1];
            float result = real_input0 - real_input1;
            *output++ = round(result / param->scale[2]) + param->zero[2];
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
                    float real_input0 = (input0[offset] - param->zero[0]) * param->scale[0];
                    float real_input1 = (input1[c] - param->zero[1]) * param->scale[1];
                    float result = real_input0 - real_input1;
                    output[offset] = round(result / param->scale[2]) + param->zero[2];
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
                    float real_input0 = (input0[c] - param->zero[0]) * param->scale[0];
                    float real_input1 = (input1[offset] - param->zero[1]) * param->scale[1];
                    float result = real_input0 - real_input1;
                    output[offset] = round(result / param->scale[2]) + param->zero[2];
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

void elt_divid_uint8(int input_hw, int input_hw_1, int input_count4, int input1_count4, uint8_t* input0,
                     uint8_t* input1, uint8_t* output, eltwise_param* param)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = ((*input0++) - param->zero[0]) * param->scale[0];
            float real_input1 = (input1[0] - param->zero[1]) * param->scale[1];
            float result = real_input0 / real_input1;
            *output++ = round(result / param->scale[2]) + param->zero[2];
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            float real_input0 = (input0[i] - param->zero[0]) * param->scale[0];
            float real_input1 = (input1[i] - param->zero[1]) * param->scale[1];
            float result = real_input0 / real_input1;
            *output++ = round(result / param->scale[2]) + param->zero[2];
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            float real_input0 = (input0[0] - param->zero[0]) * param->scale[0];
            float real_input1 = ((*input1++) - param->zero[1]) * param->scale[1];
            float result = real_input0 / real_input1;
            *output++ = round(result / param->scale[2]) + param->zero[2];
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
                    float real_input0 = (input0[offset] - param->zero[0]) * param->scale[0];
                    float real_input1 = (input1[c] - param->zero[1]) * param->scale[1];
                    float result = real_input0 / real_input1;
                    output[offset] = round(result / param->scale[2]) + param->zero[2];
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
                    float real_input0 = (input0[c] - param->zero[0]) * param->scale[0];
                    float real_input1 = (input1[offset] - param->zero[1]) * param->scale[1];
                    float result = real_input0 / real_input1;
                    output[offset] = round(result / param->scale[2]) + param->zero[2];
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

static int ref_eltwise_uint8(uint8_t* input0, uint8_t* input1, uint8_t* output, eltwise_param* param)
{
    int input_hw = param->shape0[2] * param->shape0[3];
    int input_hw_1 = param->shape1[2] * param->shape1[3];
    int input_count4 = param->shape0[0] * param->shape0[1] * input_hw;
    int input1_count4 = param->shape1[0] * param->shape1[1] * input_hw_1;

    switch(param->type)
    {
        case 0:    // ELT_PROD
        {
            elt_prod_uint8(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, output, param);
            break;
        }
        case 2:
        {
            elt_sum_uint8(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, output, param);
            break;
        }
        case 4:
        {
            elt_sub_uint8(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, output, param);
            break;
        }
        case 6:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                float real_input0 = (input0[i]) * param->scale[0];
                float real_input1 = (input1[i]) * param->scale[0];
                *output++ = (std::max(real_input0, real_input1)) / param->scale[2] + param->zero[2];
            }
            break;
        }
        case 10:
        {
            elt_divid_uint8(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, output, param);
            break;
        }
        case 12:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                float real_input0 = (input0[i] - param->zero[0]) * param->scale[0];
                float result = exp(real_input0);
                *output++ = round(result / param->scale[2]) + param->zero[2];
            }
            break;
        }
        default:
            break;
    }
    return 0;
}
