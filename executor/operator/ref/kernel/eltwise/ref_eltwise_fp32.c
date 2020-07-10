void elt_prod(int input_hw, int input_hw_1, int input_count4, int input1_count4, float* input0, float* input1,
              eltwise_param* param, float* output)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = (*input0++) * input1[0];
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = input0[i] * input1[i];
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            *output++ = (*input1++) * input0[0];
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

                    output[offset] = input0[offset] * input1[c];
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

                    output[offset] = input0[c] * input1[offset];
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

void elt_sum(int input_hw, int input_hw_1, int input_count4, int input1_count4, float* input0, float* input1,
             eltwise_param* param, float* output)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = (*input0++) + input1[0];
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = (*input0++) + (*input1++);
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            *output++ = (*input1++) + input0[0];
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

                    output[offset] = input0[offset] + input1[c];
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

                    output[offset] = input0[c] + input1[offset];
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

void elt_sub(int input_hw, int input_hw_1, int input_count4, int input1_count4, float* input0, float* input1,
             eltwise_param* param, float* output)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = (*input0++) - input1[0];
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = (*input0++) - (*input1++);
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            *output++ = input0[0] - (*input1++);
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

                    output[offset] = input0[offset] - input1[c];
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

                    output[offset] = input0[c] - input1[offset];
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

void elt_rsqrt(int input_count4, float* input0, float* output)
{
    for(int i = 0; i < input_count4; ++i)
    {
        *output++ = 1 / sqrt(input0[i]);
    }
    return;
}

void elt_divid(int input_hw, int input_hw_1, int input_count4, int input1_count4, float* input0, float* input1,
               eltwise_param* param, float* output)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = input0[i] / input1[0];
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = input0[i] / input1[i];
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            *output++ = input0[0] / (*input1++);
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

                    output[offset] = input0[offset] / input1[c];
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

                    output[offset] = input0[c] / input1[offset];
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

void elt_pow(int size1, int size2, float* input0, float* input1, float* output)
{
    if(size2 == 1)
    {
        for(int i = 0; i < size1; ++i)
        {
            *output++ = pow(input0[i], input1[0]);
        }
    }
    else if(size1 == size2)
    {
        for(int i = 0; i < size2; ++i)
        {
            *output++ = pow(input0[i], input1[i]);
        }
    }
    else if(size1 == 1)
    {
        for(int i = 0; i < size2; ++i)
        {
            *output++ = pow(input0[0], input1[i]);
        }
    }
    else
    {
        return;
    }
    return;
}

void elt_power(int size, float* input0, eltwise_param* param, float* output)
{
    for(int i = 0; i < size; ++i)
    {
        *output++ = pow((param->shift + param->pScale * input0[i]), param->power);
    }
    return;
}

static int ref_eltwise_fp32(float* input0, float* input1, float* output, eltwise_param* param)
{
    int input_hw = param->shape0[2] * param->shape0[3];
    int input_hw_1 = param->shape1[2] * param->shape1[3];
    int input_count4 = param->shape0[0] * param->shape0[1] * input_hw;
    int input1_count4 = param->shape1[0] * param->shape1[1] * input_hw_1;
    switch(param->type)
    {
        case 0:    // ELT_PROD
        case 1:    // ELT_PROD_SCALAR
        {
            elt_prod(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
            break;
        }
        case 2:    // ELT_SUM
        case 3:    // ELT_SUM_SCALAR
        {
            elt_sum(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
            break;
        }
        case 4:    // ELT_SUB
        case 5:    // ELT_SUB_SCALAR
        {
            elt_sub(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
            break;
        }
        case 6:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = std::max(input0[i], input1[i]);
            }
            break;
        }
        case 7:    // ELT_RSQRT
        {
            elt_rsqrt(input_count4, input0, output);
            break;
        }
        case 10:    // ELT_DIV
        {
            elt_divid(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
            break;
        }
        case 11:    // ELT_LOG
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = log(input0[i]);
            }
            break;
        }
        case 12:    // ELT_EXP
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = exp(input0[i]);
            }
            break;
        }
        case 13:    // ELT_SQRT
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = sqrt(input0[i]);
            }
            break;
        }
        case 14:    // ELT_FLOOR
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = floor(input0[i]);
            }
            break;
        }
        case 15:    // ELT_SQUARE
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = pow(input0[i], 2);
            }
            break;
        }
        case 16:    // ELT_POW
        {
            elt_pow(input_count4, input1_count4, input0, input1, output);
            break;
        }
        case 17:
        {
            // std::cout<<"Test in power op\n";
            elt_power(input1_count4, input0, param, output);
            break;
        }
        default:
            return -1;
            ;
    }
    return 0;
}
