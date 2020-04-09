void elt_prod_fp16(int input_hw, int input_hw_1, int input_count4, int input1_count4, __fp16* input0, __fp16* input1,
                   eltwise_param* param, __fp16* output)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = fp32_to_fp16(fp16_to_fp32(*input0++) * fp16_to_fp32(input1[0]));
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = fp32_to_fp16(fp16_to_fp32(input0[i]) * fp16_to_fp32(input1[i]));
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            *output++ = fp32_to_fp16(fp16_to_fp32(*input1++) * fp16_to_fp32(input0[0]));
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
                    output[offset] = fp32_to_fp16(fp16_to_fp32(input0[offset]) * fp16_to_fp32(input1[c]));
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
                    output[offset] = fp32_to_fp16(fp16_to_fp32(input0[c]) * fp16_to_fp32(input1[offset]));
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
void elt_sum_fp16(int input_hw, int input_hw_1, int input_count4, int input1_count4, __fp16* input0, __fp16* input1,
                  eltwise_param* param, __fp16* output)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = fp32_to_fp16(fp16_to_fp32(*input0++) + fp16_to_fp32(input1[0]));
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = fp32_to_fp16(fp16_to_fp32(*input0++) + fp16_to_fp32(*input1++));
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            *output++ = fp32_to_fp16(fp16_to_fp32(*input1++) + fp16_to_fp32(input0[0]));
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
                    output[offset] = fp32_to_fp16(fp16_to_fp32(input0[offset]) + fp16_to_fp32(input1[c]));
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
                    output[offset] = fp32_to_fp16(fp16_to_fp32(input0[c]) + fp16_to_fp32(input1[offset]));
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

void elt_sub_fp16(int input_hw, int input_hw_1, int input_count4, int input1_count4, __fp16* input0, __fp16* input1,
                  eltwise_param* param, __fp16* output)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = fp32_to_fp16(fp16_to_fp32(*input0++) - fp16_to_fp32(input1[0]));
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = fp32_to_fp16(fp16_to_fp32(*input0++) - fp16_to_fp32(*input1++));
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            *output++ = fp32_to_fp16(fp16_to_fp32(input0[0]) - fp16_to_fp32(*input1++));
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
                    output[offset] = fp32_to_fp16(fp16_to_fp32(input0[offset]) - fp16_to_fp32(input1[c]));
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
                    output[offset] = fp32_to_fp16(fp16_to_fp32(input0[c]) - fp16_to_fp32(input1[offset]));
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
void elt_divid_fp16(int input_hw, int input_hw_1, int input_count4, int input1_count4, __fp16* input0, __fp16* input1,
                    eltwise_param* param, __fp16* output)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = fp32_to_fp16(fp16_to_fp32(*input0++) / fp16_to_fp32(input1[0]));
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = fp32_to_fp16(fp16_to_fp32(input0[i]) / fp16_to_fp32(input1[i]));
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            *output++ = fp32_to_fp16(fp16_to_fp32(input0[0]) / fp16_to_fp32(*input1++));
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
                    output[offset] = fp32_to_fp16(fp16_to_fp32(input0[offset]) / fp16_to_fp32(input1[c]));
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
                    output[offset] = fp32_to_fp16(fp16_to_fp32(input0[c]) / fp16_to_fp32(input1[offset]));
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
void elt_pow_fp16(int input_hw, int input_hw_1, int input_count4, int input1_count4, __fp16* input0, __fp16* input1,
                  eltwise_param* param, __fp16* output)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = fp32_to_fp16(pow(fp16_to_fp32(input0[i]), fp16_to_fp32(input1[0])));
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = fp32_to_fp16(pow(fp16_to_fp32(input0[i]), fp16_to_fp32(input1[i])));
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            *output++ = fp32_to_fp16(pow(fp16_to_fp32(input0[0]), fp16_to_fp32(input1[i])));
        }
    }
    else
    {
        return;
    }

    return;
}

static int ref_eltwise_fp16(__fp16* input0, __fp16* input1, __fp16* output, eltwise_param* param)
{
    int input_hw = param->shape0[2] * param->shape0[3];
    int input_hw_1 = param->shape1[2] * param->shape1[3];
    int input_count4 = param->shape0[0] * param->shape0[1] * input_hw;
    int input1_count4 = param->shape1[0] * param->shape1[1] * input_hw_1;
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
    switch(param->type)
    {
        case 0:    // ELT_PROD
        {
            elt_prod_fp16(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
            break;
        }
        case 2:    // ELT_SUM
        {
            elt_sum_fp16(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
            break;
        }
        case 4:    // ELT_SUB
        {
            elt_sub_fp16(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
            break;
        }
        case 6:    // ELT_RSQRT
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = fp32_to_fp16(std::max(fp16_to_fp32(input0[i]), fp16_to_fp32(input1[i])));
            }
            break;
        }

        case 7:    // ELT_RSQRT
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = fp32_to_fp16(1 / sqrt(fp16_to_fp32(input0[i])));
            }
            break;
        }
        case 10:    // ELT_DIV
        {
            elt_divid_fp16(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
            break;
        }
        case 11:    // ELT_LOG
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = fp32_to_fp16(log(fp16_to_fp32(input0[i])));
            }
            break;
        }
        case 12:    // ELT_EXP
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = fp32_to_fp16(exp(fp16_to_fp32(input0[i])));
            }
            break;
        }
        case 13:    // ELT_SQRT
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = fp32_to_fp16(sqrt(fp16_to_fp32(input0[i])));
            }
            break;
        }
        case 14:    // ELT_FLOOR
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = fp32_to_fp16(floor(fp16_to_fp32(input0[i])));
            }
            break;
        }
        case 15:    // ELT_SQUARE
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = fp32_to_fp16(pow(fp16_to_fp32(input0[i]), 2));
            }
            break;
        }
        case 16:    // ELT_POW
        {
            elt_pow_fp16(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
            break;
        }
        default:
            return -1;
            ;
    }

#else
    switch(param->type)
    {
        case 0:    // ELT_PROD
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
                            if(param->layout == 0)
                            {
                                int offset = n * param->shape0[1] * input_hw + c * input_hw + i;
                                output[offset] = input0[offset] * input1[c];
                            }
                            else
                            {
                                int offset = n * param->shape0[1] * input_hw + i * param->shape0[1] + c;
                                output[offset] = input0[offset] * input1[c];
                            }
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
                            if(param->layout == 0)
                            {
                                int offset = n * param->shape1[1] * input_hw_1 + c * input_hw_1 + i;
                                output[offset] = input0[c] * input1[offset];
                            }
                            else
                            {
                                int offset = n * param->shape1[1] * input_hw_1 + i * param->shape1[1] + c;

                                output[offset] = input0[c] * input1[offset];
                            }
                        }
                    }
                }
            }
            else
                return -1;
            break;
        }
        case 2:    // ELT_SUM
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
                            if(param->layout == 0)
                            {
                                int offset = n * param->shape0[1] * input_hw + c * input_hw + i;
                                output[offset] = input0[offset] + input1[c];
                            }
                            else
                            {
                                int offset = n * param->shape0[1] * input_hw + i * param->shape0[1] + c;
                                output[offset] = input0[offset] + input1[c];
                            }
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
                            if(param->layout == 0)
                            {
                                int offset = n * param->shape1[1] * input_hw_1 + c * input_hw_1 + i;
                                output[offset] = input0[c] + input1[offset];
                            }
                            else
                            {
                                int offset = n * param->shape1[1] * input_hw_1 + i * param->shape1[1] + c;

                                output[offset] = input0[c] + input1[offset];
                            }
                        }
                    }
                }
            }
            else
                return -1;
            break;
        }
        case 4:    // ELT_SUB
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
                            if(param->layout == 0)
                            {
                                int offset = n * param->shape0[1] * input_hw + c * input_hw + i;
                                output[offset] = input0[offset] - input1[c];
                            }
                            else
                            {
                                int offset = n * param->shape0[1] * input_hw + i * param->shape0[1] + c;
                                output[offset] = input0[offset] - input1[c];
                            }
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
                            if(param->layout == 0)
                            {
                                int offset = n * param->shape1[1] * input_hw_1 + c * input_hw_1 + i;
                                output[offset] = input0[c] - input1[offset];
                            }
                            else
                            {
                                int offset = n * param->shape1[1] * input_hw_1 + i * param->shape1[1] + c;

                                output[offset] = input0[c] - input1[offset];
                            }
                        }
                    }
                }
            }
            else
                return -1;
            break;
        }

        case 6:    // ELT_RSQRT
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = std::max(input0[i], input1[i]);
            }
            break;
        }
        case 7:    // ELT_RSQRT
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = 1 / sqrt(input0[i]);
            }
            break;
        }
        case 10:    // ELT_DIV
        {
            if(input1_count4 == 1)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = (*input0++) / input1[0];
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
                            // nchw
                            if(param->layout == 0)
                            {
                                int offset = n * param->shape0[1] * input_hw + c * input_hw + i;
                                output[offset] = input0[offset] / input1[c];
                            }
                            // nhwc
                            else
                            {
                                int offset = n * input_hw * param->shape0[1] + i * param->shape0[1] + c;
                                output[offset] = input0[offset] / input1[c];
                            }
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
                            if(param->layout == 0)
                            {
                                int offset = n * param->shape1[1] * input_hw_1 + c * input_hw_1 + i;
                                output[offset] = input0[c] / input1[offset];
                            }
                            else
                            {
                                int offset = n * param->shape1[1] * input_hw_1 + i * param->shape1[1] + c;

                                output[offset] = input0[c] / input1[offset];
                            }
                        }
                    }
                }
            }
            else
            {
                return -1;
            }
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
            if(input1_count4 == 1)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = pow(input0[i], input1[0]);
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = pow(input0[i], input1[i]);
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    *output++ = pow(input0[0], input1[i]);
                }
            }
            else
            {
                return -1;
            }
            break;
        }
        default:
            return -1;
            ;
    }
#endif
    return 0;
}
