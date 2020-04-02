void logical_and(int input_hw, int input_hw_1, int input_count4, int input1_count4, float* input0, float* input1,
             logical_param* param, float* output)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = (*input0++) && input1[0];
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = (*input0++) && (*input1++);
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            *output++ = (*input1++) && input0[0];
        }
    }
    
    else
    {
        return;
    }
    return;
}

void logical_or(int input_hw, int input_hw_1, int input_count4, int input1_count4, float* input0, float* input1,
             logical_param* param, float* output)
{
    if(input1_count4 == 1)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = (*input0++) || input1[0];
        }
    }
    else if(input_count4 == input1_count4)
    {
        for(int i = 0; i < input_count4; ++i)
        {
            *output++ = (*input0++) || (*input1++);
        }
    }
    else if(input_count4 == 1)
    {
        for(int i = 0; i < input1_count4; ++i)
        {
            *output++ = (*input1++) || input0[0];
        }
    }
    
    else
    {
        return;
    }
    return;
}


static int ref_logical_fp32(float* input0, float* input1, float* output, logical_param* param)
{
    int input_hw = param->shape0[2] * param->shape0[3];
    int input_hw_1 = param->shape1[2] * param->shape1[3];
    int input_count4 = param->shape0[0] * param->shape0[1] * input_hw;
    int input1_count4 = param->shape1[0] * param->shape1[1] * input_hw_1;

    switch(param->type)
    {
        case 0:     // LogicalAnd
        {
            logical_and(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
            break;
        }
        case 1:     // LogicalOr
        {
            logical_or(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
            break;
        }
        default:
            return -1;
            ;
    }
    return 0;
}
