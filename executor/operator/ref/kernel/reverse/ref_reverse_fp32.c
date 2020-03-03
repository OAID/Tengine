static int ref_reverse_fp32(float* input, int* input_axis, float* output, reverse_param* param)
{
    float* out_ptr = output;
    float* in_ptr = input;
    int* axis_ptr = input_axis;
    int axis = axis_ptr[0];

    int in_w = param->in_shape[3];
    int in_hw = param->in_shape[2] * in_w;
    int in_chw = param->in_shape[1] * in_hw;

    if (param->dim_size == 4)
    {
        if (axis == 0 || axis == -4)
        {
            for(int i = 0; i < param->in_shape[0]; i++)
            {
                for(int j = 0; j < param->in_shape[1]; j++)
                {
                    for(int y = 0; y < param->in_shape[2]; y++)
                    {
                        for(int x = 0; x < param->in_shape[3]; x++)
                        {
                            out_ptr[i * in_chw + j * in_hw + y * in_w + x] = in_ptr[(param->in_shape[0] - 1 - i) * in_chw + j * in_hw + y * in_w + x];
                        }
                    }
                }
            }
        }

        else if(axis == 1 || axis == -3)
        {
            for(int i = 0; i < param->in_shape[0]; i++)
            {
                for(int j = 0; j < param->in_shape[1]; j++)
                {
                    for(int y = 0; y < param->in_shape[2]; y++)
                    {
                        for(int x = 0; x < param->in_shape[3]; x++)
                        {
                            out_ptr[i * in_chw + j * in_hw + y * in_w + x] = in_ptr[i * in_chw + (param->in_shape[1] - 1 - j) * in_hw + y * in_w + x];
                        }
                    }
                }
            }
        }

        else if(axis == 2 || axis == -2)
        {
            for(int i = 0; i < param->in_shape[0]; i++)
            {
                for(int j = 0; j < param->in_shape[1]; j++)
                {
                    for(int y = 0; y < param->in_shape[2]; y++)
                    {
                        for(int x = 0; x < param->in_shape[3]; x++)
                        {
                            out_ptr[i * in_chw + j * in_hw + y * in_w + x] = in_ptr[i * in_chw + j * in_hw + (param->in_shape[2] - 1 - y) * in_w + x];
                        }
                    }
                }
            }
        }

        else if(axis == 3 || axis == -1)
        {
            for(int i = 0; i < param->in_shape[0]; i++)
            {
                for(int j = 0; j < param->in_shape[1]; j++)
                {
                    for(int y = 0; y < param->in_shape[2]; y++)
                    {
                        for(int x = 0; x < param->in_shape[3]; x++)
                        {
                            out_ptr[i * in_chw + j * in_hw + y * in_w + x] = in_ptr[i * in_chw + j * in_hw + y * in_w + (param->in_shape[3] - 1 - x)];
                        }
                    }
                }
            }
        }
    }

    else
    {
        return -1;
    }

    return 0;
    
}