static int ref_argmax_fp32(float* input, int* output, const ref_argmax_param* param)
{
    float max_value;
    int max_value_index;
    float current;

    int axis_size = param->axis_size;
    int outer_size = param->outer_size;
    int inner_size = param->inner_size;

    for(int outer = 0; outer < outer_size; ++outer)
    {
        for(int inner = 0; inner < inner_size; ++inner)
        {
            max_value = input[outer * axis_size * inner_size + inner];
            max_value_index = 0;
            for(int i = 1; i < axis_size; ++i)
            {
                current = input[(outer * axis_size + i) * inner_size + inner];
                if(current > max_value)
                {
                    max_value = current;
                    max_value_index = i;
                }
            }
            output[outer * inner_size + inner] = max_value_index;
        }
    }
    return 0;
}
