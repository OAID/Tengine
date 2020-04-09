static int ref_argmin_fp32(float* input, int* output, const ref_argmin_param* param)
{
    float min_value;
    int min_value_index;
    float current;

    int axis_size = param->axis_size;
    int outer_size = param->outer_size;
    int inner_size = param->inner_size;

    for(int outer = 0; outer < outer_size; ++outer)
    {
        for(int inner = 0; inner < inner_size; ++inner)
        {
            min_value = input[outer * axis_size * inner_size + inner];
            min_value_index = 0;
            for(int i = 1; i < axis_size; ++i)
            {
                current = input[(outer * axis_size + i) * inner_size + inner];
                if(current < min_value)
                {
                    min_value = current;
                    min_value_index = i;
                }
            }
            output[outer * inner_size + inner] = min_value_index;
        }
    }
    return 0;
}
