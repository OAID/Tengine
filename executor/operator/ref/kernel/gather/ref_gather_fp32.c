static int ref_gather_fp32(float* input, int* input_indices, float* output, gather_param* param)
{
    float* out_ptr = output;
    float* in_ptr = input;
    int axis = param->axis;
    int outer_size = 1;
    int inner_size = 1;
    int axis_size = param->in_shape[axis];
    for(int i = 0; i < axis; i++)
    {
        outer_size *= param->in_shape[i];
    }
    for(int i = axis + 1; i < param->dim_size; i++)
    {
        inner_size *= param->in_shape[i];
    }

#if 0
    if(param->indices_num == 1)
    {
        for(int outer = 0; outer < outer_size; ++outer)
        {
            for(int i = 0; i < input_indices[0]; i++)
            {
                memcpy(out_ptr + (outer * input_indices[0] + i) * inner_size,
                       in_ptr + (outer* axis_size + i) * inner_size, inner_size * sizeof(float));
            }
        }
        
    }
    else
    {
#endif
        for(int outer = 0; outer < outer_size; ++outer)
        {
            for(int i = 0; i < param->indices_num; i++)
            {
                 memcpy(out_ptr + (outer * param->indices_num + i) * inner_size,in_ptr + (outer* axis_size + (int)input_indices[i]) * inner_size, inner_size* sizeof(float));
            }
        }

    //}


    return 0;
}
