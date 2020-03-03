static int ref_spacetodepth_fp32(float* in_data, float* out_data, struct spacetodepth_param* param)
{
    int size = param->size;

    for(int i = 0; i < size; i++)
        out_data[i] = in_data[i];
    
    return 0;
}