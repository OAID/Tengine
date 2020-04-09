static int ref_depthtospace_fp32(float* in_data, float* out_data, struct depthtospace_param* param)
{
    int size = param->size;

    for(int i = 0; i < size; i++)
        out_data[i] = in_data[i];
    
    return 0;
}