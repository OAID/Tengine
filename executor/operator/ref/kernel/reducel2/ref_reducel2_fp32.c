static int ref_reducel2_fp32(float *in_data,float* out_data,const reducel2_param* param)
{
    int in_size = 1;
    int out_size = 1;
   
    for(int i = 0; i < param->axis; i++)
    {
        out_size = out_size * param->dims[i];
    }
    for(int i = param->axis; i < 4;i++)
    {
        in_size = in_size *  param->dims[i];
    }
    
    for(int i = 0; i < out_size; i++)
    {
        float *data = in_data + i * in_size;
        float sum = 0;
        for(int j = 0; j < in_size;j++)
        {
            sum += data[j] * data[j];
        }

        out_data[i] = sqrt(sum);
    }
    return 0;
}
