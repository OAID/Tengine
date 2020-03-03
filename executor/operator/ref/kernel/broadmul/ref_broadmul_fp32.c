static int ref_broadmul_fp32(float* in0,float *in1,float* out, const ref_broadmul_param* param)
{
    int out_size = param->out_size;
    int in_size = param->in_size;
    int on_size = param->on_size;
    
    for(int o = 0; o < out_size; o++)
    {
        for(int j = 0; j < on_size;j++)
        {
            float data1 = in1[j];
            for(int i = 0; i < in_size; i++)
            {
                int index = (o * on_size + j) * in_size + i;
                out[index] = in0[index]*data1;
            }
        }
    }
   
    return 0;
}
