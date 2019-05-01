

int sigmoid_uint8(uint8_t * data,int size,const sigmoid_param * param)
{
    for(int i=0;i<size;i++)
    {
        float real_in = (data[i]-param->zero[0])*param->scale[0];
        float real_comp = T_MIN(real_in, 30);
        real_comp = T_MAX(real_in, -30);
        
        real_comp = 1 / (1 + exp(-real_comp));
        data[i] = round(real_comp/param->scale[1]) + param->zero[1];
    }
    return 0;
}