

int sigmoid_int8(int8_t * data,int size,const sigmoid_param * param)
{
    for(int i=0;i<size;i++)
    {
        float real_in = data[i]*param->scale[0];
        float real_comp = T_MIN(real_in, 30);
        real_comp = T_MAX(real_in, -30);

        real_comp = 1 / (1 + exp(-real_comp));
        data[i] = round(real_comp*127);

    }
    return 0;
}