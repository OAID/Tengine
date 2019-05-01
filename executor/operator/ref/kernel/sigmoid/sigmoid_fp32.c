

int sigmoid_fp32(float * data,int size,const sigmoid_param * param)
{
    for(int i=0;i<size;i++)
    {
        data[i] = T_MIN(data[i], 30.0f);
        data[i] = T_MAX(data[i], -30.0f);

        data[i] = 1 / (1 + exp(-data[i]));
    }
    return 0;
}