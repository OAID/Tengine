

int sigmoid_fp16(__fp16 * data,int size,const sigmoid_param * param)
{
    for(int i=0;i<size;i++)
    {
#if!defined( __ARM_ARCH) || __ARM_ARCH <8
        float realdata = fp16_to_fp32(data[i]);
        float realcompt = T_MIN(realdata, 30.f);
        realcompt = T_MAX(realdata, -30.f);
        realcompt = 1 / (1 + exp(-realcompt));
        data[i] = fp32_to_fp16(realcompt);

#else
        data[i] = T_MIN(data[i], 30.0f);
        data[i] = T_MAX(data[i], -30.0f);

        data[i] = 1 / (1 + exp(-data[i]));
#endif
    }
    return 0;
}