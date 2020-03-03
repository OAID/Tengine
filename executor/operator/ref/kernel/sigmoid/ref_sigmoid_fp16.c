

int ref_sigmoid_fp16(__fp16* data, __fp16* out_data, int size, sigmoid_param* param)
{
    for(int i = 0; i < size; i++)
    {
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
        float realdata = fp16_to_fp32(data[i]);
        float realcompt = SIGMOID_MIN(realdata, 30.f);
        realcompt = SIGMOID_MAX(realdata, -30.f);
        realcompt = 1 / (1 + exp(-realcompt));
        out_data[i] = fp32_to_fp16(realcompt);

#else
        data[i] = SIGMOID_MIN(data[i], 30.0f);
        data[i] = SIGMOID_MAX(data[i], -30.0f);

        out_data[i] = 1 / (1 + exp(-data[i]));
#endif
    }
    return 0;
}