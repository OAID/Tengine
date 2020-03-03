

int ref_sigmoid_fp32(float* data, float* out_data, int size, sigmoid_param* param)
{
    for(int i = 0; i < size; i++)
    {
        data[i] = SIGMOID_MIN(data[i], 30.0f);
        data[i] = SIGMOID_MAX(data[i], -30.0f);

        out_data[i] = (float)1 / ((float)1 + exp(-data[i]));
    }
    return 0;
}