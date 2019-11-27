static int ref_addn_fp32(float** input, float* output, const ref_addn_param* param)
{
    int input_size = (param->input_size / sizeof(float));
    int in_num = param->in_num;

    for(int i = 0; i < in_num; ++i)
    {
        float* input_data = input[i];
        for(int j = 0; j < input_size; ++j)
        {
            output[j] += input_data[j];
        }
    }
    return 0;
}
