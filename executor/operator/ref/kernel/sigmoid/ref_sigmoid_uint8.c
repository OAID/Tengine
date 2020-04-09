

int ref_sigmoid_uint8(uint8_t* data, uint8_t* out_data, int size, sigmoid_param* param)
{
    for(int i = 0; i < size; i++)
    {
        float real_in = (data[i] - param->zero[0]) * param->scale[0];
        float real_comp = SIGMOID_MIN(real_in, 30);
        real_comp = SIGMOID_MAX(real_in, -30);

        real_comp = 1 / (1 + exp(-real_comp));
        out_data[i] = round(real_comp / param->scale[1]) + param->zero[1];
    }
    return 0;
}