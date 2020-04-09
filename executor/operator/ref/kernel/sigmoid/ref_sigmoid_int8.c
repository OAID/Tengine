

int ref_sigmoid_int8(int8_t* data, int8_t* out_data, int size, sigmoid_param* param)
{
    float* tmp = ( float* )malloc(size * sizeof(float));
    float max_val = 0.f;
    for(int i = 0; i < size; i++)
    {
        float real_in = data[i] * param->scale[0];

        float real_comp = SIGMOID_MIN(real_in, 30);
        real_comp = SIGMOID_MAX(real_in, -30);

        real_comp = 1 / (1 + exp(-real_comp));
        tmp[i] = real_comp;
        if(max_val < fabs(real_comp))
            max_val = fabs(real_comp);
    }

    float out_scale = max_val / 127;

    for(int i = 0; i < size; i++)
    {
        out_data[i] = round(tmp[i] / out_scale);
    }
    param->scale[1] = out_scale;

    free(tmp);
    return 0;
}