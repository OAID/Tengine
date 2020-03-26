
static int ref_fm_int8(const int8_t* input, int8_t* output, const int8_t* weight, const int32_t* bias, feature_match_data* param)
{
    int batch = param->batch;
    int hidden = param->hidden;
    int out_number = param->out_number;

    /* malloc  output_buffer */
    int output_size = batch * out_number;
    float* output_buf = ( float* )malloc(sizeof(float) * output_size);

    float int32_to_fp32_scale = param->scale[0] * param->scale[1];

    int n, i, j;
    for(n = 0; n < batch; ++n)
    {
        for(i = 0; i < out_number; ++i)
        {
            int tmp = bias ? bias[i] : 0.0;
            for(j = 0; j < hidden; ++j)
            {
                // if(param->need_trans == 0)
                    tmp += input[n * hidden + j] * weight[i * hidden + j];
                // else
                //     tmp += input[n * hidden + j] * weight[i + j * out_number];
            }

            output_buf[n * out_number + i] = tmp * int32_to_fp32_scale;
        }
    }
    float out_scale = param->scale[2];
    for(int i = 0; i < output_size; i++)
    {
        int tmp  = round(output_buf[i]/out_scale);
        if(tmp > 127)
            tmp = 127;
        if(tmp < -127)
            tmp= -127;
        output[i] = (int8_t)tmp;
    }
    free(output_buf);

    return 0;
}
