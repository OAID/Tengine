

static int ref_fm_fp32(const float* input, float* output, const float* weight, const float* bias, feature_match_data* param)
{
    int batch = param->batch;
    int hidden = param->hidden;
    int out_number = param->out_number;

    int n, i, j;
    for(n = 0; n < batch; ++n)
    {
        for(i = 0; i < out_number; ++i)
        {
            float tmp = bias ? bias[i] : 0.0;
            for(j = 0; j < hidden; ++j)
            {
                // if(param->need_trans == 0)
                tmp += input[n * hidden + j] * weight[i * hidden + j];
                // else
                //     tmp += input[n * hidden + j] * weight[i + j * out_number];
            }
            output[n * out_number + i] = tmp;
        }
    }
    return 0;
}
