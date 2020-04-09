
static int ref_fm_uint8(const uint8_t* input, uint8_t* output, const uint8_t* weight, const int* bias, feature_match_data* param)
{
    int batch = param->batch;
    int hidden = param->hidden;
    int out_number = param->out_number;

    /* dequant input  */
    int input_size = batch * hidden;
    float* input_buf = ( float* )malloc(sizeof(float) * input_size);
    for(int i = 0; i < input_size; i++)
    {
        input_buf[i] = (input[i] - param->zero[0]) * param->scale[0];
    }

    /* dequant kernel  */
    int kernel_size = hidden * out_number;
    float* weight_buf = ( float* )malloc(sizeof(float) * kernel_size);
    for(int i = 0; i < kernel_size; i++)
    {
        weight_buf[i] = (weight[i] - param->zero[1]) * param->scale[1];
    }

    int n, i, j;
    for(n = 0; n < batch; ++n)
    {
        for(i = 0; i < out_number; ++i)
        {
            float tmp = bias ? bias[i] * param->scale[0] * param->scale[1] : 0.0;
            for(j = 0; j < hidden; ++j)
            {
                // if(param->need_trans == 0)
                    tmp += input_buf[n * hidden + j] * weight_buf[i * hidden + j];
                // else
                //     tmp += input_buf[n * hidden + j] * weight_buf[i + j * out_number];
            }
            int quant_tmp = round(tmp / param->scale[2]) + param->zero[2];
            if(quant_tmp > 255)
                quant_tmp = 255;
            if(quant_tmp < 0)
                quant_tmp = 0;
            output[n * out_number + i] = quant_tmp;
        }
    }

    free(weight_buf);
    free(input_buf);
    return 0;
}
