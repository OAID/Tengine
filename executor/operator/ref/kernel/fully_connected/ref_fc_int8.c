
static int ref_fc_int8(const int8_t * input, int8_t * output, const int8_t* weight, const float* bias, fc_data* param)
{
    int batch = param->batch;
    int hidden = param->hidden;
    int out_number = param->out_number;

    /* dequant input  */
    int input_size = batch * hidden;
    float* input_buf = (float*)malloc(sizeof(float) * input_size);
    for(int i=0; i<input_size;i++)
    {
        input_buf[i] = input[i] * param->scale[0];
    }

    /* dequant kernel  */
    int kernel_size = hidden * out_number;
    float* weight_buf = (float*)malloc(sizeof(float) * kernel_size);
    for(int i=0; i<kernel_size;i++)
    {
        weight_buf[i] = weight[i] * param->scale[1];
    }

    /* malloc  output_buffer */
    int output_size = batch * out_number;
    float* output_buf = (float*)malloc(sizeof(float) * output_size);

    int n,i,j;
    for ( n = 0; n < batch; ++n)
    {
        for( i = 0; i < out_number; ++i)
        {
            float tmp = bias ? bias[i] :0.0;
            for ( j = 0; j < hidden; ++j)
            {
                if(param->need_trans == 0)
                    tmp += input_buf[n* hidden + j] * weight_buf[i*hidden + j];
                else
                    tmp += input_buf[n* hidden + j] * weight_buf[i + j*out_number];
            }
            output_buf[n*out_number + i ] = tmp;
        }
    }

    /* quant output */
    float output_max = 0.0f;
    for(int i =0; i< output_size; i++)
    {
        if(output_max < fabs(output_buf[i]))
            output_max = fabs(output_buf[i]);
    }
    param->scale[2] = output_max/127;
    for(int i =0; i< output_size; i++)
    {
        output[i] = round(output_buf[i]*127/output_max);
    }
    free(output_buf);
    free(weight_buf);
    free(input_buf);
    return 0;
}

