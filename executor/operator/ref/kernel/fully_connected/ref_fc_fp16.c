
<<<<<<< HEAD
static int ref_fc_fp16(const __fp16 * input, __fp16 * output, const __fp16* weight, const __fp16* bias, fc_data* param)
=======
static int ref_fc_fp16(const __fp16* input, __fp16* output, const __fp16* weight, const __fp16* bias, fc_data* param)
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
{
    int batch = param->batch;
    int hidden = param->hidden;
    int out_number = param->out_number;

<<<<<<< HEAD
    int n,i,j;
    for ( n = 0; n < batch; ++n)
    {
        for( i = 0; i < out_number; ++i)
        {
#if!defined( __ARM_ARCH) || __ARM_ARCH <8
            float tmp = bias ? fp16_to_fp32(bias[i]) : 0.0;
            for ( j = 0; j < hidden; ++j)
            {
                if(param->need_trans == 0)
                    tmp += fp16_to_fp32(input[n*hidden + j]) * fp16_to_fp32(weight[i*hidden + j]);
                else
                    tmp += fp16_to_fp32(input[n*hidden + j]) * fp16_to_fp32(weight[i + j*out_number]);
            }

            output[n*out_number + i ] = fp32_to_fp16(tmp);
#else
            __fp16 tmp = bias ? bias[i] : 0.0;
            for ( j = 0; j < hidden; ++j)
            {
                if(param->need_trans == 0)
                    tmp += input[n*hidden + j] * weight[i*hidden + j];
                else
                    tmp += input[n*hidden + j] * weight[i + j*out_number];
            }

            output[n*out_number + i ] = tmp;
=======
    int n, i, j;
    for(n = 0; n < batch; ++n)
    {
        for(i = 0; i < out_number; ++i)
        {
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
            float tmp = bias ? fp16_to_fp32(bias[i]) : 0.0;
            for(j = 0; j < hidden; ++j)
            {
                if(param->need_trans == 0)
                    tmp += fp16_to_fp32(input[n * hidden + j]) * fp16_to_fp32(weight[i * hidden + j]);
                else
                    tmp += fp16_to_fp32(input[n * hidden + j]) * fp16_to_fp32(weight[i + j * out_number]);
            }

            output[n * out_number + i] = fp32_to_fp16(tmp);
#else
            __fp16 tmp = bias ? bias[i] : 0.0;
            for(j = 0; j < hidden; ++j)
            {
                if(param->need_trans == 0)
                    tmp += input[n * hidden + j] * weight[i * hidden + j];
                else
                    tmp += input[n * hidden + j] * weight[i + j * out_number];
            }

            output[n * out_number + i] = tmp;
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
#endif
        }
    }
    return 0;
}
<<<<<<< HEAD

=======
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
