
static inline void calc_sum_fp16(const __fp16* input, __fp16* sum, int layout, int c, int h, int w, int cur_ch,
                                 int start_h, int start_w, int end_h, int end_w)
{
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
    float sum_f = 0.0f;
    for(int i = start_h; i < end_h; i++)
        for(int j = start_w; j < end_w; j++)
        {
            if(layout == 0)
                sum_f += fp16_to_fp32(input[cur_ch * h * w + i * w + j]);
            else
                sum_f += fp16_to_fp32(input[i * w * c + j * c + cur_ch]);
        }
    *sum = fp32_to_fp16(sum_f);
#else
    *sum = 0.0f;
    for(int i = start_h; i < end_h; i++)
        for(int j = start_w; j < end_w; j++)
        {
            if(layout == 0)
                *sum += input[cur_ch * h * w + i * w + j];
            else
                *sum += input[i * w * c + j * c + cur_ch];
        }
#endif
}

static inline void calc_max_fp16(const __fp16* input, __fp16* max, int layout, int c, int h, int w, int cur_ch,
                                 int start_h, int start_w, int end_h, int end_w)
{
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
    float max_f = 0.0f;
    float tmp = 0.0f;
    if(layout == 0)
        max_f = fp16_to_fp32(input[cur_ch * h * w + start_h * w + start_w]);
    else
        max_f = fp16_to_fp32(input[start_h * w * c + start_w * c + cur_ch]);
    for(int i = start_h; i < end_h; i++)
        for(int j = start_w; j < end_w; j++)
        {
            if(layout == 0)
                tmp = fp16_to_fp32(input[cur_ch * h * w + i * w + j]);
            else
                tmp = fp16_to_fp32(input[i * w * c + j * c + cur_ch]);
            // if(i ==start_h && j == start_w) printf("tmp :%f \n",tmp);

            max_f = max_f > tmp ? max_f : tmp;
        }
    *max = fp32_to_fp16(max_f);
#else
    *max = 0.0f;
    __fp16 tmp = 0.0f;
    if(layout == 0)
        *max = input[cur_ch * h * w + start_h * w + start_w];
    else
        *max = input[start_h * w * c + start_w * c + cur_ch];

    for(int i = start_h; i < end_h; i++)
        for(int j = start_w; j < end_w; j++)
        {
            if(layout == 0)
                tmp = input[cur_ch * h * w + i * w + j];
            else
                tmp = input[i * w * c + j * c + cur_ch];

            *max = *max > tmp ? *max : tmp;
        }

#endif
}

static int ref_pooling_fp16(const __fp16* input, __fp16* output, struct op_data* param)
{
    int input_chw = param->channel * param->input[0] * param->input[1];
    int output_chw = param->channel * param->output[0] * param->output[1];

    for(int n = 0; n < param->batch; n++)
    {
        const __fp16* input_cur = input + n * input_chw;
        for(int c = 0; c < param->channel; c++)
        {
            for(int ph = 0; ph < param->output[0]; ph++)
            {
                for(int pw = 0; pw < param->output[1]; pw++)
                {
                    int pool_size = 1;
                    int offset = 0;
                    int h_start = ph * param->strides[0] - param->pads[0];
                    int h_end = h_start + param->kernels[0];
                    if(h_end > param->input[0] + param->pads[0])
                        h_end = param->input[0] + param->pads[0];
                    int w_start = pw * param->strides[1] - param->pads[1];
                    int w_end = w_start + param->kernels[1];
                    if(w_end > param->input[1] + param->pads[1])
                        w_end = param->input[1] + param->pads[1];

                    if(param->caffe_flavor)
                        pool_size = (h_end - h_start) * (w_end - w_start);

                    h_start = h_start > 0 ? h_start : 0;
                    w_start = w_start > 0 ? w_start : 0;
                    h_end = h_end < param->input[0] ? h_end : param->input[0];
                    w_end = w_end < param->input[1] ? w_end : param->input[1];
                    // printf("w: %d,%d ,h: %d,%d\n",w_start,w_end,h_start,h_end);

                    if(!param->caffe_flavor)
                        pool_size = (h_end - h_start) * (w_end - w_start);
                    if(param->layout == 0)    // nchw
                        offset = n * output_chw + c * param->output[0] * param->output[1] + ph * param->output[1] + pw;
                    else
                        offset = n * output_chw + ph * param->output[1] * param->channel + pw * param->channel + c;

                    if(param->method == 0)
                    {
                        __fp16 max;
                        calc_max_fp16(input_cur, &max, param->layout, param->channel, param->input[0], param->input[1],
                                      c, h_start, w_start, h_end, w_end);
                        output[offset] = max;
                    }
                    else if(param->method == 1)
                    {
                        __fp16 sum;
                        calc_sum_fp16(input_cur, &sum, param->layout, param->channel, param->input[0], param->input[1],
                                      c, h_start, w_start, h_end, w_end);
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
                        output[offset] = fp32_to_fp16(fp16_to_fp32(sum) / pool_size);
#else
                        output[offset] = sum / pool_size;
#endif
                    }
                    else
                        return -1;
                }
            }
        }
    }
    return 0;
}
