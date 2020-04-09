
static inline float calc_sum(const float* input, int layout, int c, int h, int w, int cur_ch, int start_h, int start_w,
                             int end_h, int end_w)
{
    float sum = 0.0f;
    for(int i = start_h; i < end_h; i++)
        for(int j = start_w; j < end_w; j++)
        {
            if(layout == 0)
                sum += input[cur_ch * h * w + i * w + j];
            else
                sum += input[i * w * c + j * c + cur_ch];
        }

    return sum;
}

static inline float calc_max(const float* input, int layout, int c, int h, int w, int cur_ch, int start_h, int start_w,
                             int end_h, int end_w)
{
    float max = 0.0f;
    if(layout == 0)
        max = input[cur_ch * h * w + start_h * w + start_w];
    else
        max = input[start_h * w * c + start_w * c + cur_ch];

    float tmp = 0.0f;
    for(int i = start_h; i < end_h; i++)
        for(int j = start_w; j < end_w; j++)
        {
            if(layout == 0)
                tmp = input[cur_ch * h * w + i * w + j];
            else
                tmp = input[i * w * c + j * c + cur_ch];

            max = max > tmp ? max : tmp;
        }

    return max;
}

static int ref_pooling_fp32(const float* input, float* output, struct op_data* param)
{
    int input_chw = param->channel * param->input[0] * param->input[1];
    int output_chw = param->channel * param->output[0] * param->output[1];

    for(int n = 0; n < param->batch; n++)
    {
        const float* input_cur = input + n * input_chw;
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
                        float max = calc_max(input_cur, param->layout, param->channel, param->input[0], param->input[1],
                                             c, h_start, w_start, h_end, w_end);
                        output[offset] = max;
                    }
                    else if(param->method == 1)
                    {
                        float sum = calc_sum(input_cur, param->layout, param->channel, param->input[0], param->input[1],
                                             c, h_start, w_start, h_end, w_end);
                        output[offset] = sum / pool_size;
                    }
                    else
                        return -1;
                }
            }
        }
    }
    return 0;
}
