static int ref_upsample_int8(int8_t* input, int8_t* output, upsample_param* param)
{
     for(int n = 0; n < param->batch; ++n)
    {
        for(int c = 0; c < param->channel; c++)
        {
            for(int h = 0; h < param->out_h; h++)
            {
                for(int w = 0; w < param->out_w; w++)
                {
                    int in_w = w / param->scale;
                    int in_h = h / param->scale;
                    int out_idx = n * param->channel * param->out_h * param->out_w + c * param->out_h * param->out_w + h * param->out_w + w;
                    int in_idx = n * param->channel * param->input_h * param->input_w + c * param->input_w * param->input_h + in_h * param->input_w + in_w;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
    return 0;
}