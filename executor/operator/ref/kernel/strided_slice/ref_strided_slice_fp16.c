static int ref_strided_slice_fp16(__fp16* in_data, __fp16* out_data, strided_slice_param* param)
{
    int out_chw = param->out_c * param->out_h * param->out_w;
    int out_cw = param->out_c * param->out_w;
    int in_chw = param->in_c * param->in_h * param->in_w;
    int in_cw = param->in_c * param->in_w;
    for(int n = 0; n < param->batch_num; n++)
    {
        for(int h = 0; h < param->out_h; h++)
        {
            for(int w = 0; w < param->out_w; w++)
            {
                for(int c = 0; c < param->out_c; c++)
                {
                    out_data[n * out_chw + h * out_cw + w * param->out_c + c] =
                        in_data[(param->begin[0] + n * param->stride[0]) * in_chw +
                                (param->begin[1] + h * param->stride[1]) * in_cw +
                                (param->begin[2] + w * param->stride[2]) * param->in_c +
                                (param->begin[3] + c * param->stride[3])];
                }
            }
        }
    }
    return 0;
}
