
static void bilinear_resize_fp16(__fp16* inp, __fp16* output, int h, int w, int c, float scale_x, float scale_y, int oh,
                                 int ow)
{
    int out_hw = oh * ow;
    int in_hw = h * w;
    for(int j = 0; j < oh; j++)
    {
        float fy = (j + 0.5) * scale_y - 0.5;
        int sy = floor(fy);
        fy -= sy;
        sy = T_MIN(sy, h - 2);
        sy = T_MAX(0, sy);
        float fy_0 = 1.f - fy;

        for(int i = 0; i < ow; i++)
        {
            float fx = (i + 0.5) * scale_x - 0.5;
            int sx = floor(fx);
            fx -= sx;
            if(sx < 0)
            {
                sx = 0;
                fx = 0;
            }
            if(sx >= w - 1)
            {
                fx = 0;
                sx = w - 2;
            }
            float fx_0 = 1.f - fx;
            int out_idx = j * ow + i;
            int in_idx = sy * w + sx;
            // printf("i=%d j=%d\t sx=%d fx=%f\t sy=%d fy=%f\n",i,j,sx,fx,sy,fy);
            for(int k = 0; k < c; k++)
            {
                int in_index = in_idx + k * in_hw;
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
                output[k * out_hw + out_idx] = fp32_to_fp16(
                    fp16_to_fp32(inp[in_index]) * fx_0 * fy_0 + fp16_to_fp32(inp[in_index + w]) * fx_0 * fy +
                    fp16_to_fp32(inp[in_index + 1]) * fx * fy_0 + fp16_to_fp32(inp[in_index + w + 1]) * fx * fy);
#else
                output[k * out_hw + out_idx] = inp[in_index] * fx_0 * fy_0 + inp[in_index + w] * fx_0 * fy +
                                               inp[in_index + 1] * fx * fy_0 + inp[in_index + w + 1] * fx * fy;
#endif
            }
        }
    }
}

static int ref_resize_fp16(__fp16* input, __fp16* output, struct resize_param* param, int type)
{
    int batch = param->batch;
    int channel = param->channel;
    int in_chw = channel * param->input_h * param->input_w;
    int out_chw = channel * param->output_h * param->output_w;

    for(int n = 0; n < batch; n++)
    {
        if(param->type == 0)
        {
            int si, sj;
            for(int k = 0; k < channel; k++)
            {
                __fp16* input_c = input + n * in_chw + k * param->input_h * param->input_w;
                __fp16* output_c = output + k * param->output_h * param->output_w;
                for(int i = 0; i < param->output_h; i++)
                {
                    si = T_MIN(( int )(i * param->scale_y), param->input_h - 1);
                    for(int j = 0; j < param->output_w; j++)
                    {
                        sj = T_MIN(( int )(j * param->scale_x), param->output_w - 1);
                        output_c[i * param->output_w + j] = input_c[si * param->input_w + sj];
                    }
                }
            }
            input += in_chw;
            output += out_chw;
        }
        else
        {
            bilinear_resize_fp16(input + n * in_chw, output + n * out_chw, param->input_h, param->input_w, channel,
                                 param->scale_x, param->scale_y, param->output_h, param->output_w);
            input += in_chw;
            output += out_chw;
        }
    }

    return 0;
}
