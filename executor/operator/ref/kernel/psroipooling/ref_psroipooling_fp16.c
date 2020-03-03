static int ref_psroipooling_fp16(__fp16* featmap, __fp16* roi, __fp16* output, psroipooling_ref_param* param)
{
    int pool_hw = param->out_h * param->out_w;
    int output_dim = param->output_dim;

    for(int n = 0; n < param->num_rois; ++n)
    {
        __fp16* roi_ptr = roi + n * 4;
        int roi_x0 = round(fp16_to_fp32(roi_ptr[0]) * param->spatial_scale);
        int roi_y0 = round(fp16_to_fp32(roi_ptr[1]) * param->spatial_scale);
        int roi_x1 = round(fp16_to_fp32(roi_ptr[2]) * param->spatial_scale);
        int roi_y1 = round(fp16_to_fp32(roi_ptr[3]) * param->spatial_scale);
        int roi_w = std::max(roi_x1 - roi_x0 + 1, 1);
        int roi_h = std::max(roi_y1 - roi_y0 + 1, 1);
        float bin_w = ( float )roi_w / ( float )param->out_w;
        float bin_h = ( float )roi_h / ( float )param->out_h;
        for(int c = 0; c < output_dim; ++c)
        {
            __fp16* outptr = output + c * pool_hw;
            for(int h = 0; h < param->out_h; ++h)
            {
                for(int w = 0; w < param->out_w; ++w)
                {
                    __fp16* inptr = featmap + (c*param->out_h + h)*param->out_w + w;

                    int hstart = floor(roi_y1 + (float)(h) * bin_h);
                    int wstart = floor(roi_x1 + (float)(w) * bin_w);
                    int hend = ceil(roi_y1 + (float)(h + 1) * bin_h);
                    int wend = ceil(roi_x1 + (float)(w + 1) * bin_w);

                    hstart = std::min(std::max(hstart, 0), h);
                    wstart = std::min(std::max(wstart, 0), w);
                    hend = std::min(std::max(hend, 0), h);
                    wend = std::min(std::max(wend, 0), w);

                    bool is_empty = (hend <= hstart) || (wend <= wstart);
                    int area = (hend - hstart) * (wend - wstart);

                    float sum = 0.f;
                    for (int y = hstart; y < hend; y++)
                    {
                        for (int x = wstart; x < wend; x++)
                        {
                            int index = y * w + x;
                            sum += fp16_to_fp32(inptr[index]);
                        }
                    }
                    outptr[w] = fp32_to_fp16(is_empty ? 0.f : (sum / (float)area));
                }
            }
            
            outptr += param->out_w;
        }
    }
    return 0;
}
