static int ref_psroipooling_fp32(float* featmap, float* roi, float* output, psroipooling_ref_param* param)
{
    int pool_hw = param->out_h * param->out_w;
    int output_dim = param->output_dim;

    for(int n = 0; n < param->num_rois; ++n)
    {
        float* roi_ptr = roi + n * 4;
        float roi_x0 = round(roi_ptr[0]) * param->spatial_scale;
        float roi_y0 = round(roi_ptr[1]) * param->spatial_scale;
        float roi_x1 = round(roi_ptr[2] + 1.f) * param->spatial_scale;
        float roi_y1 = round(roi_ptr[3] + 1.f) * param->spatial_scale;

        int roi_w = T_MAX(roi_x1 - roi_x0 , 0);
        int roi_h = T_MAX(roi_y1 - roi_y0 , 0);
        
        float bin_w = ( float )roi_w / ( float )param->out_w;
        float bin_h = ( float )roi_h / ( float )param->out_h;

        for(int c = 0; c < output_dim; ++c)
        {
            float* outptr = output + c * pool_hw;
            for(int h = 0; h < param->out_h; ++h)
            {
                for(int w = 0; w < param->out_w; ++w)
                {
                    float* inptr = featmap + (c*param->out_h + h)*param->out_w + w;

                    int hstart = floor(roi_y0 + (float)(h) * bin_h);
                    int wstart = floor(roi_x0 + (float)(w) * bin_w);
                    int hend = ceil(roi_y0 + (float)(h + 1) * bin_h);
                    int wend = ceil(roi_x0 + (float)(w + 1) * bin_w);

                    hstart = T_MIN(T_MAX(hstart, 0), param->in_h);
                    wstart = T_MIN(T_MAX(wstart, 0), param->in_w);
                    hend = T_MIN(T_MAX(hend, 0), param->in_h);
                    wend = T_MIN(T_MAX(wend, 0), param->in_w);

                    bool is_empty = (hend <= hstart) || (wend <= wstart);
                    int area = (hend - hstart) * (wend - wstart);

                    float sum = 0.f;
                    for (int y = hstart; y < hend; y++)
                    {
                        for (int x = wstart; x < wend; x++)
                        {
                            int index = y * param->in_w + x;
                            sum += inptr[index];
                        }
                    }
                    outptr[w] = is_empty ? 0.f : (sum / (float)area);
                }
                outptr += param->out_w;
            }


        }
    }
    return 0;
}
