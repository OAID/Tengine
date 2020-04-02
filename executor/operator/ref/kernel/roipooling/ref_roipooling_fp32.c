static int ref_roipooling_fp32(float* featmap, float* roi, float* output, roipooling_ref_param* param)
{
    int channel = param->channel;
    int feat_size = param->in_h * param->in_w;
    int pool_hw = param->out_h * param->out_w;

    for(int n = 0; n < param->num_rois; ++n)
    {
        float* roi_ptr = roi + n * 4;
        int roi_x0 = round(roi_ptr[0] * param->spatial_scale);
        int roi_y0 = round(roi_ptr[1] * param->spatial_scale);
        int roi_x1 = round(roi_ptr[2] * param->spatial_scale);
        int roi_y1 = round(roi_ptr[3] * param->spatial_scale);
        int roi_w = std::max(roi_x1 - roi_x0 + 1, 1);
        int roi_h = std::max(roi_y1 - roi_y0 + 1, 1);
        float bin_w = ( float )roi_w / ( float )param->out_w;
        float bin_h = ( float )roi_h / ( float )param->out_h;
        for(int c = 0; c < channel; ++c)
        {
            const float* feat_ptr = featmap + c * feat_size;
            for(int h = 0; h < param->out_h; ++h)
            {
                for(int w = 0; w < param->out_w; ++w)
                {
                    int h0 = roi_y0 + ( int )floor(( float )( h )*bin_h);
                    int h1 = roi_y0 + ( int )ceil(( float )(h + 1) * bin_h);
                    int w0 = roi_x0 + ( int )floor(( float )( w )*bin_w);
                    int w1 = roi_x0 + ( int )ceil(( float )(w + 1) * bin_w);
                    h0 = std::min(std::max(h0, 0), param->in_h);
                    h1 = std::min(std::max(h1, 0), param->in_h);
                    w0 = std::min(std::max(w0, 0), param->in_w);
                    w1 = std::min(std::max(w1, 0), param->in_w);
                    bool is_empty = (h1 <= h0) || (w1 <= w0);
                    float max_value = is_empty ? 0.f : feat_ptr[h0 * param->in_w + w0];
                    for(int y = h0; y < h1; y++)
                    {
                        for(int x = w0; x < w1; x++)
                        {
                            int idx = y * param->in_w + x;
                            max_value = std::max(max_value, feat_ptr[idx]);
                        }
                    }
                    output[h * param->out_w + w] = max_value;
                }
            }
            output += pool_hw;
        }
    }
    return 0;
}
