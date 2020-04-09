static int ref_pad_fp32(float* data, float* out_data, pad_param* param)
{
    if(param->mode == 0)
    {
        // support pad on h,w dim only
        if(param->pad_0_h == 0 && param->pad_0_w == 0 && param->pad_3_h == 0 && param->pad_3_w == 0)
        {
            for(int n = 0; n < param->in_n; ++n)
            {
                for(int ph = 0; ph < param->out_h; ++ph)
                {
                    for(int pw = 0; pw < param->out_w; ++pw)
                    {
                        int h = ph - param->pad_1_h;
                        int w = pw - param->pad_2_h;
                        const int pad_index = (ph * param->out_w + pw) * param->in_c;
                        if(h < 0 || w < 0 || h >= param->in_h || w >= param->in_w)
                        {
                            for(int c = 0; c < param->in_c; ++c)
                            {
                                out_data[pad_index + c] = param->cv_f32;
                            }
                        }
                        else
                        {
                            const int input_index = (h * param->in_w + w) * param->in_c;
                            for(int c = 0; c < param->in_c; ++c)
                            {
                                out_data[pad_index + c] = data[input_index + c];
                            }
                        }
                    }
                }
                // Do offset.
                data += param->in_size / param->in_n;
                out_data += param->out_size / param->out_n;
            }
        }
        else
        {
            return -1;
        }
    }
    else if(param->mode == 1)
    {
        if(param->pad_0_h == 0 && param->pad_0_w == 0 && param->pad_3_h == 0 && param->pad_3_w == 0)
        {
            for(int n = 0; n < param->in_n; ++n)
            {
                for(int ph = 0; ph < param->out_h; ++ph)
                {
                    for(int pw = 0; pw < param->out_w; ++pw)
                    {
                        const int pad_index = (ph * param->out_w + pw) * param->in_c;
                        int h = ph - param->pad_1_h;
                        int w = pw - param->pad_2_h;
                        h = MAX(h, -h);
                        h = MIN(h, 2 * param->in_h - h - 2);
                        w = MAX(w, -w);
                        w = MIN(w, 2 * param->in_w - w - 2);
                        const int input_index = (h * param->in_w + w) * param->in_c;
                        for(int c = 0; c < param->in_c; ++c)
                        {
                            out_data[pad_index + c] = data[input_index + c];
                        }
                    }
                }
                // Do offset.
                data += param->in_size / param->in_n;
                out_data += param->out_size / param->out_n;
            }
        }
        else
        {
            return -1;
        }
    }
    else if(param->mode == 2)
    {
        if(param->pad_0_h == 0 && param->pad_0_w == 0 && param->pad_3_h == 0 && param->pad_3_w == 0)
        {
            for(int n = 0; n < param->in_n; ++n)
            {
                for(int ph = 0; ph < param->out_h; ++ph)
                {
                    for(int pw = 0; pw < param->out_w; ++pw)
                    {
                        const int pad_index = (ph * param->out_w + pw) * param->in_c;
                        // int h = ph - param->pad_1_h;
                        int w = pw - param->pad_2_h;
                        int h = ph - param->pad_1_h;
                        h = MAX(h, -h - 1);
                        h = MIN(h, 2 * param->in_h - h - 1);
                        w = MAX(w, -w - 1);
                        w = MIN(w, 2 * param->in_w - w - 1);

                        const int input_index = (h * param->in_w + w) * param->in_c;
                        for(int c = 0; c < param->in_c; ++c)
                        {
                            out_data[pad_index + c] = data[input_index + c];
                        }
                    }
                }
                // Do offset.
                data += param->in_size / param->in_n;
                out_data += param->out_size / param->out_n;
            }
        }
        else
        {
            return -1;
        }
    }
    else
    {
        return -1;
    }

    return 0;
}