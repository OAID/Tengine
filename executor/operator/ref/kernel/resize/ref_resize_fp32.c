
static void bilinear_resize_fp32(float* inp, float* output, int h, int w, int c, float scale_x, float scale_y, int oh,
                                 int ow, int layout, int b)
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
                for(int k = 0; k < c; k++)
                {
                    int in_index = in_idx + k * in_hw;
                    output[k * out_hw + out_idx] = inp[in_index] * fx_0 * fy_0 + inp[in_index + w] * fx_0 * fy +
                                                inp[in_index + 1] * fx * fy_0 + inp[in_index + w + 1] * fx * fy;
                }
            }
        }
    
}

static int ref_resize_fp32(float* input, float* output, struct resize_param* param, int layout)
{
    int batch = param->batch;
    int channel = param->channel;
    int in_chw = channel * param->input_h * param->input_w;
    int out_chw = channel * param->output_h * param->output_w;
    if(layout == 0){
        for(int n = 0; n < batch; n++)
        {
            if(param->type == 0)
            {
                int si, sj;
                for(int k = 0; k < channel; k++)
                {
                    float* input_c = input + n * in_chw + k * param->input_h * param->input_w;
                    float* output_c = output + n * out_chw + k * param->output_h * param->output_w;
                    for(int i = 0; i < param->output_h; i++)
                    {
                        si = T_MIN(( int )(i * param->scale_y), param->input_h - 1);
                        for(int j = 0; j < param->output_w; j++)
                        {
                            sj = T_MIN(( int )(j * param->scale_x), param->input_w - 1);
                            output_c[i * param->output_w + j] = input_c[si * param->input_w + sj];
                        }
                    }
                }

                input += in_chw;
                output += out_chw;
            }
            else
            {
                bilinear_resize_fp32(input + n * in_chw, output + n * out_chw, param->input_h, param->input_w, channel,
                                    param->scale_x, param->scale_y, param->output_h, param->output_w, layout, n);
                input += in_chw;
                output += out_chw;
            }
        }
    } 
    else {
        if(param->type == 1)
        {
            int oh = param->output_h;
            int ow = param->output_w;
            int h = param->input_h;
            int w = param->input_w;
            int c = channel;
            float height_scale = (float)(param->input_h)/(oh);
            float width_scale = (float)(param->input_w)/(ow);
            for(int b = 0; b < batch; b++){   
                for(int y = 0; y < oh; y++){
                    float iy = y * height_scale;
                    int y0 = floor(iy);
                    int y1 = std::min(y0 + 1, h -1);
                    for(int x = 0; x < ow; x++){
                        float ix = x * width_scale;
                        int x0 = floor(ix);
                        int x1 = std::min(x0 + 1, w - 1);
                        for(int ic = 0; ic < c; ic++ ){
                            int outIndex = b*oh*ow*c + y*ow*c + x*c +ic;
                            int index0 = b*h*w*c + y0*w*c + x0*c + ic;
                            int index1 = b*h*w*c + y1*w*c + x0*c + ic;
                            int index2 = b*h*w*c + y0*w*c + x1*c + ic;
                            int index3 = b*h*w*c + y1*w*c + x1*c + ic;
                            output[outIndex] = input[index0]*(1-(iy-y0))*(1-(ix-x0))+input[index1]*(iy-y0)*(1-(ix-x0))+input[index2]*(1-(iy-y0))*(ix-x0)+input[index3]*(iy-y0)*(ix-x0);
                        }
                    }
                }

            }
        }
        else
        
        {
            int oh = param->output_h;
            int ow = param->output_w;
            int h = param->input_h;
            int w = param->input_w;
            int c = channel;
            float height_scale = (float)(param->input_h)/(oh);
            float width_scale = (float)(param->input_w)/(ow);
            for (int b = 0; b < batch; ++b) {
                for (int y = 0; y < oh; ++y) {
                int in_y = std::min(static_cast<int>(floor(y * height_scale)),
                                        h - 1);
                const float* y_input_ptr = input + in_y * w*c;
                for (int x = 0; x < ow; ++x) {
                    int in_x = std::min(static_cast<int>(floor(x * width_scale)),
                                        w - 1);
                    const float* x_input_ptr = y_input_ptr + in_x * c;
                    memcpy(output, x_input_ptr, c * sizeof(float));
                    output += channel;
                }
                }
                input += h*w*c;
            }
        }
}

    return 0;
}
