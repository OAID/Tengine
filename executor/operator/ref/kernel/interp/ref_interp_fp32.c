int ref_interp_fp32(float* input, float* output,interp_param* param)
{
    for (int n = 0; n < param->batch_number; ++n) {
    for (int c = 0; c < param->inc; ++c) {
        for (int y = 0; y < param->output_height; ++y) {
            float in_y = INTERP_MIN(y / param->height_scale, (float)(param->inh - 1));
            const int in_y1 = INTERP_MIN((int)(in_y), param->inh - 1);
            const int in_y2 = INTERP_MIN(in_y1 + 1, param->inh - 1);
            float dy1 = fabs(in_y - in_y1);
            float dy2 = fabs(in_y - in_y2);
            if (in_y1 == in_y2) {
            dy1 = 0.5f;
            dy2 = 0.5f;
            }

            const int input_width_mul_y1 = param->inw * in_y1;
            const int input_width_mul_y2 = param->inw * in_y2;

            for (int x = 0; x < param->output_width; ++x) {
                float in_x = INTERP_MIN(x / param->width_scale, (float)(param->inw - 1));
                const int in_x1 = INTERP_MIN((int)(in_x), param->inw - 1);
                const int in_x2 = INTERP_MIN(in_x1 + 1, param->inw - 1);

                float dx1 = abs(in_x - in_x1);
                float dx2 = abs(in_x - in_x2);
                if (in_x1 == in_x2) {
                    dx1 = 0.5f;
                    dx2 = 0.5f;
                }

                float X11 = input[input_width_mul_y1 + in_x1];
                float X21 = input[input_width_mul_y1 + in_x2];
                float X12 = input[input_width_mul_y2 + in_x1];
                float X22 = input[input_width_mul_y2 + in_x2];
                output[param->output_width * y + x] = dx2 * dy2 * X11 +dx1 * dy2 * X21 +dx2 * dy1 * X12 +dx1 * dy1 * X22;
            }
        }
        input += param->inh * param->inw;
        output += param->output_width * param->output_height;
        }
    }
    return 0;
}
