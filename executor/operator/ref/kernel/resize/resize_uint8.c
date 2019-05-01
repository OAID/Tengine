#include <math.h>

static int prelu_fp32(int batch_number,int in_chw,int out_chw,float* input, float* output, int h, int w, int c, float scale_x, float scale_y, int oh, int ow)
{   

    for(int i = 0; i < batch_number; i++)
    {
        bilinear_resize(input, output, h, w, c, scale_x, scale_y, oh, ow);
        input += in_chw;
        output += out_chw;
    }

    return 0;
}
void bilinear_resize(float* inp, float* output, int h, int w, int c, float scale_x, float scale_y, int oh, int ow)
{
    int out_hw = oh * ow;
    int in_hw = h * w;
    for(int j = 0; j < oh; j++)
    {
        float fy = (j + 0.5) * scale_y - 0.5;
        int sy = floor(fy);
        fy -= sy;
        sy = min(sy, h - 2);
        sy = max(0, sy);
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
                output[k * out_hw + out_idx] = inp[in_index] * fx_0 * fy_0 + inp[in_index + w] * fx_0 * fy +
                                                inp[in_index + 1] * fx * fy_0 + inp[in_index + w + 1] * fx * fy;
            }
        }
    }
}