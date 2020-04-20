static int ref_matmul_fp32(const float* input0, float* input1,float* output, matmul_data* param)
{
    int batch = param->batch;
    int c = param->c;
    int h = param->h;
    int w = param->w;
    int k = param->k;

    for(int n = 0; n < batch; ++n)
    {
        for(int in_c = 0; in_c < c; in_c++)
        {
            const float *data0 = input0 + n * c * h * w + in_c * h * w;
            float *data1 = input1 + n * c * w * k + in_c * w * k;
            for(int in_h = 0; in_h < h; in_h++)
            {
                for(int in_k = 0; in_k < k; in_k++)
                {
                    float tmp = 0;
                    for(int in_w = 0; in_w < w;in_w++)
                    {
                        int index0 = in_h * w + in_w;
                        int index1 = in_w * k + in_k;
                        tmp += data0[index0] * data1[index1];
                    }
                    *output = tmp;
                    output++;
                }
            }
        }
    }
    
    return 0;
}
