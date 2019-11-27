
static int ref_prelu_fp32(float* data, float* out_data, int dim0, int dim1, int dim2, int dim3, float* slope,
                          const prelu_param* param)
{
    int offset = 0;
    // nchw
    // nhwc
    for(int i = 0; i < dim0; i++)
    {
        for(int c = 0; c < dim1; c++)
        {
            for(int l = 0; l < dim2; l++)
            {
                for(int k = 0; k < dim3; k++)
                {
                    if(param->layout == 0)
                    {
                        // nchw
                        offset = i * dim1 * dim2 * dim3 + c * dim2 * dim3 + l * dim3 + k;
                    }
                    else
                    {
                        // nhwc
                        offset = i * dim1 * dim2 * dim3 + l * dim3 * dim1 + k * dim1 + c;
                    }
                    out_data[offset] = MAX(data[offset], 0) + slope[c] * MIN(data[offset], 0.f);
                }
            }
        }
    }
    return 0;
}