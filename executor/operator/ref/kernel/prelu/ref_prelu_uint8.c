
static int ref_prelu_uint8(uint8_t* data, uint8_t* out_data, int dim0, int dim1, int dim2, int dim3, float* slope,
                           const prelu_param* param)
{
    int offset;

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
                    float real_input = (data[offset] - param->zero) * param->scale;
                    float real_output = MAX(real_input, 0) + slope[c] * MIN(real_input, 0.f);
                    out_data[offset] = round(real_output / param->scale) + param->zero;
                }
            }
        }
    }
    return 0;
}