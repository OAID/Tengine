static int ref_prelu_fp16(__fp16* data, __fp16* out_data, int dim0, int dim1, int dim2, int dim3, float* slope,
                          prelu_param* param)
{
    int offset = 0;
    //__fp16* data = ( __fp16* )data;
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
/* for arm32 && x86 */
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
                    float output_real =
                        MAX(fp16_to_fp32(data[offset]), 0) + slope[c] * MIN(fp16_to_fp32(data[offset]), 0.f);
                    out_data[offset] = fp32_to_fp16(output_real);

#else
                    out_data[offset] = MAX(data[offset], 0) + slope[c] * MIN(data[offset], 0.f);
#endif
                }
            }
        }
    }
    return 0;
}
