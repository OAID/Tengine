#include <stdio.h>
int ref_instancenorm_fp32(float* input_data, float* output_data, float* gamma_data, float* beta_data, int size, int channels, int n, float eps, float scale, float zero_point, int layout)
{   
    int image_size = channels * size;
    float sum = 0.f;
    float sqsum = 0.f;
    int offset = 0;
    for(int s = 0; s < n; s++)
    {
        for(int i = 0; i < channels; i++)
        {
            for(int j = 0; j < size; j++)
            {
                if(TENGINE_LAYOUT_NCHW == layout)
                    offset = s * image_size + i * size + j;
                else
                    offset = s * image_size + j * channels + i;
                sum += input_data[offset];
            }
            float mean = sum / size;
            float tmp = 0.f;
            for(int j = 0; j < size; j++)
            {
                if(TENGINE_LAYOUT_NCHW == layout)
                    offset = s * image_size + i * size + j;
                else
                    offset = s * image_size + j * channels + i;
                tmp = input_data[offset] - mean;
                sqsum += tmp * tmp; 
            }
            float var = sqsum / size;

            float a = gamma_data[i] / (sqrt(var+eps));
            float b = -mean * a + beta_data[i];
            for(int j = 0; j < size; j++)
            {
                if(TENGINE_LAYOUT_NCHW == layout)
                    offset = s * image_size + i * size + j;
                else
                    offset = s * image_size + j * channels + i;
                output_data[offset] = input_data[offset] * a + b;
            }
        }
    }
    return 0;
}