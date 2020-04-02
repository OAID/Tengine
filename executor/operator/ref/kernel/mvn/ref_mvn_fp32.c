#include <stdio.h>
int ref_mvn_fp32(float* in_data, float* out_data, const ref_mvn_param* param)
{
    int batch_num = param->input_n;
    int in_h = param->input_h;
    int in_w = param->input_w;
    int in_c = param->input_c;
    int in_size = in_h * in_w;
    int image_size = in_size * in_c;
    int offset = 0;
    int layout = param->layout;
    int across_channels = param->across_channels;
    int normalize_variance = param->normalize_variance;
    float eps = param->eps;

    float* sum = (float*)malloc(in_c*sizeof(float));

    if(NULL == sum)
        return -100;

    for(int n = 0; n < batch_num; n++)
    {
        for(int c = 0; c < in_c; c++)
        {
            float s = 0.f;
            for(int i =0; i < in_size; i++)
            {
                if(TENGINE_LAYOUT_NCHW == layout)
                    offset = n * image_size + c * in_size + i;
                else
                    offset = n * image_size + i * in_c + c;
                s += in_data[offset];
            }
            sum[c] = s;
        }

        if(across_channels)
        {
            float mean = 0.f;
            for(int c = 0; c < in_c; c++)
            {
                mean += sum[c];
            }
            mean = mean/(in_size * in_c);

            for(int c = 0; c < in_c; c++)
            {
                for(int i = 0; i < in_size; i++)
                {
                    if(TENGINE_LAYOUT_NCHW == layout)
                        offset = n * image_size + c * in_size + i;
                    else
                        offset = n * image_size + i * in_c + c;
                    out_data[offset] = in_data[offset] - mean;
                }
            }
        }
        else
        {
            for(int c = 0; c < in_c; c++)
            {
                float mean = sum[c] / in_size;

                for(int i = 0; i < in_size; i++)
                {
                    if(TENGINE_LAYOUT_NCHW == layout)
                        offset = n * image_size + c * in_size + i;
                    else
                        offset = n * image_size + i * in_c + c;
                    out_data[offset] = in_data[offset] - mean;
                }
            }
        }
        
        if(normalize_variance)
        {
            float* sqsum = (float*)malloc(in_c*sizeof(float));
            if(NULL == sqsum)
                return -100;

            for(int c = 0; c < in_c; c++)
            {
                float s = 0.f;
                for(int i =0; i < in_size; i++)
                {
                    if(TENGINE_LAYOUT_NCHW == layout)
                        offset = n * image_size + c * in_size + i;
                    else
                        offset = n * image_size + i * in_c + c;
                    s += in_data[offset] * in_data[offset];
                }   
                sqsum[c] = s;
            }

            if(across_channels)
            {
                float sqmean = 0.f;
                for(int c = 0; c < in_c; c++)
                {
                    sqmean += sqsum[c];
                }
                sqmean = sqmean / (in_c * in_size);

                float norm_var = sqrt(sqmean) + eps;

                for(int c = 0; c < in_c; c++)
                {
                    for(int i = 0; i < in_size; i++)
                    {
                        if(TENGINE_LAYOUT_NCHW == layout)
                            offset = n * image_size + c * in_size + i;
                        else
                            offset = n * image_size + i * in_c + c;
                        out_data[offset] = out_data[offset] / norm_var;
                    }
                }
            }
            else
            {
                for(int c = 0; c < in_c; c++)
                {
                    float sqmean = sqsum[c] / in_size;
                    float norm_var = sqrt(sqmean) + eps;
                    for(int i = 0; i < in_size; i++)
                    {
                        if(TENGINE_LAYOUT_NCHW == layout)
                            offset = n * image_size + c * in_size + i;
                        else
                            offset = n * image_size + i * in_c + c;
                        out_data[offset] = out_data[offset] / norm_var;
                    }
                }
            }
            free(sqsum);
            sqsum = NULL;
        }
    }

    free(sum);
    sum = NULL;
    return 0;
}