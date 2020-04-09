/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2019, Open AI Lab
 * Author: bingzhang@openailab.com
 */
static inline float bilinear_interpolate(const float* ptr, int w, int h, float x, float y)
{
    int x0 = x;
    int x1 = x0 + 1;
    int y0 = y;
    int y1 = y0 + 1;

    float a0 = x1 - x;
    float a1 = x - x0;
    float b0 = y1 - y;
    float b1 = y - y0;

    if (x1 >= w)
    {
        x1 = w-1;
        a0 = 1.f;
        a1 = 0.f;
    }
    if (y1 >= h)
    {
        y1 = h-1;
        b0 = 1.f;
        b1 = 0.f;
    }

    float r0 = ptr[ y0 * w + x0 ] * a0 + ptr[ y0 * w + x1 ] * a1;
    float r1 = ptr[ y1 * w + x0 ] * a0 + ptr[ y1 * w + x1 ] * a1;

    float v = r0 * b0 + r1 * b1;

    return v;
}

int ref_roialign_fp32(float* data_in, float* data_out, float* roi_ptr,int size, struct roialign_param *param  , float scale, int zero_point)
{
    //printf("roi_in: %f %f %f %f %f \n", roi_ptr[0],roi_ptr[1],roi_ptr[2],roi_ptr[3],roi_ptr[4]);
    //printf("data_in: %f %f %f %f \n", data_in[0],data_in[1],data_in[2],data_in[3]);

    int w = param->pooled_width;
    int h = param->pooled_height;
    float spatial_scale = param->spatial_scale;

    //printf("spatial %f \n", spatial_scale);

    float roi_x1 = roi_ptr[0] * spatial_scale;
    float roi_y1 = roi_ptr[1] * spatial_scale;
    float roi_x2 = roi_ptr[2] * spatial_scale;
    float roi_y2 = roi_ptr[3] * spatial_scale;

    float roi_w = T_MAX(roi_x2 - roi_x1, 1);
    float roi_h = T_MAX(roi_y2 - roi_y1, 1);

    float bin_size_w = roi_w / (float)w;
    float bin_size_h = roi_h / (float)h;  
    int channel = param->channel;

    int inDataHW = param->in_height*param->in_width;
    //printf("in height, width : %d %d \n",param->in_height, param->in_width);
    int outDataHW = param->out_height*param->out_width;
    //printf("out height, width : %d %d \n",param->out_height, param->out_width);
    //printf("%f %f \n", bin_size_h, bin_size_w);

    for(int q = 0; q < channel; q++){
        float* ptr = data_in + q * inDataHW;
        float* outptr = data_out + q * outDataHW;
        for(int ph = 0; ph < h ; ph++){
            for(int pw = 0; pw < w; pw++){
                
                float hstart = roi_y1 + ph * bin_size_h;
                float wstart = roi_x1 + pw * bin_size_w;
                float hend = roi_y1 + (ph + 1) * bin_size_h;
                float wend = roi_x1 + (pw + 1) * bin_size_w;

                hstart = T_MIN(T_MAX(hstart, 0.f), (float)param->in_height);
                wstart = T_MIN(T_MAX(wstart, 0.f), (float)param->in_width);
                hend = T_MIN(T_MAX(hend, 0.f), (float)param->in_height);
                wend = T_MIN(T_MAX(wend, 0.f), (float)param->in_width);

                int bin_grid_h = ceil(hend - hstart);
                int bin_grid_w = ceil(wend - wstart);

                bool is_empty = (hend <= hstart) || (wend <= wstart);
                int area = bin_grid_h * bin_grid_w;  

                float sum = 0.f;
                for (int by = 0; by < bin_grid_h; by++)
                {
                    float y = hstart + (by + 0.5f) * bin_size_h / (float)bin_grid_h;

                    for (int bx = 0; bx < bin_grid_w; bx++)
                    {
                        float x = wstart + (bx + 0.5f) * bin_size_w / (float)bin_grid_w;

                        // bilinear interpolate at (x,y)
                        float v = bilinear_interpolate(ptr, param->in_width, param->in_height, x, y);
                        sum += v;
                    }
                }      
                outptr[pw] = is_empty ? 0.f : (sum / (float)area);        
            }
            outptr += w;
        }

    }


    return 0;
}
