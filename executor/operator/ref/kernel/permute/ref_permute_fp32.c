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
 * Author: jjzeng@openailab.com
 */

static void __hwc(const float* input, float* output, int hh, int ww, int cc, int wc, int hw)
{
    for(int h = 0; h < hh; ++h)
    {
        float* out_ptr = output + h * wc;

        for(int w = 0; w < ww; ++w)
        {
            for(int c = 0; c < cc; ++c)
            {
                const float* in_ptr = input + c * hw + h * ww;
                out_ptr[w * cc + c] = in_ptr[w];
            }
        }
    }
}

static void __chw(const float* input, float* output, int hh, int ww, int cc, int wc, int hw)
{
    for(int c = 0; c < cc; ++c)
    {
        float* output_ptr = output + c * hw;    // chw
        for(int h = 0; h < hh; ++h)
        {
            for(int w = 0; w < ww; ++w)
            {
                const float* input_ptr = input + h * wc + w * cc;    // input hwc + wc
                // hw + w = input_ptr[c]
                output_ptr[h * ww + w] = input_ptr[c];
            }
        }
    }
}

static int ref_permute_fp32(const float* in_data, float* out_data, const permute_param* param)
{
    int n;
    int c;
    int h;
    int w;
    if(param->layout == TENGINE_LAYOUT_NCHW)
    {
        n = param->in_dim[0];
        c = param->in_dim[1];
        h = param->in_dim[2];
        w = param->in_dim[3];
    }
    else
    {
        n = param->in_dim[0];
        h = param->in_dim[1];
        w = param->in_dim[2];
        c = param->in_dim[3];
    }

    int wc = w * c;
    int hw = h * w;
    int chw = c * hw;

    const float* input = in_data;
    float* output = out_data;
    if(param->order0 == 0 && param->order1 == 2 && param->order2 == 3 && param->order3 == 1)
    {
        for(int ii = 0; ii < n; ++ii)
        {
            __hwc(input, output, h, w, c, wc, hw);

            input += chw;
            output += chw;
        }
    }
    else if(param->order0 == 0 && param->order1 == 3 && param->order2 == 1 && param->order3 == 2)
    {
        for(int ii = 0; ii < n; ++ii)
        {
            __chw(input, output, h, w, c, wc, hw);

            input += chw;
            output += chw;
        }
    }
    else if((param->order0 == 1) && (param->order1 == 0) && (param->order2 == 2))
    {
        int channel = param->in_dim[0];
        int width = param->in_dim[2];
        int height = param->in_dim[1];
        int _hw = height * width;
        int _cw = channel * width;
        for(int q = 0; q < height; q++)
        {
            float* outptr = output + q * _cw;

            for(int i = 0; i < channel; i++)
            {
                const float* ptr = input + i * _hw;

                for(int j = 0; j < width; j++)
                {
                    outptr[i * width + j] = ptr[q * width + j];
                }
            }
        }
    }
    else
    {
        return -1;
    }

    return 0;
}
