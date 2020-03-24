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
 * Author: zpluo@openailab.com
 */

void run_l2pool(float* data, float* out_data, l2pool_param* param)
{
    for(int c = 0; c < param->inc; c++)
    {
        for(int ph = 0; ph < param->outh; ph++)
        {
            for(int pw = 0; pw < param->outw; pw++)
            {
                // int index = inc * (ph * outw + pw) + c;
                int index = param->inc * (ph * param->outw + pw) + c;
                int h_start = ph * param->stride_h - param->pad_h;
                int h_end = L2POOL_MIN(h_start + param->k_h, param->inh + param->pad_h);
                int w_start = pw * param->stride_w - param->pad_w;
                int w_end = L2POOL_MIN(w_start + param->k_w, param->inw + param->pad_w);
                h_start = L2POOL_MAX(0, ph * param->stride_h - param->pad_h);
                w_start = L2POOL_MAX(0, pw * param->stride_w - param->pad_w);
                h_end = L2POOL_MIN(h_end, param->inh);
                w_end = L2POOL_MIN(w_end, param->inw);
                int pool_size = 0;

                float tmp = 0.0f;
                float val = 0.0f;
                for(int h = h_start; h < h_end; h++)
                {
                    for(int w = w_start; w < w_end; w++)
                    {
                        // val = data[i*param->inh*param->inc * param->inw +h * param->inc * param->inw + w * param->inc
                        // +c];
                        val = data[h * param->inc * param->inw + w * param->inc + c];
                        tmp += val * val;
                        pool_size++;
                    }
                }
                if(tmp == 0)
                {
                    out_data[index] = 0;
                }
                else
                {
                    out_data[index] = sqrt(tmp / pool_size);
                }
            }
        }
    }
}

int ref_l2pool_fp32(float* data, float* out_data, l2pool_param* param)
{
    int input_size = param->inc * param->inh * param->inw;
    int output_size = param->outh * param->outw * param->outc;
    for(int i = 0; i < param->inn; i++)
    {
        run_l2pool(data + i * input_size, out_data + i * output_size,param);
    }
    return 0;
}
