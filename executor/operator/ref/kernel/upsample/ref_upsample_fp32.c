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

static int ref_upsample_fp32(float* input, float* output, upsample_param* param)
{
    for(int n = 0; n < param->batch; ++n)
    {
        for(int c = 0; c < param->channel; c++)
        {
            for(int h = 0; h < param->out_h; h++)
            {
                for(int w = 0; w < param->out_w; w++)
                {
                    int in_w = w / param->scale;
                    int in_h = h / param->scale;
                    int out_idx = n * param->channel * param->out_h * param->out_w + c * param->out_h * param->out_w + h * param->out_w + w;
                    int in_idx = n * param->channel * param->input_h * param->input_w + c * param->input_w * param->input_h + in_h * param->input_w + in_w;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }

    return 0;
}
