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

int ref_relu_fp16(__fp16* in_data, __fp16* out_data, int size, float negative_slope, float scale, int zero_point)
{
/* for arm32 && x86 */
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8
    //__fp16* data = ( __fp16* )data;
    if(negative_slope == 0.0)
    {
        for(int i = 0; i < size; i++)
        {
            out_data[i] = fp32_to_fp16(RELU_MAX(fp16_to_fp32(in_data[i]), 0.0));
        }
    }
    else
    {
        for(int i = 0; i < size; i++)
        {
            float bias = negative_slope * RELU_MIN(fp16_to_fp32(in_data[i]), 0.f);
            out_data[i] = fp32_to_fp16(RELU_MAX(fp16_to_fp32(in_data[i]), 0.f) + bias);
        }
    }
#else
    if(negative_slope == 0)
    {
        for(int i = 0; i < size; i++)
        {
            out_data[i] = RELU_MAX(in_data[i], 0.f);
        }
    }
    else
    {
        for(int i = 0; i < size; i++)
        {
            out_data[i] = RELU_MAX(in_data[i], 0.f) + negative_slope * RELU_MIN(in_data[i], 0.f);
        }
    }
#endif
    return 0;
}
