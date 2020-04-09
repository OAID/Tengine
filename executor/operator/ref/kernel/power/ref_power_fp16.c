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

int ref_power_fp16(__fp16* input, __fp16* output,ref_power_param* param)
{
/* for arm32 && x86 */
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8

    for(int n = 0; n < param->iDataN; n++)
    {
        for(int c = 0; c < param->iDataC; c++)
        {
            for(int h = 0; h < param->iDataH; h++)
            {
                for(int w = 0; w < param->iDataW; w++)
                {
                    int size = n * param->iDataC * param->iDataH * param->iDataW + c * param->iDataH * param->iDataW + h * param->iDataW + w;
                    output[size] = fp32_to_fp16(pow((param->shift + param->scale * fp16_to_fp32(input[size])), param->power));
                }
            }
        }
    }
#else
    for(int n = 0; n < param->iDataN; n++)
    {
        for(int c = 0; c < param->iDataC; c++)
        {
            for(int h = 0; h < param->iDataH; h++)
            {
                for(int w = 0; w < param->iDataW; w++)
                {
                    int size = n * param->iDataC * param->iDataH * param->iDataW + c * param->iDataH * param->iDataW + h * param->iDataW + w;
                    output[size] = pow((param->shift + param->scale * input[size]), param->power);
                }
            }
        }
    }
#endif
    return 0;
}
