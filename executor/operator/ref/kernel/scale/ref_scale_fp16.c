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

int ref_scale_fp16(__fp16* in_data, __fp16* out_data,__fp16* gamma_data,__fp16* beta_data, scale_param* param)
{
/* for arm32 && x86 */
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8

    int img_size = param->channel_number * param->channel_size;
    for(int i = 0; i < param->batch_number; i++)
    {
        for(int c = 0; c < param->channel_number; c++)
        {
            int offset = i * img_size + c * param->channel_size;
            for(int l = 0; l < param->channel_size; l++)
            {
                if(beta_data != NULL)
                    out_data[offset + l] = fp32_to_fp16(fp16_to_fp32(in_data[offset + l]) * gamma_data[c] + beta_data[c]);
                else
                    out_data[offset + l] = fp32_to_fp16(fp16_to_fp32(in_data[offset + l]) * gamma_data[c]);
            }
        }
    }


#else
    int img_size = param->channel_number * param->channel_size;
    for(int i = 0; i < param->batch_number; i++)
    {
        for(int c = 0; c < param->channel_num; c++)
        {
            int offset = i * img_size + c * param->channel_size;
            for(int l = 0; l < param->channel_size; l++)
            {
                if(beta_data != NULL)
                    out_data[offset + l] = in_data[offset + l] * gamma_data[c] + beta_data[c];
                else
                    out_data[offset + l] = in_data[offset + l] * gamma_data[c];
            }
        }
    }
#endif
    return 0;
}
