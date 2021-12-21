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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#include "batchnorm_kernel_arm.h"

#include <arm_neon.h>

static void batchnorm_kernel(int i, int id, void* data, const float* input, float* output, float* scale_mean,
                             float* scale_var, int channel_size, int num_thread)
{
    int step = ((int*)data)[0];

#pragma omp parallel for num_threads(num_thread)
    for (int c = 0; c < step; c++)
    {
        int cur_c = id * step + c;
        float s_mean = scale_mean[cur_c];
        float s_var = scale_var[cur_c];
        float32x4_t _mean = vdupq_n_f32(s_mean);
        float32x4_t _var = vdupq_n_f32(s_var);
        int offset = cur_c * channel_size;
        const float* input_ptr = input + offset;
        float* output_ptr = output + offset;

        for (int l = 0; l < (channel_size & -4); l += 4)
        {
            float32x4_t _input = vld1q_f32(input_ptr);
            vst1q_f32(output_ptr, vmlaq_f32(_mean, _input, _var));
            input_ptr += 4;
            output_ptr += 4;
        }
        for (int l = channel_size & ~3; l < channel_size; l++)
        {
            *output_ptr = (*input_ptr) * s_var + s_mean;
            input_ptr++;
            output_ptr++;
        }
    }
}

int batchnorm_run(struct tensor* output_tensor, struct tensor* input_tensor, float* scale_mean,
                  float* scale_var_inv, int num_thread)
{
    int batch_number = input_tensor->dims[0];
    int channel_num = input_tensor->dims[1];
    int channel_size;
    if (4 == input_tensor->dim_num)
    {
        channel_size = (input_tensor->dims[2]) * (input_tensor->dims[3]);
    }
    else if (3 == input_tensor->dim_num)
    {
        channel_size = (input_tensor->dims[2]);
    }
    else if (2 == input_tensor->dim_num)
    {
        channel_size = 1;
    }
    else
    {
        return -1;
    }
    int img_size = channel_num * channel_size;

    const float* input = (const float*)input_tensor->data;
    float* output = (float*)output_tensor->data;

    float* scale_mean_t = (float*)scale_mean;
    float* scale_var_inv_t = (float*)scale_var_inv;

    /* only use mean and var */
    for (int i = 0; i < batch_number; i++)
    {
        const float* cur_input = input + i * img_size;
        float* cur_output = output + i * img_size;

        batchnorm_kernel(0, 0, &channel_num, cur_input, cur_output, scale_mean_t, scale_var_inv_t, channel_size,
                         num_thread);
    }

    return 0;
}
