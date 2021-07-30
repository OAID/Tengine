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
 * Author: xlchen@openailab.com
 */

#include "fc_kernel_fp16_arm82.h"

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <arm_neon.h>

void hgemv_1x8_a55(__fp16* biases, __fp16* input, __fp16* kernel, long kernel_size, __fp16* output);
void hgemv_1x2_a55(__fp16* biases, __fp16* input, __fp16* kernel, long kernel_size, __fp16* output);

// start and end channel must be 8 aligned
void hgemv1x8(const __fp16* input, const __fp16* output, __fp16* weight_interleaved, const __fp16* biases,
              int kernel_size, int start_channel, int end_channel, int num_thread, int cpu_affinity)
{
    int ch = 0;
    __fp16 *cur_kernel, *cur_biases, *cur_result;

    // #pragma omp parallel for num_threads(num_thread)
    for (ch = start_channel; ch < end_channel; ch += 8)
    {
        cur_kernel = (__fp16*)(weight_interleaved + kernel_size * ch);
        cur_result = (__fp16*)(output + ch);
        cur_biases = biases ? (__fp16*)(biases + ch) : NULL;
        hgemv_1x8_a55(cur_biases, (__fp16*)input, cur_kernel, kernel_size, cur_result); // todo implement with A76
    }
}

// start channel must be 2 aligned
void hgemv1x2(const __fp16* input, const __fp16* output, __fp16* weight_interleaved, const __fp16* biases,
              int kernel_size, int start_channel, int end_channel, int num_thread, int cpu_affinity)
{
    __fp16 sum;
    int ch = 0;
    __fp16 *cur_kernel, *cur_biases, *cur_result;

    for (ch = start_channel; ch < (end_channel & -2); ch += 2)
    {
        cur_kernel = (__fp16*)(weight_interleaved + kernel_size * ch);
        cur_result = (__fp16*)(output + ch);
        cur_biases = biases ? (__fp16*)(biases + ch) : NULL;
        hgemv_1x2_a55(cur_biases, (__fp16*)input, cur_kernel, kernel_size, cur_result);
    }

    if (end_channel & 0x1)
    {
        cur_kernel = (__fp16*)(weight_interleaved + kernel_size * ch);
        cur_result = (__fp16*)(output + ch);
        sum = biases ? *(biases + ch) : 0.f;
        for (int j = 0; j < kernel_size; j++)
            sum = sum + input[j] * cur_kernel[j];
        *cur_result = sum;
    }
}

static void interleave_kernel(const __fp16* kernel, __fp16* kernel_interleaved, int out_chan, int kernel_size)
{
    int i, j, k;
    __fp16* cur_kernel[8];
    __fp16* cur_kernel_interleaved;

    // interleave 8 kernel
    for (i = 0; i < (out_chan & -8); i += 8)
    {
        for (j = 0; j < 8; j++)
            cur_kernel[j] = (__fp16*)kernel + kernel_size * (i + j);
        cur_kernel_interleaved = (__fp16*)kernel_interleaved + kernel_size * i;
        for (k = 0; k < kernel_size; k++)
            for (j = 0; j < 8; j++)
                cur_kernel_interleaved[8 * k + j] = *(cur_kernel[j] + k);
    }

    // interleave 2 kernel
    for (; i < (out_chan & -2); i += 2)
    {
        for (j = 0; j < 2; j++)
            cur_kernel[j] = (__fp16*)kernel + kernel_size * (i + j);
        cur_kernel_interleaved = (__fp16*)kernel_interleaved + kernel_size * i;
        for (k = 0; k < kernel_size; k++)
            for (j = 0; j < 2; j++)
                cur_kernel_interleaved[2 * k + j] = *(cur_kernel[j] + k);
    }

    // copy last kernel
    if (out_chan & 0x1)
    {
        cur_kernel[0] = (__fp16*)kernel + kernel_size * i;
        cur_kernel_interleaved = (__fp16*)kernel_interleaved + kernel_size * i;
        for (k = 0; k < kernel_size; k++)
            cur_kernel_interleaved[k] = *(cur_kernel[0] + k);
    }

    return;
}

int fp16_fc_kernel_prerun(struct tensor* input_tensor,
                          struct tensor* filter_tensor,
                          struct tensor* output_tensor,
                          struct fc_priv_info* priv_info,
                          struct fc_param* param)
{
    int num_output = param->num_output;
    int kernel_size = filter_tensor->dims[1];
    int kernel_align = ((kernel_size + 1) & -2);

    if (!priv_info->interleave_buffer)
    {
        int mem_size = sizeof(__fp16) * num_output * kernel_align;
        void* mem = sys_malloc(mem_size);
        priv_info->interleave_buffer = mem;
        priv_info->interleave_buffer_size = mem_size;
    }
    if (!priv_info->input_buffer)
    {
        int mem_size = sizeof(__fp16) * kernel_align;
        void* mem = sys_malloc(mem_size);
        priv_info->input_buffer = mem;
        priv_info->input_buffer_size = mem_size;
    }

    __fp16* filter_data = (__fp16*)filter_tensor->data;

    interleave_kernel(filter_data, (__fp16*)priv_info->interleave_buffer, num_output, kernel_size);

    return 0;
}

int fp16_fc_kernel_run(struct tensor* input_tensor,
                       struct tensor* filter_tensor,
                       struct tensor* bias_tensor,
                       struct tensor* output_tensor,
                       struct fc_priv_info* priv_info,
                       struct fc_param* param,
                       int num_thread, int cpu_affinity)
{
    int out_num = param->num_output;
    int kernel_size = filter_tensor->dims[1];

    __fp16* input = (__fp16*)input_tensor->data;
    __fp16* output = (__fp16*)output_tensor->data;
    __fp16* weight = (__fp16*)priv_info->interleave_buffer;
    __fp16* biases = NULL;
    if (bias_tensor)
        biases = (__fp16*)bias_tensor->data;

    int out_num_8 = out_num & ~7;

    for (int i = 0; i < input_tensor->dims[0]; i++)
    {
        __fp16* cur_input = input + i * kernel_size;
        __fp16* cur_output = output + i * out_num;

        hgemv1x8(cur_input, cur_output, weight, biases, kernel_size, 0, out_num_8, num_thread, cpu_affinity);
        if (out_num & 0x7)
            hgemv1x2(cur_input, cur_output, weight, biases, kernel_size, out_num_8, out_num, num_thread, cpu_affinity);
    }

    return 0;
}
