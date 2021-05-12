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
 * Author: haoluo@openailab.com
 */

#include "fc_kernel_arm.h"

#include "api/c_api.h"
#include "utility/sys_port.h"

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <arm_neon.h>


#ifdef __aarch64__
void sgemv_1x8_a72(float* biases, float* input, float* kernel, long kernel_size, float* output);
void sgemv_1x2_a72(float* biases, float* input, float* kernel, long kernel_size, float* output);
#else
void sgemv_1x8_a17(float* biases, float* input, float* kernel, int kernel_size, float* output);
void sgemv_1x2_a17(float* biases, float* input, float* kernel, int kernel_size, float* output);
#endif

typedef void (*kernel_t)(float* biases, float* input, float* kernel, int kernel_size, float* output);

static void sgemv1x8(float* input, float* output, float* kernel, float* bias, int kernel_size, int start_ch, int end_ch,
                     int num_thread, kernel_t kernel_1x8)
{
#pragma omp parallel for num_threads(num_thread)
    for (int ch = start_ch; ch < end_ch; ch += 8)
    {
        float* cur_kernel = kernel + ch * kernel_size;
        float* cur_output = output + ch;
        float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float* cur_bias = bias ? bias + ch : zeros;

        kernel_1x8(cur_bias, input, cur_kernel, kernel_size, cur_output);
    }
}

static void sgemv1x2(float* input, float* output, float* kernel, float* bias, int kernel_size, int start_ch, int end_ch,
                     int num_thread, kernel_t kernel_1x2)
{
    int end_ch2 = end_ch & -2;

#pragma omp parallel for num_threads(num_thread)
    for (int ch = start_ch; ch < end_ch2; ch += 2)
    {
        float* cur_kernel = kernel + ch * kernel_size;
        float* cur_output = output + ch;
        float zeros[2] = {0.f, 0.f};
        float* cur_bias = bias ? bias + ch : zeros;

        kernel_1x2(cur_bias, input, cur_kernel, kernel_size, cur_output);
    }
    int ch = end_ch2;
    if (end_ch & 0x1)
    {
        float* cur_kernel = kernel + end_ch2 * kernel_size;
        float* cur_output = output + end_ch2;
        float sum = bias ? bias[ch] : 0.f;
        for (int i = 0; i < kernel_size; i++)
            sum += input[i] * cur_kernel[i];

        *cur_output = sum;
    }
}

static void interleave_kernel(const float* kernel, float* kernel_interleaved, int out_chan, int kernel_size)
{
    int i, j, k;
    float* cur_kernel[8];
    float* cur_kernel_interleaved;

    // interleave 8 kernel
    for (i = 0; i < (out_chan & -8); i += 8)
    {
        for (j = 0; j < 8; j++)
            cur_kernel[j] = ( float* )kernel + kernel_size * (i + j);
        cur_kernel_interleaved = ( float* )kernel_interleaved + kernel_size * i;
        for (k = 0; k < kernel_size; k++)
            for (j = 0; j < 8; j++)
                cur_kernel_interleaved[8 * k + j] = *(cur_kernel[j] + k);
    }

    // interleave 2 kernel
    for (; i < (out_chan & -2); i += 2)
    {
        for (j = 0; j < 2; j++)
            cur_kernel[j] = ( float* )kernel + kernel_size * (i + j);
        cur_kernel_interleaved = ( float* )kernel_interleaved + kernel_size * i;
        for (k = 0; k < kernel_size; k++)
            for (j = 0; j < 2; j++)
                cur_kernel_interleaved[2 * k + j] = *(cur_kernel[j] + k);
    }

    // copy last kernel
    if (out_chan & 0x1)
    {
        cur_kernel[0] = ( float* )kernel + kernel_size * i;
        cur_kernel_interleaved = ( float* )kernel_interleaved + kernel_size * i;
        for (k = 0; k < kernel_size; k++)
            cur_kernel_interleaved[k] = *(cur_kernel[0] + k);
    }
}

int fc_kernel_prerun(struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* output_tensor,
                     struct fc_priv_info* priv_info, struct fc_param* param)
{
    int num_output  = param->num_output;
    int kernel_size = filter_tensor->dims[1];

    if (!priv_info->interleave_buffer)
    {
        int elemsize = input_tensor->elem_size;
        int mem_size = elemsize * num_output * kernel_size;
        void* mem = sys_malloc(mem_size);
        priv_info->interleave_buffer = mem;
        priv_info->interleave_buffer_size = mem_size;
    }

    float* filter_data = ( float* )filter_tensor->data;
    interleave_kernel(filter_data, ( float* )priv_info->interleave_buffer, num_output, kernel_size);

    return 0;
}

int fc_kernel_postrun(struct fc_priv_info* priv_info)
{
    if (priv_info->interleave_buffer != NULL)
    {
        sys_free(priv_info->interleave_buffer);
        priv_info->interleave_buffer = NULL;
        priv_info->interleave_buffer_size = 0;
    }
    if (priv_info->input_buffer != NULL)
    {
        sys_free(priv_info->input_buffer);
        priv_info->input_buffer = NULL;
        priv_info->input_buffer_size = 0;
    }

    return 0;
}

int fc_kernel_run(struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* bias_tensor,
                  struct tensor* output_tensor, struct fc_priv_info* priv_info, struct fc_param* param,
                  int num_thread, int cpu_affinity)
{
    int out_num = param->num_output;
    int kernel_size = filter_tensor->dims[1];

    float* input  = input_tensor->data;
    float* output = output_tensor->data;
    float* biases = NULL;
    if (bias_tensor)
        biases = bias_tensor->data;
    float* weight = priv_info->interleave_buffer;
    int remain_out_start = (out_num >> 3) << 3;

    /* set cpu affinity sgemv kernel */
    kernel_t kernel_1x8;
    kernel_t kernel_1x2;

#ifdef __aarch64__
    kernel_1x8 = (kernel_t)sgemv_1x8_a72;
    kernel_1x2 = (kernel_t)sgemv_1x2_a72;
#else
    kernel_1x8 = (kernel_t)sgemv_1x8_a17;
    kernel_1x2 = (kernel_t)sgemv_1x2_a17;
#endif

    /* process */
    for (int i = 0; i < input_tensor->dims[0]; i++)
    {
        float* cur_input = input + i * kernel_size;
        float* cur_output = output + i * out_num;

        sgemv1x8(cur_input, cur_output, weight, biases, kernel_size, 0, remain_out_start, num_thread, kernel_1x8);
        if (out_num & 0x7)
            sgemv1x2(cur_input, cur_output, weight, biases, kernel_size, remain_out_start, out_num, num_thread, kernel_1x2);
    }

    return 0;
}
