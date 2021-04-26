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

#include "hybrid_fc_kernel_arm.h"

#include "api/c_api.h"
#include "utility/sys_port.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <arm_neon.h>


#ifdef __aarch64__
#define ROUND(x) round(x)
void i8gemv_1x8_a72_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale);
void i8gemv_1x2_a72_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale);
void i8gemv_1x8_a53_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale);
void i8gemv_1x2_a53_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale);
void get_max_arm64(float* data, int size, float* max);

static void get_max(const float* data, int size, float* max)
{
    get_max_arm64(( float* )data, size, ( float* )max);

    if(*max < 0.0001)
        *max = 0.0001;

    return;
}
#else
#define ROUND(x) round_na(x)

static inline int round_na(float input)
{
    int output;

#ifdef ANDROID
    asm("vcvtr.s32.f32 %[fp32], %[fp32] \n\t"
        "vmov   %[int32], %[fp32]\n\t"
        : [int32] "=r"(output), [fp32] "+X"(input));
#else
    asm("vcvtr.s32.f32 %[fp32], %[fp32] \n\t"
        "vmov   %[int32], %[fp32]\n\t"
        : [int32] "=r"(output), [fp32] "+w"(input));

#endif
    return output;
};

void i8gemv_1x8_a17_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale);
void i8gemv_1x2_a17_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale);
void i8gemv_1x8_a7_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale);
void i8gemv_1x2_a7_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale);
void get_max_arm32(float* data, int size, float* max);

static void get_max(const float* data, int size, float* max)
{
    get_max_arm32(( float* )data, size, ( float* )max);

    if(*max < 0.0001)
        *max = 0.0001;

    return;
}

#endif

static void i8gemv1x8(int8_t* input, float* output, int8_t* kernel, float* bias, float* scale,
        int kernel_size, int start_ch, int end_ch, int num_thread, int cpu_affinity)
{
#pragma omp parallel for num_threads(num_thread)
    for(int ch = start_ch; ch < end_ch; ch +=8)
    {
        float* cur_scale = scale + ch;
        int8_t* cur_kernel = kernel + ch * kernel_size;
        float* cur_output = output + ch;
        float* cur_bias = bias ? bias + ch : bias;
#ifdef __aarch64__
        if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
            i8gemv_1x8_a53_chan(cur_bias, input, cur_kernel, kernel_size, cur_output, cur_scale);
        else
            i8gemv_1x8_a72_chan(cur_bias, input, cur_kernel, kernel_size, cur_output, cur_scale);
#else
        if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
            i8gemv_1x8_a7_chan(cur_bias, input, cur_kernel, kernel_size, cur_output, cur_scale);
        else
            i8gemv_1x8_a17_chan(cur_bias, input, cur_kernel, kernel_size, cur_output, cur_scale);
#endif
    }
}

static void i8gemv1x2(int8_t* input, float* output, int8_t* kernel, float* bias, float* scale,
        int kernel_size, int start_ch, int end_ch, int num_thread, int cpu_affinity)
{
    int end_ch2 = end_ch & -2;
#pragma omp parallel for num_threads(num_thread)
    for(int ch = start_ch; ch < end_ch2; ch +=2)
    {
        float* cur_scale = scale + ch;
        int8_t* cur_kernel = kernel + ch * kernel_size;
        float* cur_output = output + ch;
        float* cur_bias = bias ? bias + ch : bias;
#ifdef __aarch64__
        if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
            i8gemv_1x2_a53_chan(cur_bias, input, cur_kernel, kernel_size, cur_output, cur_scale);
        else
            i8gemv_1x2_a72_chan(cur_bias, input, cur_kernel, kernel_size, cur_output, cur_scale);
#else
        if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
            i8gemv_1x2_a7_chan(cur_bias, input, cur_kernel, kernel_size, cur_output, cur_scale);
        else
            i8gemv_1x2_a17_chan(cur_bias, input, cur_kernel, kernel_size, cur_output, cur_scale);
#endif
    }
    if(end_ch & 0x1)
    {
        int ch = end_ch2;
        int8_t* cur_kernel = kernel + ch * kernel_size;
        float* cur_output = output + ch;
        float sum = 0;
        for(int i = 0; i < kernel_size; i++)
            sum += input[i] * cur_kernel[i];

        *cur_output = sum * scale[ch] + (bias ? bias[ch] : 0.f);
    }
}

static void interleave_kernel(const float* kernel, int8_t* kernel_interleaved, int out_chan, int kernel_size, float* kernel_max)
{
    int i, j, k;
    float* cur_kernel[8];
    int8_t* cur_kernel_interleaved;
    int kernel_size_align = (kernel_size + 1) & -2;
    float scale[out_chan];
    for(int i = 0; i < out_chan; i++)
        scale[i] = 127 / kernel_max[i];
    // interleave 8 kernel
    for(i = 0; i < (out_chan & -8); i += 8)
    {
        for(j = 0; j < 8; j++)
            cur_kernel[j] = ( float* )kernel + kernel_size * (i + j);
        cur_kernel_interleaved = ( int8_t* )kernel_interleaved + kernel_size_align * i;
        for(k = 0; k < (kernel_size & -2); k+=2)
            for(j = 0; j < 8; j++)
            {
                cur_kernel_interleaved[8 * k + j * 2] = ROUND(*(cur_kernel[j] + k) * scale[i + j]);
                cur_kernel_interleaved[8 * k + j * 2 + 1] = ROUND(*(cur_kernel[j] + k + 1) * scale[i + j]);
            }
        if(kernel_size & 0x1)
        {
            for(j = 0; j < 8; j++)
            {
                cur_kernel_interleaved[8 * k + j * 2] = ROUND(*(cur_kernel[j] + k) * scale[i + j]);
                cur_kernel_interleaved[8 * k + j * 2 + 1] = 0;
            }
        }
    }
    // interleave 2 kernel
    for(; i < (out_chan & -2); i += 2)
    {
        for(j = 0; j < 2; j++)
            cur_kernel[j] = ( float* )kernel + kernel_size * (i + j);
        cur_kernel_interleaved = ( int8_t* )kernel_interleaved + kernel_size_align * i;
        for(k = 0; k < (kernel_size & -2); k+=2)
            for(j = 0; j < 2; j++)
            {
                cur_kernel_interleaved[2 * k + j * 2] = ROUND(*(cur_kernel[j] + k) * scale[i + j]);
                cur_kernel_interleaved[2 * k + j * 2 + 1] = ROUND(*(cur_kernel[j] + k + 1) * scale[i + j]);
            }
        if(kernel_size & 0x1)
        {
            for(j = 0; j < 2; j++)
            {
                cur_kernel_interleaved[2 * k + j * 2] = ROUND(*(cur_kernel[j] + k) * scale[i + j]);
                cur_kernel_interleaved[2 * k + j * 2 + 1] = 0;
            }
        }
    }
    // copy last kernel
    if(out_chan & 0x1)
    {
        cur_kernel[0] = ( float* )kernel + kernel_size * i;
        cur_kernel_interleaved = ( int8_t* )kernel_interleaved + kernel_size_align * i;
        for(k = 0; k < kernel_size; k++)
            cur_kernel_interleaved[k] = ROUND(*(cur_kernel[0] + k) * scale[i + j]);
    }
    return;
}

int hybrid_fc_kernel_prerun(struct tensor*  input_tensor , \
                    struct tensor*  filter_tensor ,  \
                    struct tensor*  output_tensor , \
                    struct fc_priv_info*  priv_info , \
                    struct fc_param* param)
{
	int num_output = param->num_output;
	int kernel_size = filter_tensor->dims[1];
    int kernel_align = ((kernel_size + 1) & -2);
	
    if (!priv_info->interleave_buffer)
    {
        int mem_size = sizeof(int8_t) * num_output * kernel_align;
        void* mem = sys_malloc(mem_size);
        priv_info->interleave_buffer = mem;
        priv_info->interleave_buffer_size = mem_size;
    }
    if (!priv_info->input_buffer)
    {
        int mem_size = sizeof(int8_t) * kernel_align;
        void* mem = sys_malloc(mem_size);
        priv_info->input_buffer = mem;
        priv_info->input_buffer_size = mem_size;
    }
    if (!priv_info->kernel_max)
        priv_info->kernel_max = (float*)sys_malloc(num_output * sizeof(float));
    float* filter_data = (float*)filter_tensor->data;
    float* kernel_max = priv_info->kernel_max;
    for(int i = 0; i < num_output; i ++)
    {
        float* cur_weight = filter_data + i * kernel_size;
        float tmp_max;
        get_max(cur_weight, kernel_size, &tmp_max);
        kernel_max[i] = tmp_max;
    }
    interleave_kernel(filter_data, (int8_t*)priv_info->interleave_buffer, num_output, kernel_size, kernel_max);
    
    return 0;
}

static void quantify_input(const float* input, int8_t* input_int8, int size, float input_max)
{
    float* cur_input = ( float* )input;
    int8_t* cur_input_int8 = input_int8;
    float scale = 127 / input_max;
    int i, j;

    for(i = 0; i < (size & -16); i += 16)
        for(j = 0; j < 16; j++)
            *cur_input_int8++ = ROUND(*cur_input++ * scale);

    for(; i < (size & -2); i++)
        *cur_input_int8++ = ROUND(*cur_input++ * scale);

    if(size & 0x1)
        *cur_input_int8++ = 0;

    return;
}

int hybrid_fc_kernel_run(struct tensor* input_tensor , \
                    struct tensor* filter_tensor , \
                    struct tensor* bias_tensor ,  \
                    struct tensor* output_tensor , \
                    struct fc_priv_info* priv_info , \
                    struct fc_param* param, \
                    int num_thread, int cpu_affinity)
{
    int out_num = param->num_output;
    int kernel_size = filter_tensor->dims[1];

    float* input = input_tensor->data;
    float* output = output_tensor->data;
    float* biases = NULL;
    if (bias_tensor)
        biases = bias_tensor->data;
    int8_t* weight = (int8_t*)priv_info->interleave_buffer;
    int8_t* input_int8 = (int8_t*)priv_info->input_buffer;
    float* kernel_max = (float*)priv_info->kernel_max;
    int out_num_8 = out_num & ~7;

    float out_scale[out_num];
    for(int i = 0; i < input_tensor->dims[0]; i++)
    {
        float* cur_input = input + i * kernel_size;
        float* cur_output = output + i * out_num;

        float input_max;
        get_max(cur_input, kernel_size, &input_max);
        
        quantify_input(cur_input, input_int8, kernel_size, input_max);
        for(int i = 0; i < out_num; i++)
        {
            out_scale[i] = (input_max * kernel_max[i]) / (127 * 127);
        }

        i8gemv1x8(input_int8, cur_output, weight, biases, out_scale, kernel_size, 0, out_num_8, num_thread, cpu_affinity);
        if(out_num & 0x7)
            i8gemv1x2(input_int8, cur_output, weight, biases, out_scale, kernel_size, out_num_8, out_num, num_thread, cpu_affinity);
    }

    return 0;
}

int hybrid_fc_kernel_postrun(struct fc_priv_info* priv_info)
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
    if (priv_info->kernel_max != NULL)
    {
        sys_free(priv_info->kernel_max);
        priv_info->kernel_max = NULL;
    }

    return 0;
}