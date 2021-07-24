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
 * Author: 1091545398@qq.com
 */

#include "fc_kernel_int8_arm.h"

#include "utility/sys_port.h"

#include <stdint.h>
#include <string.h>
#include <math.h>

#include <arm_neon.h>

void gemv_1x8_int8(int32_t* biases, const float* scales, int8_t* inp, int8_t* kernel, long kernel_size,
                   int8_t* output)
{
    int8x8_t input;
    int8x16_t weight_0_1, weight_2_3, weight_4_5, weight_6_7;
    int16x8_t weight0_16, weight1_16, weight2_16, weight3_16;
    int16x8_t weight4_16, weight5_16, weight6_16, weight7_16;
    int32x4_t res = {0, 0, 0, 0};
    int32x4_t res1 = {0, 0, 0, 0};
    int8_t* input_ptr = inp;
    int8_t* weight_ptr = kernel;
    int remainw = (kernel_size >> 3) << 3;
    for (int i = 0; i < remainw; i = i + 8)
    {
        input = vld1_s8(input_ptr);
        weight_0_1 = vld1q_s8(weight_ptr);
        weight_2_3 = vld1q_s8(weight_ptr + 16);
        weight_4_5 = vld1q_s8(weight_ptr + 32);
        weight_6_7 = vld1q_s8(weight_ptr + 48);

        weight0_16 = vmull_s8(vdup_n_s8(vget_lane_s8(input, 0)), vget_low_s8(weight_0_1));
        weight1_16 = vmull_s8(vdup_n_s8(vget_lane_s8(input, 1)), vget_high_s8(weight_0_1));
        weight2_16 = vmull_s8(vdup_n_s8(vget_lane_s8(input, 2)), vget_low_s8(weight_2_3));
        weight3_16 = vmull_s8(vdup_n_s8(vget_lane_s8(input, 3)), vget_high_s8(weight_2_3));
        weight4_16 = vmull_s8(vdup_n_s8(vget_lane_s8(input, 4)), vget_low_s8(weight_4_5));
        weight5_16 = vmull_s8(vdup_n_s8(vget_lane_s8(input, 5)), vget_high_s8(weight_4_5));
        weight6_16 = vmull_s8(vdup_n_s8(vget_lane_s8(input, 6)), vget_low_s8(weight_6_7));
        weight7_16 = vmull_s8(vdup_n_s8(vget_lane_s8(input, 7)), vget_high_s8(weight_6_7));

        res = vaddq_s32(res, vaddl_s16(vget_low_s16(weight0_16), vget_low_s16(weight1_16)));
        res = vaddq_s32(res, vaddl_s16(vget_low_s16(weight2_16), vget_low_s16(weight3_16)));
        res = vaddq_s32(res, vaddl_s16(vget_low_s16(weight4_16), vget_low_s16(weight5_16)));
        res = vaddq_s32(res, vaddl_s16(vget_low_s16(weight6_16), vget_low_s16(weight7_16)));

        res1 = vaddq_s32(res1, vaddl_s16(vget_high_s16(weight0_16), vget_high_s16(weight1_16)));
        res1 = vaddq_s32(res1, vaddl_s16(vget_high_s16(weight2_16), vget_high_s16(weight3_16)));
        res1 = vaddq_s32(res1, vaddl_s16(vget_high_s16(weight4_16), vget_high_s16(weight5_16)));
        res1 = vaddq_s32(res1, vaddl_s16(vget_high_s16(weight6_16), vget_high_s16(weight7_16)));

        input_ptr += 8;
        weight_ptr += 64;
    }

    for (int i = remainw; i < kernel_size; ++i)
    {
        weight0_16 = vmull_s8(vdup_n_s8(input_ptr[0]), vld1_s8(weight_ptr));
        res = vaddq_s32(vmovl_s16(vget_low_s16(weight0_16)), res);
        res1 = vaddq_s32(vmovl_s16(vget_high_s16(weight0_16)), res1);
        input_ptr += 1;
        weight_ptr += 8;
    }

    if (biases)
    {
        int32x4_t bias = vld1q_s32(biases);
        int32x4_t bias1 = vld1q_s32(biases + 4);
        res = vaddq_s32(res, bias);
        res1 = vaddq_s32(res1, bias1);
    }

    float32x4_t res_f = vcvtq_f32_s32(res);
    float32x4_t res1_f = vcvtq_f32_s32(res1);

    float32x4_t scale = vld1q_f32(scales);
    float32x4_t scale_1 = vld1q_f32(scales + 4);

    res_f = vmulq_f32(res_f, scale);
    res1_f = vmulq_f32(res1_f, scale_1);
    res_f = vaddq_f32(res_f, vdupq_n_f32(0.5f));
    res1_f = vaddq_f32(res1_f, vdupq_n_f32(0.5f));

    res = vcvtq_s32_f32(res_f);
    res1 = vcvtq_s32_f32(res1_f);
    int16x4_t res_16 = vmovn_s32(res);
    int16x4_t res1_16 = vmovn_s32(res1);
    int8x8_t result = vmovn_s16(vcombine_s16(res_16, res1_16));
    int8x8_t _m127 = vdup_n_s8(127);
    int8x8_t _m_127 = vdup_n_s8(-127);
    result = vmax_s8(_m_127, result);
    result = vmin_s8(_m127, result);
    vst1_s8(output, result);
}

void gemv_1x2_int8(const int32_t* biases, const float* scales, int8_t* inp, int8_t* kernel, long kernel_size,
                   int8_t* output)
{
    int8_t* input_ptr = inp;
    int8_t* weight_ptr = kernel;
    int remainw = (kernel_size << 3) >> 3;
    int8x8x2_t weight;
    int8x8_t input;
    int16x8_t out_16_0, out_16_1;
    int32x4_t out_32_0, out_32_1;
    int32_t sum0 = 0, sum1 = 0;
    for (int i = 0; i < remainw; i = i + 8)
    {
        weight = vld2_s8(weight_ptr);
        input = vld1_s8(input_ptr);
        out_16_0 = vmull_s8(weight.val[0], input);
        out_16_1 = vmull_s8(weight.val[1], input);
        out_32_0 = vpaddlq_s16(out_16_0);
        out_32_1 = vpaddlq_s16(out_16_1);
        sum0 += vgetq_lane_s32(out_32_0, 0) + vgetq_lane_s32(out_32_0, 1) + vgetq_lane_s32(out_32_0, 2) + vgetq_lane_s32(out_32_0, 3);
        sum1 += vgetq_lane_s32(out_32_1, 0) + vgetq_lane_s32(out_32_1, 1) + vgetq_lane_s32(out_32_1, 2) + vgetq_lane_s32(out_32_1, 3);
        weight_ptr += 16;
        input_ptr += 8;
    }

    for (int i = remainw; i < kernel_size; ++i)
    {
        sum0 += weight_ptr[0] * input_ptr[0];
        sum1 += weight_ptr[1] * input_ptr[0];
        input_ptr++;
        weight_ptr += 2;
    }

    if (biases)
    {
        sum0 += biases[0];
        sum1 += biases[1];
    }

    int data_i32_0 = round(sum0 * scales[0]);
    if (data_i32_0 > 127)
        data_i32_0 = 127;
    else if (data_i32_0 < -127)
        data_i32_0 = -127;

    int data_i32_1 = round(sum1 * scales[1]);
    if (data_i32_1 > 127)
        data_i32_1 = 127;
    else if (data_i32_0 < -127)
        data_i32_1 = -127;

    output[0] = data_i32_0;
    output[1] = data_i32_1;
}

// start and end channel must be 8 aligned
void gemv1x8(const int8_t* input, const int8_t* output, int8_t* weight_interleaved,
             const int32_t* biases, const float* scales,
             int kernel_size, int start_channel, int end_channel, int num_thread,
             int cpu_affinity)
{
    int ch = 0;
    int8_t *cur_kernel, *cur_result;
    int32_t* cur_biases;
    const float* cur_scales;

    // #pragma omp parallel for num_threads(num_thread)
    for (ch = start_channel; ch < end_channel; ch += 8)
    {
        cur_kernel = (int8_t*)(weight_interleaved + kernel_size * ch);
        cur_result = (int8_t*)(output + ch);
        cur_biases = biases ? (int32_t*)(biases + ch) : NULL;
        cur_scales = scales + ch;
        gemv_1x8_int8(cur_biases, cur_scales, (int8_t*)input, cur_kernel, kernel_size,
                      cur_result);
    }
}

// start channel must be 2 aligned
void gemv1x2(const int8_t* input, int8_t* output, int8_t* weight_interleaved,
             const int32_t* biases, const float* scales,
             int kernel_size, int start_channel, int end_channel, int num_thread, int cpu_affinity)
{
    int32_t sum;
    int ch = 0;
    int8_t* cur_kernel;
    int32_t* cur_biases;
    int8_t* cur_result;
    const float* cur_scales;

    for (ch = start_channel; ch < (end_channel & -2); ch += 2)
    {
        cur_kernel = (int8_t*)(weight_interleaved + kernel_size * ch);
        cur_result = (int8_t*)(output + ch);
        cur_biases = biases ? (int32_t*)(biases + ch) : NULL;
        cur_scales = scales + ch;
        gemv_1x2_int8(cur_biases, cur_scales, (int8_t*)input, cur_kernel, kernel_size, cur_result);
    }

    if (end_channel & 0x1)
    {
        cur_kernel = (int8_t*)(weight_interleaved + kernel_size * ch);
        cur_result = (int8_t*)(output + ch);
        sum = biases ? *(biases + ch) : 0;
        for (int j = 0; j < kernel_size; j++)
            sum = sum + input[j] * cur_kernel[j];
        int data_i32_0 = round(sum * cur_scales[0]);
        if (data_i32_0 > 127)
            data_i32_0 = 127;
        else if (data_i32_0 < -127)
            data_i32_0 = -127;
        *cur_result = data_i32_0;
    }
}

static void interleave_kernel(const int8_t* kernel, int8_t* kernel_interleaved, int out_chan, int kernel_size)
{
    int i, j, k;
    int8_t* cur_kernel[8];
    int8_t* cur_kernel_interleaved;

    // interleave 8 kernel
    for (i = 0; i < (out_chan & -8); i += 8)
    {
        for (j = 0; j < 8; j++)
            cur_kernel[j] = (int8_t*)kernel + kernel_size * (i + j);
        cur_kernel_interleaved = (int8_t*)kernel_interleaved + kernel_size * i;
        for (k = 0; k < kernel_size; k++)
            for (j = 0; j < 8; j++)
                cur_kernel_interleaved[8 * k + j] = *(cur_kernel[j] + k);
    }

    // interleave 2 kernel
    for (; i < (out_chan & -2); i += 2)
    {
        for (j = 0; j < 2; j++)
            cur_kernel[j] = (int8_t*)kernel + kernel_size * (i + j);
        cur_kernel_interleaved = (int8_t*)kernel_interleaved + kernel_size * i;
        for (k = 0; k < kernel_size; k++)
            for (j = 0; j < 2; j++)
                cur_kernel_interleaved[2 * k + j] = *(cur_kernel[j] + k);
    }

    // copy last kernel
    if (out_chan & 0x1)
    {
        cur_kernel[0] = (int8_t*)kernel + kernel_size * i;
        cur_kernel_interleaved = (int8_t*)kernel_interleaved + kernel_size * i;
        for (k = 0; k < kernel_size; k++)
            cur_kernel_interleaved[k] = *(cur_kernel[0] + k);
    }

    return;
}

int int8_fc_kernel_prerun(struct tensor* input_tensor,
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
        int mem_size = num_output * kernel_align;
        void* mem = sys_malloc(mem_size);
        priv_info->interleave_buffer = mem;
        priv_info->interleave_buffer_size = mem_size;
    }
    if (!priv_info->input_buffer)
    {
        int mem_size = kernel_align;
        void* mem = sys_malloc(mem_size);
        priv_info->input_buffer = mem;
        priv_info->input_buffer_size = mem_size;
    }

    int8_t* filter_data = (int8_t*)filter_tensor->data;

    interleave_kernel(filter_data, (int8_t*)priv_info->interleave_buffer, num_output,
                      kernel_size);

    return 0;
}

int int8_fc_kernel_run(struct tensor* input_tensor,
                       struct tensor* filter_tensor,
                       struct tensor* bias_tensor,
                       struct tensor* output_tensor,
                       struct fc_priv_info* priv_info,
                       struct fc_param* param,
                       int num_thread, int cpu_affinity)
{
    int out_num = param->num_output;
    int kernel_size = filter_tensor->dims[1];

    int8_t* input = (int8_t*)input_tensor->data;
    int8_t* output = (int8_t*)output_tensor->data;
    int8_t* weight = (int8_t*)priv_info->interleave_buffer;
    int32_t* biases = NULL;
    if (bias_tensor)
        biases = (int32_t*)bias_tensor->data;

    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    float* weight_scales = filter_tensor->scale_list;
    float* requant_scales = (float*)malloc(out_num * sizeof(float));

    for (int i = 0; i < out_num; i++)
        requant_scales[i] = (input_scale * weight_scales[i]) / output_scale;

    int out_num_8 = out_num & ~7;

    for (int i = 0; i < input_tensor->dims[0]; i++)
    {
        int8_t* cur_input = input + i * kernel_size;
        int8_t* cur_output = output + i * out_num;

        gemv1x8(cur_input, cur_output, weight, biases, requant_scales, kernel_size, 0, out_num_8, num_thread, cpu_affinity);
        if (out_num & 0x7)
            gemv1x2(cur_input, cur_output, weight, biases, requant_scales, kernel_size, out_num_8, out_num, num_thread, cpu_affinity);
    }

    return 0;
}
