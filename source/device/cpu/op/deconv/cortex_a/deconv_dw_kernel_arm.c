/*
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    License); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
*/

/*
    Copyright (c) 2021, OPEN AI LAB
    Author: haoluo@openailab.com
*/

#include "deconv_dw_kernel_arm.h"

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include <arm_neon.h>

#define DECONV_DW_MAX(a, b) ((a) > (b) ? (a) : (b))
#define DECONV_DW_MIN(a, b) ((a) < (b) ? (a) : (b))

#ifdef __aarch64__
#else
#endif

inline static float do_activation(float input, int activation)
{
    if (activation == 0)
    {
        input = DECONV_DW_MAX(input, 0);

        if (activation == 6)
            input = DECONV_DW_MIN(input, 6);
    }

    return input;
}

inline static void initial_output(float* output, float* bias, int output_ch, int output_wh)
{
    int i, j;

    // no bias
    if (bias == NULL)
    {
        memset(output, 0.f, output_ch * output_wh * sizeof(float));
    }
    else
    {
        float* out_ptr = output;

        for (i = 0; i < output_ch; i++)
            for (j = 0; j < output_wh; j++)
                *out_ptr++ = bias[i];
    }
}

inline static void deconv_dw_genreal_3x3s1(const float* input, const float* kernel, float* output, int group,
                                           int input_h, int input_w, int output_h, int output_w, int num_thread)
{
#pragma omp parallel for num_threads(num_thread)
    for (int c = 0; c < group; c++)
    {
        const float* cur_input = input + c * input_h * input_w;
        const float* cur_kernel = kernel + c * 9;
        float32x4_t _k0 = vld1q_f32(cur_kernel);
        float32x4_t _k1 = vld1q_f32(cur_kernel + 3);
        float32x4_t _k2 = vld1q_f32(cur_kernel + 6);

        for (int i = 0; i < input_h; i++)
        {
            float* cur_out0 = output + c * output_h * output_w + output_w * i;
            float* cur_out1 = output + c * output_h * output_w + output_w * (i + 1);
            float* cur_out2 = output + c * output_h * output_w + output_w * (i + 2);

            int j = 0;

            for (; j + 3 < input_w; j += 4)
            {
                float32x4_t input_4 = vld1q_f32(cur_input);

                float32x4_t out_00 = vld1q_f32(cur_out0 + 0);
                out_00 = vmlaq_lane_f32(out_00, input_4, vget_low_f32(_k0), 0);
                vst1q_f32(cur_out0 + 0, out_00);

                float32x4_t out_01 = vld1q_f32(cur_out0 + 1);
                out_01 = vmlaq_lane_f32(out_01, input_4, vget_low_f32(_k0), 1);
                vst1q_f32(cur_out0 + 1, out_01);

                float32x4_t out_02 = vld1q_f32(cur_out0 + 2);
                out_02 = vmlaq_lane_f32(out_02, input_4, vget_high_f32(_k0), 0);
                vst1q_f32(cur_out0 + 2, out_02);

                float32x4_t out_10 = vld1q_f32(cur_out1 + 0);
                out_10 = vmlaq_lane_f32(out_10, input_4, vget_low_f32(_k1), 0);
                vst1q_f32(cur_out1 + 0, out_10);

                float32x4_t out_11 = vld1q_f32(cur_out1 + 1);
                out_11 = vmlaq_lane_f32(out_11, input_4, vget_low_f32(_k1), 1);
                vst1q_f32(cur_out1 + 1, out_11);

                float32x4_t out_12 = vld1q_f32(cur_out1 + 2);
                out_12 = vmlaq_lane_f32(out_12, input_4, vget_high_f32(_k1), 0);
                vst1q_f32(cur_out1 + 2, out_12);

                float32x4_t out_20 = vld1q_f32(cur_out2 + 0);
                out_20 = vmlaq_lane_f32(out_20, input_4, vget_low_f32(_k2), 0);
                vst1q_f32(cur_out2 + 0, out_20);

                float32x4_t out_21 = vld1q_f32(cur_out2 + 1);
                out_21 = vmlaq_lane_f32(out_21, input_4, vget_low_f32(_k2), 1);
                vst1q_f32(cur_out2 + 1, out_21);

                float32x4_t out_22 = vld1q_f32(cur_out2 + 2);
                out_22 = vmlaq_lane_f32(out_22, input_4, vget_high_f32(_k2), 0);
                vst1q_f32(cur_out2 + 2, out_22);

                cur_input += 4;
                cur_out0 += 4;
                cur_out1 += 4;
                cur_out2 += 4;
            }
            for (; j < input_w; j++)
            {
                float val = cur_input[0];

                cur_out0[0] += val * cur_kernel[0];
                cur_out0[1] += val * cur_kernel[1];
                cur_out0[2] += val * cur_kernel[2];

                cur_out1[0] += val * cur_kernel[3];
                cur_out1[1] += val * cur_kernel[4];
                cur_out1[2] += val * cur_kernel[5];

                cur_out2[0] += val * cur_kernel[6];
                cur_out2[1] += val * cur_kernel[7];
                cur_out2[2] += val * cur_kernel[8];

                cur_input++;
                cur_out0++;
                cur_out1++;
                cur_out2++;
            }
        }
    }
}

inline static void deconv_dw_genreal_3x3s2(const float* input, const float* kernel, float* output, int group,
                                           int input_h, int input_w, int output_h, int output_w, int num_thread)
{
#pragma omp parallel for num_threads(num_thread)
    for (int c = 0; c < group; c++)
    {
        const float* cur_input = input + c * input_h * input_w;
        const float* cur_kernel = kernel + c * 9;
        float32x4_t _k0 = vld1q_f32(cur_kernel);
        float32x4_t _k1 = vld1q_f32(cur_kernel + 3);
        float32x4_t _k2 = vld1q_f32(cur_kernel + 6);

        for (int i = 0; i < input_h; i++)
        {
            float* cur_out0 = output + c * output_h * output_w + output_w * i * 2;
            float* cur_out1 = output + c * output_h * output_w + output_w * (i * 2 + 1);
            float* cur_out2 = output + c * output_h * output_w + output_w * (i * 2 + 2);

            int j = 0;

            for (; j + 3 < input_w; j += 4)
            {
                float32x4_t input_4 = vld1q_f32(cur_input);

                // out row 0
                float32x4_t out_00 = vmulq_lane_f32(input_4, vget_low_f32(_k0), 0);  // 0,2,4,6
                float32x4_t out_01 = vmulq_lane_f32(input_4, vget_low_f32(_k0), 1);  // 1,3,5,7
                float32x4_t out_02 = vmulq_lane_f32(input_4, vget_high_f32(_k0), 0); // 2,4,6,8

                float32x4x2_t out_0 = vld2q_f32(cur_out0);
                out_0.val[0] = vaddq_f32(out_0.val[0], out_00); // 0,2,4,6
                out_0.val[1] = vaddq_f32(out_0.val[1], out_01); // 1,3,5,7
                vst2q_f32(cur_out0, out_0);

                out_0 = vld2q_f32(cur_out0 + 2);
                out_0.val[0] = vaddq_f32(out_0.val[0], out_02); // 2,4,6,8
                vst2q_f32(cur_out0 + 2, out_0);

                // out row 1
                float32x4_t out_10 = vmulq_lane_f32(input_4, vget_low_f32(_k1), 0);  // 0,2,4,6
                float32x4_t out_11 = vmulq_lane_f32(input_4, vget_low_f32(_k1), 1);  // 1,3,5,7
                float32x4_t out_12 = vmulq_lane_f32(input_4, vget_high_f32(_k1), 0); // 2,4,6,8

                float32x4x2_t out_1 = vld2q_f32(cur_out1);
                out_1.val[0] = vaddq_f32(out_1.val[0], out_10); // 0,2,4,6
                out_1.val[1] = vaddq_f32(out_1.val[1], out_11); // 1,3,5,7
                vst2q_f32(cur_out1, out_1);

                out_1 = vld2q_f32(cur_out1 + 2);
                out_1.val[0] = vaddq_f32(out_1.val[0], out_12); // 2,4,6,8
                vst2q_f32(cur_out1 + 2, out_1);

                // out row 2
                float32x4_t out_20 = vmulq_lane_f32(input_4, vget_low_f32(_k2), 0);  // 0,2,4,6
                float32x4_t out_21 = vmulq_lane_f32(input_4, vget_low_f32(_k2), 1);  // 1,3,5,7
                float32x4_t out_22 = vmulq_lane_f32(input_4, vget_high_f32(_k2), 0); // 2,4,6,8

                float32x4x2_t out_2 = vld2q_f32(cur_out2);
                out_2.val[0] = vaddq_f32(out_2.val[0], out_20); // 0,2,4,6
                out_2.val[1] = vaddq_f32(out_2.val[1], out_21); // 1,3,5,7
                vst2q_f32(cur_out2, out_2);

                out_2 = vld2q_f32(cur_out2 + 2);
                out_2.val[0] = vaddq_f32(out_2.val[0], out_22); // 2,4,6,8
                vst2q_f32(cur_out2 + 2, out_2);

                cur_input += 4;
                cur_out0 += 8;
                cur_out1 += 8;
                cur_out2 += 8;
            }
            for (; j < input_w; j++)
            {
                float val = cur_input[0];

                cur_out0[0] += val * cur_kernel[0];
                cur_out0[1] += val * cur_kernel[1];
                cur_out0[2] += val * cur_kernel[2];

                cur_out1[0] += val * cur_kernel[4];
                cur_out1[1] += val * cur_kernel[5];
                cur_out1[2] += val * cur_kernel[6];

                cur_out2[0] += val * cur_kernel[8];
                cur_out2[1] += val * cur_kernel[9];
                cur_out2[2] += val * cur_kernel[10];

                cur_input++;
                cur_out0 += 2;
                cur_out1 += 2;
                cur_out2 += 2;
            }
        }
    }
}

inline static void deconv_dw_genreal_4x4s1(const float* input, const float* kernel, float* output, int group,
                                           int input_h, int input_w, int output_h, int output_w, int num_thread)
{
#pragma omp parallel for num_threads(num_thread)
    for (int c = 0; c < group; c++)
    {
        const float* cur_input = input + c * input_h * input_w;
        const float* cur_kernel = kernel + c * 16;
        float32x4_t _k0 = vld1q_f32(cur_kernel);
        float32x4_t _k1 = vld1q_f32(cur_kernel + 4);
        float32x4_t _k2 = vld1q_f32(cur_kernel + 8);
        float32x4_t _k3 = vld1q_f32(cur_kernel + 12);

        for (int i = 0; i < input_h; i++)
        {
            float* cur_out0 = output + c * output_h * output_w + output_w * i;
            float* cur_out1 = output + c * output_h * output_w + output_w * (i + 1);
            float* cur_out2 = output + c * output_h * output_w + output_w * (i + 2);
            float* cur_out3 = output + c * output_h * output_w + output_w * (i + 3);

            int j = 0;

            for (; j + 3 < input_w; j += 4)
            {
                float32x4_t input_4 = vld1q_f32(cur_input);

                float32x4_t out_00 = vld1q_f32(cur_out0 + 0);
                out_00 = vmlaq_lane_f32(out_00, input_4, vget_low_f32(_k0), 0);
                vst1q_f32(cur_out0 + 0, out_00);

                float32x4_t out_01 = vld1q_f32(cur_out0 + 1);
                out_01 = vmlaq_lane_f32(out_01, input_4, vget_low_f32(_k0), 1);
                vst1q_f32(cur_out0 + 1, out_01);

                float32x4_t out_02 = vld1q_f32(cur_out0 + 2);
                out_02 = vmlaq_lane_f32(out_02, input_4, vget_high_f32(_k0), 0);
                vst1q_f32(cur_out0 + 2, out_02);

                float32x4_t out_03 = vld1q_f32(cur_out0 + 3);
                out_03 = vmlaq_lane_f32(out_03, input_4, vget_high_f32(_k0), 1);
                vst1q_f32(cur_out0 + 3, out_03);

                float32x4_t out_10 = vld1q_f32(cur_out1 + 0);
                out_10 = vmlaq_lane_f32(out_10, input_4, vget_low_f32(_k1), 0);
                vst1q_f32(cur_out1 + 0, out_10);

                float32x4_t out_11 = vld1q_f32(cur_out1 + 1);
                out_11 = vmlaq_lane_f32(out_11, input_4, vget_low_f32(_k1), 1);
                vst1q_f32(cur_out1 + 1, out_11);

                float32x4_t out_12 = vld1q_f32(cur_out1 + 2);
                out_12 = vmlaq_lane_f32(out_12, input_4, vget_high_f32(_k1), 0);
                vst1q_f32(cur_out1 + 2, out_12);

                float32x4_t out_13 = vld1q_f32(cur_out1 + 3);
                out_13 = vmlaq_lane_f32(out_13, input_4, vget_high_f32(_k1), 1);
                vst1q_f32(cur_out1 + 3, out_13);

                float32x4_t out_20 = vld1q_f32(cur_out2 + 0);
                out_20 = vmlaq_lane_f32(out_20, input_4, vget_low_f32(_k2), 0);
                vst1q_f32(cur_out2 + 0, out_20);

                float32x4_t out_21 = vld1q_f32(cur_out2 + 1);
                out_21 = vmlaq_lane_f32(out_21, input_4, vget_low_f32(_k2), 1);
                vst1q_f32(cur_out2 + 1, out_21);

                float32x4_t out_22 = vld1q_f32(cur_out2 + 2);
                out_22 = vmlaq_lane_f32(out_22, input_4, vget_high_f32(_k2), 0);
                vst1q_f32(cur_out2 + 2, out_22);

                float32x4_t out_23 = vld1q_f32(cur_out2 + 3);
                out_23 = vmlaq_lane_f32(out_23, input_4, vget_high_f32(_k2), 1);
                vst1q_f32(cur_out2 + 3, out_23);

                float32x4_t out_30 = vld1q_f32(cur_out3 + 0);
                out_30 = vmlaq_lane_f32(out_30, input_4, vget_low_f32(_k3), 0);
                vst1q_f32(cur_out3 + 0, out_30);

                float32x4_t out_31 = vld1q_f32(cur_out3 + 1);
                out_31 = vmlaq_lane_f32(out_31, input_4, vget_low_f32(_k3), 1);
                vst1q_f32(cur_out3 + 1, out_31);

                float32x4_t out_32 = vld1q_f32(cur_out3 + 2);
                out_32 = vmlaq_lane_f32(out_32, input_4, vget_high_f32(_k3), 0);
                vst1q_f32(cur_out3 + 2, out_32);

                float32x4_t out_33 = vld1q_f32(cur_out3 + 3);
                out_33 = vmlaq_lane_f32(out_33, input_4, vget_high_f32(_k3), 1);
                vst1q_f32(cur_out3 + 3, out_33);

                cur_input += 4;
                cur_out0 += 4;
                cur_out1 += 4;
                cur_out2 += 4;
                cur_out3 += 4;
            }

            for (; j < input_w; j++)
            {
                float val = cur_input[0];

                cur_out0[0] += val * cur_kernel[0];
                cur_out0[1] += val * cur_kernel[1];
                cur_out0[2] += val * cur_kernel[2];
                cur_out0[3] += val * cur_kernel[3];

                cur_out1[0] += val * cur_kernel[4];
                cur_out1[1] += val * cur_kernel[5];
                cur_out1[2] += val * cur_kernel[6];
                cur_out1[3] += val * cur_kernel[7];

                cur_out2[0] += val * cur_kernel[8];
                cur_out2[1] += val * cur_kernel[9];
                cur_out2[2] += val * cur_kernel[10];
                cur_out2[3] += val * cur_kernel[11];

                cur_out3[0] += val * cur_kernel[12];
                cur_out3[1] += val * cur_kernel[13];
                cur_out3[2] += val * cur_kernel[14];
                cur_out3[3] += val * cur_kernel[15];

                cur_input++;
                cur_out0++;
                cur_out1++;
                cur_out2++;
                cur_out3++;
            }
        }
    }
}

void deconv_dw_genreal(float* input, float* output, float* kernel, int input_h, int input_w, int output_h, int output_w,
                       int group, int kernel_h, int kernel_w, int stride_h, int stride_w, int dilation_h,
                       int dilation_w, int pad_h, int pad_w, int activation, int num_thread)
{
    int h, w, k_h, k_w;

#pragma omp parallel for num_threads(num_thread)
    for (int g = 0; g < group; g++)
    {
        for (h = 0; h < input_h; h++)
        {
            for (w = 0; w < input_w; w++)
            {
                int org_out_x = w * stride_w - pad_w;
                int org_out_y = h * stride_h - pad_h;
                int input_offset = g * input_h * input_w + h * input_w + w;

                for (k_h = 0; k_h < kernel_h; k_h++)
                {
                    for (k_w = 0; k_w < kernel_w; k_w++)
                    {
                        int cur_out_x = org_out_x + k_w * dilation_w;
                        int cur_out_y = org_out_y + k_h * dilation_h;

                        if (cur_out_x >= 0 && cur_out_x < output_w && cur_out_y >= 0 && cur_out_y < output_h)
                        {
                            int kernel_offset = g * kernel_h * kernel_w + k_h * kernel_w + k_w;

                            int output_offset = g * output_w * output_h + cur_out_y * output_w + cur_out_x;

                            output[output_offset] += kernel[kernel_offset] * input[input_offset];
                        }
                    }
                }
            }
        }
    }

    if (activation == 0 || activation == 6)
    {
        int size = group * output_h * output_w;

        for (int i = 0; i < size; i++)
        {
            output[i] = do_activation(output[i], activation);
        }
    }
}

int deconv_dw_run(struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* bias_tensor,
                  struct tensor* output_tensor, struct deconv_param* param, int num_thread, int cpu_affinity)
{
    /* param */
    int pads[4];
    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    pads[0] = param->pad_h0;
    pads[1] = param->pad_w0;
    pads[2] = param->pad_h1;
    pads[3] = param->pad_w1;

    if (stride_h != stride_w)
        return -1;

    int act_type = param->activation;

    int batch = input_tensor->dims[0];
    int in_c = input_tensor->dims[1] / group;
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];
    int input_size = in_c * in_h * in_w;
    int kernel_size = in_c * kernel_h * kernel_w;

    int out_c = output_tensor->dims[1] / group;
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int out_hw = out_h * out_w;
    int output_size = out_c * out_h * out_w;
    int out_c_align = ((out_c + 3) & -4);
    /* buffer addr */
    float* input_buf = (float*)input_tensor->data;
    float* kernel_buf = (float*)filter_tensor->data;
    float* output_buf = (float*)output_tensor->data;
    float* biases_buf = (float*)bias_tensor->data;

    for (int n = 0; n < batch; n++) // batch size
    {
        float* cur_input = input_buf + n * input_size * group;
        float* cur_output = output_buf + n * output_size * group;

        initial_output(cur_output, biases_buf, group, out_hw);

        if (pads[0] == 0)
        {
            if (stride_h == 1 && kernel_h == 4)
            {
                deconv_dw_genreal_4x4s1(cur_input, kernel_buf, cur_output, group, in_h, in_w, out_h, out_w, num_thread);
            }
            else if (kernel_h == 3)
            {
                if (stride_h == 1)
                    deconv_dw_genreal_3x3s1(cur_input, kernel_buf, cur_output, group, in_h, in_w, out_h, out_w,
                                            num_thread);
                else
                    deconv_dw_genreal_3x3s2(cur_input, kernel_buf, cur_output, group, in_h, in_w, out_h, out_w,
                                            num_thread);
            }
            else
            {
                deconv_dw_genreal(cur_input, cur_output, kernel_buf, in_h, in_w, out_h, out_w, group, kernel_h,
                                  kernel_w, stride_h, stride_w, dilation_h, dilation_w, pads[0], pads[1], act_type,
                                  num_thread);
            }
        }
        else
        {
            int out_h_pad = out_h + pads[0] * 2;
            int out_w_pad = out_w + pads[1] * 2;
            float* output_buf = (float*)malloc(sizeof(float) * group * out_h_pad * out_w_pad + 128);

            if (stride_h == 1 && kernel_h == 4)
            {
                deconv_dw_genreal_4x4s1(cur_input, kernel_buf, output_buf, group, in_h, in_w, out_h, out_w, num_thread);

                for (int g = 0; g < group; g++)
                {
                    for (int i = 0; i < out_h; i++)
                    {
                        float* cur_src = output_buf + g * out_h_pad * out_w_pad + (pads[0] + i) * out_w_pad + pads[1];
                        float* cur_dst = cur_output + g * out_hw + i * out_w;

                        for (int j = 0; j < out_w; j++)
                            cur_dst[j] += cur_src[j];
                    }
                }
            }
            else if (kernel_h == 3)
            {
                if (stride_h == 1)
                    deconv_dw_genreal_3x3s1(cur_input, kernel_buf, cur_output, group, in_h, in_w, out_h, out_w,
                                            num_thread);
                else
                    deconv_dw_genreal_3x3s2(cur_input, kernel_buf, cur_output, group, in_h, in_w, out_h, out_w,
                                            num_thread);

                for (int g = 0; g < group; g++)
                {
                    for (int i = 0; i < out_h; i++)
                    {
                        float* cur_src = output_buf + g * out_h_pad * out_w_pad + (pads[0] + i) * out_w_pad + pads[1];
                        float* cur_dst = cur_output + g * out_hw + i * out_w;

                        for (int j = 0; j < out_w; j++)
                            cur_dst[j] += cur_src[j];
                    }
                }
            }
            else
            {
                deconv_dw_genreal(cur_input, cur_output, kernel_buf, in_h, in_w, out_h, out_w, group, kernel_h,
                                  kernel_w, stride_h, stride_w, dilation_h, dilation_w, pads[0], pads[1], act_type,
                                  num_thread);
            }
            free(output_buf);
        }
    }

    return 0;
}
