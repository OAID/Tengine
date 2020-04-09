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
 * Author: chunyinglv@openailab.com
 */


#include <iostream>

#include "logger.hpp"
#include "tensor_mem.hpp"

#include "graph.hpp"
#include "node_ops.hpp"
#include "operator/deconvolution.hpp"
#include <arm_neon.h>


#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

#define DECONV_DW_MAX(a, b) ((a) > (b) ? (a) : (b))
#define DECONV_DW_MIN(a, b) ((a) < (b) ? (a) : (b))


#include <math.h>
namespace TEngine {

namespace deconv_2d_dw {

inline float do_activation(float input, int activation)
{
    if(activation == 0)
    {
        input = DECONV_DW_MAX(input, 0);
        if(activation == 6)
            input =DECONV_DW_MIN(input, 6);
    }
    return input;
}

const char* conv_name = "DECONV_DW";
const int default_prio = 99;

inline void deconv_dw_genreal_3x3s1(const float *input, const float *kernel,float *output, 
                                int group, int input_h, int input_w,
                                int output_h, int output_w)
{
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

            for (; j+3 < input_w; j+=4)
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

inline void deconv_dw_genreal_3x3s2(const float *input, const float *kernel,float *output, 
                                int group, int input_h, int input_w,
                                int output_h, int output_w)
{
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

            for (; j+3 < input_w; j+=4)
            {
                float32x4_t input_4 = vld1q_f32(cur_input);

                // out row 0
                float32x4_t out_00 = vmulq_lane_f32(input_4, vget_low_f32(_k0), 0);   // 0,2,4,6
                float32x4_t out_01 = vmulq_lane_f32(input_4, vget_low_f32(_k0), 1);   // 1,3,5,7
                float32x4_t out_02 = vmulq_lane_f32(input_4, vget_high_f32(_k0), 0);   // 2,4,6,8

                float32x4x2_t out_0 = vld2q_f32(cur_out0);
                out_0.val[0] = vaddq_f32(out_0.val[0], out_00);     // 0,2,4,6
                out_0.val[1] = vaddq_f32(out_0.val[1], out_01);     // 1,3,5,7
                vst2q_f32(cur_out0, out_0);

                out_0 = vld2q_f32(cur_out0 + 2);
                out_0.val[0] = vaddq_f32(out_0.val[0], out_02);     // 2,4,6,8
                vst2q_f32(cur_out0 + 2, out_0);

                // out row 1
                float32x4_t out_10 = vmulq_lane_f32(input_4, vget_low_f32(_k1), 0);    // 0,2,4,6
                float32x4_t out_11 = vmulq_lane_f32(input_4, vget_low_f32(_k1), 1);    // 1,3,5,7
                float32x4_t out_12 = vmulq_lane_f32(input_4, vget_high_f32(_k1), 0);    // 2,4,6,8

                float32x4x2_t out_1 = vld2q_f32(cur_out1);
                out_1.val[0] = vaddq_f32(out_1.val[0], out_10);     // 0,2,4,6
                out_1.val[1] = vaddq_f32(out_1.val[1], out_11);     // 1,3,5,7
                vst2q_f32(cur_out1, out_1);

                out_1 = vld2q_f32(cur_out1 + 2);
                out_1.val[0] = vaddq_f32(out_1.val[0], out_12);      // 2,4,6,8
                vst2q_f32(cur_out1 + 2, out_1);

                // out row 2
                float32x4_t out_20 = vmulq_lane_f32(input_4, vget_low_f32(_k2), 0);     // 0,2,4,6
                float32x4_t out_21 = vmulq_lane_f32(input_4, vget_low_f32(_k2), 1);     // 1,3,5,7
                float32x4_t out_22 = vmulq_lane_f32(input_4, vget_high_f32(_k2), 0);     // 2,4,6,8

                float32x4x2_t out_2 = vld2q_f32(cur_out2);
                out_2.val[0] = vaddq_f32(out_2.val[0], out_20);     // 0,2,4,6
                out_2.val[1] = vaddq_f32(out_2.val[1], out_21);     // 1,3,5,7
                vst2q_f32(cur_out2, out_2);

                out_2 = vld2q_f32(cur_out2 + 2);
                out_2.val[0] = vaddq_f32(out_2.val[0], out_22);     // 2,4,6,8
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

inline void deconv_dw_genreal_4x4s1(const float *input, const float *kernel,float *output, 
                                int group, int input_h, int input_w,
                                int output_h, int output_w)
{
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

            for (; j+3 < input_w; j+=4)
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

inline void deconv_dw_genreal_kernel(const float *input, const float *kernel,float *output, 
                                int group_start,int group_end, int activation,
                                int input_c, int input_h, int input_w,
                                int output_c, int output_h, int output_w,
                                int kernel_h, int kernel_w,
                                int pad_h, int pad_w, int stride_h, int stride_w,
                                int dilation_h, int dilation_w)
{
    
    if(stride_h == 1 && kernel_h == 4)
    {
        int group = group_end - group_start;
        const float* cur_input = input + group_start * input_h * input_w;
        const float* cur_kernel = kernel + group_start * 16;
        float* cur_output = output + group_start * output_h * output_w;
        if(pad_h == 0)
            deconv_dw_genreal_4x4s1(cur_input, cur_kernel, cur_output, group, input_h, input_w, output_h, output_w);
        else
        {
            int out_h_pad = output_h + pad_h * 2;
            int out_w_pad = output_w + pad_w * 2;
            float* output_buf = (float*)malloc( sizeof(float) * group * out_h_pad * out_w_pad + 128);
            deconv_dw_genreal_4x4s1(cur_input, cur_kernel, output_buf, group, input_h, input_w, output_h, output_w);
            for(int i = 0 ;i < output_h ;i++)
            {
                float* cur_src = output_buf + (pad_h + i) * out_w_pad + pad_h;
                float* cur_dst = output_buf + i * output_w;
                for(int j = 0; j < output_w; j++)
                    cur_dst[j] += cur_src[j];
            }
        }
        return;
    }
    if(kernel_h == 3)
    {
        int group = group_end - group_start;
        const float* cur_input = input + group_start * input_h * input_w;
        const float* cur_kernel = kernel + group_start * 9;
        float* cur_output = output + group_start * output_h * output_w;
        if(pad_h == 0)
        {
            if(stride_h == 1)
                deconv_dw_genreal_3x3s1(cur_input, cur_kernel, cur_output, group, input_h, input_w, output_h, output_w);
            else
                deconv_dw_genreal_3x3s2(cur_input, cur_kernel, cur_output, group, input_h, input_w, output_h, output_w);
        }
        else
        {
            int out_h_pad = output_h + pad_h * 2;
            int out_w_pad = output_w + pad_w * 2;
            float* output_buf = (float*)malloc( sizeof(float) * group * out_h_pad * out_w_pad + 128);
            if(stride_h == 1)
                deconv_dw_genreal_3x3s1(cur_input, cur_kernel, cur_output, group, input_h, input_w, output_h, output_w);
            else
                deconv_dw_genreal_3x3s2(cur_input, cur_kernel, cur_output, group, input_h, input_w, output_h, output_w);
            for(int i = 0 ;i < output_h ;i++)
            {
                float* cur_src = output_buf + (pad_h + i) * out_w_pad + pad_h;
                float* cur_dst = output_buf + i * output_w;
                for(int j = 0; j < output_w; j++)
                    cur_dst[j] += cur_src[j];
            }
        }
        return;
    }
    int c, h, w, kc, k_h, k_w;
    int org_out_x = 0;
    int org_out_y = 0;
    int cur_out_x = 0;
    int cur_out_y = 0;

    float input_val;
    float weight_val;

    int input_offset = 0;
    int kernel_offset = 0;
    int output_offset = 0;
    int out_hw = output_w * output_h;
    for(int g= group_start; g < group_end; g++)
    {
        for (h = 0; h < input_h; h++)
        {
            for (w = 0; w < input_w; w++)
            {
                org_out_x = w * stride_w - pad_w;
                org_out_y = h * stride_h - pad_h;
                for (kc = 0; kc < input_c; kc++)
                {
                    input_offset =(g*input_c + kc) * input_h * input_w + h * input_w + w;
                    input_val = input[input_offset];
                    for (c = 0; c < output_c; c++)
                    {
                        for (k_h = 0; k_h < kernel_h; k_h++)
                        {
                            for (k_w = 0; k_w < kernel_w; k_w++)
                            {
                                cur_out_x = org_out_x + k_w * dilation_w;
                                cur_out_y = org_out_y + k_h * dilation_h;
                                if (cur_out_x >= 0 && cur_out_x < output_w && cur_out_y >= 0 && cur_out_y < output_h)
                                {

                                    kernel_offset = (g*output_c* input_c+  kc * output_c +c)* kernel_h * kernel_w +
                                                    k_h * kernel_w + k_w;

                                    output_offset = (g * output_c+c) * output_w * output_h + cur_out_y * output_w + cur_out_x;

                                    weight_val = kernel[kernel_offset];
                                    output[output_offset] += weight_val * input_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if(activation == 0 || activation == 6)
    {
        int out_offset = group_start* out_hw;
        int out_end = group_end* out_hw;
        for(int i=out_offset;i<out_end;i++)
        {
            output[i]=do_activation(output[i],activation);
        }
    }
}


void initial_output(float* output, float* bias, int output_ch, int output_wh)
{
    int i, j;
    // no bias
    if(bias == nullptr)
    {
        memset(output, 0.f, output_ch * output_wh* sizeof(float));
    }
    else
    {
        float* out_ptr= output;
        for(i = 0; i < output_ch; i++)
            for(j = 0; j < output_wh; j++)
                *out_ptr++ = bias[i];
    }
}

struct deconv_dw_param
{
    float* input_buf;
    float* weight_buf;
    float* output_buf;

    int group_start;
    int group_end;
    int activation;

    int input_c;
    int input_h;
    int input_w;

    int output_c;
    int output_h;
    int output_w;

    int ker_h;
    int ker_w;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
};

struct DeConv2dDepth : public MTNodeOps
{
    DeConv2dDepth()
    {
        name_ = "arm_dw_conv_fp32";
    }

    int activation;

    bool Run(Node* node);

    bool Aider(int cpu, int seq, void* data);
};

bool DeConv2dDepth::Aider(int cpu, int seq, void* data)
{
    deconv_dw_param* param = ( deconv_dw_param* )data;

    deconv_dw_genreal_kernel(param->input_buf, param->weight_buf,param->output_buf,
                param->group_start,param->group_end,param->activation,
                param->input_c,param->input_h, param->input_w,  
                param->output_c, param->output_h, param->output_w,
                param->ker_h,param->ker_w,param->pad_h,param->pad_w,param->stride_h,param->stride_w,
                param->dilation_h,param->dilation_w);
    return true;
}


bool DeConv2dDepth::Run(Node* node)
{

    Tensor* input_tensor = node->GetInputTensor(0);
    Deconvolution* deconv_op = dynamic_cast<Deconvolution*>(node->GetOp());
    DeconvParam* param_ = deconv_op->GetParam();
    int group = param_->group;
    int kernel_h = param_->kernel_h;
    int kernel_w = param_->kernel_w;
    int stride_h = param_->stride_h;
    int stride_w = param_->stride_w;
    int dilation_h = param_->dilation_h;
    int dilation_w = param_->dilation_w;
    int pad_h = param_->pad_h0;
    int pad_w = param_->pad_w0;

    const TShape& input_shape = input_tensor->GetShape();

    int input_c = input_shape.GetC();
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();

    /* output */
    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& output_shape = output_tensor->GetShape();

    int output_h = output_shape.GetH();
    int output_w = output_shape.GetW();
    int output_n = output_shape.GetN();
    int output_c = output_shape.GetC();
    int output_hw = output_h * output_w;

    int input_c0 = input_c/group;
    int output_c0 = output_c/group;

    Tensor* weight_tensor = node->GetInputTensor(1);
    float* weight_buf = ( float* )get_tensor_mem(weight_tensor);
    float* input_buf = ( float* )get_tensor_mem(input_tensor);
    float* output_buf = ( float* )get_tensor_mem(output_tensor);
    int input_size = input_c * input_h * input_w;
    int output_size = output_c * output_h * output_w;

    int cpu_number = cpu_info->GetCPUNumber();

    float* bias = nullptr;

    //get bias
    if(node->GetInputNum() > 2)
    {
        Tensor* bias_tensor = node->GetInputTensor(2);
        bias = ( float* )get_tensor_mem(bias_tensor);
    }
    for(int b = 0; b < output_n; b++)
    {
        float* cur_input = input_buf + b * input_size;
        float* cur_output = output_buf + b * output_size;

        initial_output(cur_output, bias, output_c, output_hw);
        if(cpu_number == 1)
        {
            deconv_dw_genreal_kernel(cur_input,
                            weight_buf,
                            cur_output,
                            0,group, 
                            activation,
                            input_c0,input_h, input_w,
                            output_c0,output_h,output_w,
                            kernel_h,kernel_w,
                            pad_h,pad_w,stride_h,stride_w,
                            dilation_h,dilation_w);
        }
        else
        {
            std::vector<sub_op_task> task_list;
            std::vector<deconv_dw_param> param_list;
            int step = group/cpu_number;
            int task_number = cpu_number;
            if(group <=cpu_number)
            {
                task_number=group;
                step=1;
            }
            task_list.resize(task_number);
            param_list.resize(task_number);

            auto f = std::bind(&DeConv2dDepth::Aider, this, std::placeholders::_1, std::placeholders::_2,
                                std::placeholders::_3);
            for(int i = 0; i < task_number; i++)
            {
                
                deconv_dw_param* param = &param_list[i];
                sub_op_task* task = &task_list[i];
                task->exec_func = f;
                task->seq = i;
                task->data = param;

                param->input_buf = cur_input;
                param->weight_buf = weight_buf;
                param->output_buf = cur_output;
                param->group_start = i*step;
                param->group_end = param->group_start + step;
                param->activation = activation;
                param->input_c=input_c0;
                param->input_h = input_h;
                param->input_w = input_w;
                param->output_c = output_c0;
                param->output_h = output_h;
                param->output_w = output_w;

                param->ker_h = kernel_h;
                param->ker_w = kernel_w;
                param->pad_h = pad_h;
                param->pad_w = pad_w;
                param->stride_h = stride_h;
                param->stride_w = stride_w;
                param->dilation_h = dilation_h;
                param->dilation_w = dilation_w;

            }
            param_list[task_number - 1].group_end = group;
            task_dispatch(task_list, -1);
            wait_done();
        }
    }
    return true;
}

static bool isDepthwiseSupported(const DeconvParam* param)
{
    int group = param->group;
    int out_c = param->num_output;
    if(group != out_c)
    {
        return false;
    }
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));

#ifdef CONFIG_AUTH_DEVICE
    bool float_enabled = get_auth_float_enabled();

    if(!float_enabled)
        return nullptr;
#endif

    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)

        return nullptr;

    Operator* op = node->GetOp();

    Deconvolution* deconv_op = dynamic_cast<Deconvolution*>(op);
    DeconvParam* param = deconv_op->GetParam();

    if(!isDepthwiseSupported(param))
        return nullptr;

    DeConv2dDepth* ops = new DeConv2dDepth();
    ops->activation = param->activation;

    return ops;
}

}    // namespace deconv_2d_dw

void RegisterDeConv2dDepth(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Deconvolution", deconv_2d_dw::SelectFunc,
                                                      deconv_2d_dw::default_prio))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio[" << deconv_2d_dw::default_prio << "]\n";
}

}    // namespace TEngine
