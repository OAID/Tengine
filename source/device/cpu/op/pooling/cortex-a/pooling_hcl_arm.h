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
 * Author: qtang@openailab.com
 */

#include "pooling_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "operator/op.h"
#include "utility/float.h"
#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include <arm_neon.h>

#define POOL_GENERIC 0
#define POOL_K2S2    1
#define POOL_K3S2    2
#define POOL_K3S1    3

typedef void (*pooling_kernel_t)(const void* input, void* output, int inc, int inh, int inw, int outh, int outw, int,
                                 int, int, int, int, int, int pad_h1, int pad_w1, int);

static void avg_2x2s2(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                      int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if (pad_w1 > 0)
    {
        outw--;
    }
    if (pad_h1 > 0)
    {
        outh--;
    }
    int block_w = outw >> 2;
    int remain_w = inw - outw * 2;

    for (int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        const float* line1 = line0 + inw;
        float* out_ptr = output + c * out_hw;

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p10 = vld1q_f32(line1);
                float32x4_t sum0 = vaddq_f32(p00, p10);

                float32x4_t p01 = vld1q_f32(line0 + 4);
                float32x4_t p11 = vld1q_f32(line1 + 4);
                float32x4_t sum1 = vaddq_f32(p01, p11);
#ifdef __aarch64__
                sum0 = vpaddq_f32(sum0, sum1);
#else
                float32x2_t sum0_1 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                float32x2_t sum0_2 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                sum0 = vcombine_f32(sum0_1, sum0_2);
#endif
                sum0 = vmulq_n_f32(sum0, 0.25f);
                vst1q_f32(out_ptr, sum0);
                line0 += 8;
                line1 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4; j < outw; j++)
            {
                float32x2_t p1 = vld1_f32(line0);
                float32x2_t p2 = vld1_f32(line1);
                float32x2_t sum = vadd_f32(p1, p2);

                *out_ptr = (sum[0] + sum[1]) * 0.25f;
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if (pad_w1)
            {
                *out_ptr = (line0[0] + line1[0]) * 0.5f;
                out_ptr++;
            }
            line0 += remain_w + inw;
            line1 += remain_w + inw;
        }

        if (pad_h1)
        {
            for (int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p01 = vld1q_f32(line0 + 4);
#ifdef __aarch64__
                p00 = vpaddq_f32(p00, p01);
#else
                float32x2_t sum0_1 = vpadd_f32(vget_low_f32(p00), vget_high_f32(p00));
                float32x2_t sum0_2 = vpadd_f32(vget_low_f32(p01), vget_high_f32(p01));
                p00 = vcombine_f32(sum0_1, sum0_2);
#endif
                p00 = vmulq_n_f32(p00, 0.5f);
                vst1q_f32(out_ptr, p00);
                line0 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4; j < outw; j++)
            {
                float32x2_t p1 = vld1_f32(line0);
                *out_ptr = (p1[0] + p1[1]) * 0.5f;
                out_ptr++;
                line0 += 2;
            }
            if (pad_w1)
            {
                *out_ptr = line0[0];
                out_ptr++;
            }
        }
    }
}

static void max_2x2s2(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                      int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if (pad_w1 > 0)
    {
        outw--;
    }
    if (pad_h1 > 0)
    {
        outh--;
    }
    int block_w = outw >> 2;
    int remain_w = inw - outw * 2;

    for (int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        const float* line1 = line0 + inw;
        float* out_ptr = output + c * out_hw;

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p10 = vld1q_f32(line1);

                float32x4_t p01 = vld1q_f32(line0 + 4);
                float32x4_t p11 = vld1q_f32(line1 + 4);
#ifdef __aarch64__
                float32x4_t max0 = vmaxq_f32(p00, p10);
                float32x4_t max1 = vmaxq_f32(p01, p11);
                /* pairwaise max */
                float32x4_t _max = vpmaxq_f32(max0, max1);
#else
                float32x2_t max0_1 = vpmax_f32(vget_low_f32(p00), vget_low_f32(p10));
                float32x2_t max0_2 = vpmax_f32(vget_high_f32(p00), vget_high_f32(p10));
                max0_1 = vpmax_f32(max0_1, max0_2);
                float32x2_t max1_1 = vpmax_f32(vget_low_f32(p01), vget_low_f32(p11));
                float32x2_t max1_2 = vpmax_f32(vget_high_f32(p01), vget_high_f32(p11));
                max1_1 = vpmax_f32(max1_1, max1_2);

                float32x4_t _max = vcombine_f32(max0_1, max1_1);
#endif
                vst1q_f32(out_ptr, _max);
                line0 += 8;
                line1 += 8;
                out_ptr += 4;
            }

            for (int j = block_w * 4; j < outw; j++)
            {
                float32x2_t p1 = vld1_f32(line0);
                float32x2_t p2 = vld1_f32(line1);
                float32x2_t _max = vmax_f32(p1, p2);
                *out_ptr = fmax(_max[0], _max[1]);
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }

            if (pad_w1 > 0)
            {
                *out_ptr = fmax(line0[0], line1[0]);
                out_ptr++;
            }
            line0 += remain_w + inw;
            line1 += remain_w + inw;
        }

        if (pad_h1 > 0)
        {
            for (int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p01 = vld1q_f32(line0 + 4);
#ifdef __aarch64__
                p00 = vpmaxq_f32(p00, p01);
#else
                float32x2_t max0_1 = vpmax_f32(vget_low_f32(p00), vget_high_f32(p00));
                float32x2_t max0_2 = vpmax_f32(vget_low_f32(p01), vget_high_f32(p01));
                p00 = vcombine_f32(max0_1, max0_2);
#endif
                vst1q_f32(out_ptr, p00);
                line0 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4; j < outw; j++)
            {
                float32x2_t p1 = vld1_f32(line0);
                *out_ptr = fmax(p1[0], p1[1]);
                out_ptr++;
                line0 += 2;
            }
            if (pad_w1 > 0)
            {
                *out_ptr = line0[0];
                out_ptr++;
            }
        }
    }
}

static void avg_3x3s2(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                      int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if (pad_w1 > 0)
    {
        outw--;
    }
    if (pad_h1 > 0)
    {
        outh--;
    }
    int block_w = outw >> 2;
    int remain_w = inw - outw * 2;

    for (int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        const float* line1 = line0 + inw;
        const float* line2 = line1 + inw;
        float* out_ptr = output + c * out_hw;
        for (int i = 0; i < outh; i++)
        {
            float32x4x2_t p00 = vld2q_f32(line0);
            float32x4x2_t p10 = vld2q_f32(line1);
            float32x4x2_t p20 = vld2q_f32(line2);
            for (int j = 0; j < block_w; j++)
            {
                float32x4x2_t p00_new = vld2q_f32(line0 + 8);
                float32x4_t sum0 = vaddq_f32(p00.val[0], p00.val[1]);
                float32x4_t p01 = vextq_f32(p00.val[0], p00_new.val[0], 1);
                sum0 = vaddq_f32(sum0, p01);

                float32x4x2_t p10_new = vld2q_f32(line1 + 8);
                float32x4_t sum1 = vaddq_f32(p10.val[0], p10.val[1]);
                float32x4_t p11 = vextq_f32(p10.val[0], p10_new.val[0], 1);
                sum1 = vaddq_f32(sum1, p11);

                float32x4x2_t p20_new = vld2q_f32(line2 + 8);
                float32x4_t sum2 = vaddq_f32(p20.val[0], p20.val[1]);
                float32x4_t p21 = vextq_f32(p20.val[0], p20_new.val[0], 1);
                sum2 = vaddq_f32(sum2, p21);

                sum0 = vaddq_f32(vaddq_f32(sum0, sum1), sum2);
                sum0 = vmulq_n_f32(sum0, 0.11111111f);
                vst1q_f32(out_ptr, sum0);

                p00 = p00_new;
                p10 = p10_new;
                p20 = p20_new;

                line0 += 8;
                line1 += 8;
                line2 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4; j < outw; j++)
            {
                *out_ptr = (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2] + line2[0] + line2[1] + line2[2]) * 0.11111111f;
                out_ptr++;
                line0 += 2;
                line1 += 2;
                line2 += 2;
            }
            if (pad_w1 == 1)
            {
                *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * 0.16666667f;
                out_ptr++;
            }
            line0 += remain_w + inw;
            line1 += remain_w + inw;
            line2 += remain_w + inw;
        }
        if (pad_h1 == 1)
        {
            float32x4x2_t p00 = vld2q_f32(line0);
            float32x4x2_t p10 = vld2q_f32(line1);
            for (int j = 0; j < block_w; j++)
            {
                float32x4x2_t p00_new = vld2q_f32(line0 + 8);
                float32x4_t sum0 = vaddq_f32(p00.val[0], p00.val[1]);
                float32x4_t p01 = vextq_f32(p00.val[0], p00_new.val[0], 1);
                sum0 = vaddq_f32(sum0, p01);

                float32x4x2_t p10_new = vld2q_f32(line1 + 8);
                float32x4_t sum1 = vaddq_f32(p10.val[0], p10.val[1]);
                float32x4_t p11 = vextq_f32(p10.val[0], p10_new.val[0], 1);
                sum1 = vaddq_f32(sum1, p11);

                sum0 = vaddq_f32(sum0, sum1);
                sum0 = vmulq_n_f32(sum0, 0.16666667f);
                vst1q_f32(out_ptr, sum0);

                p00 = p00_new;
                p10 = p10_new;

                line0 += 8;
                line1 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4; j < outw; j++)
            {
                *out_ptr = (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2]) * 0.16666667f;
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if (pad_w1 == 1)
            {
                *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1]) * 0.25f;
                out_ptr++;
            }
            else if (pad_w1 == 2)
            {
                *out_ptr = (line0[0] + line1[0]) * 0.5f;
                out_ptr++;
            }
        }
        else if (pad_h1 == 2)
        {
            float32x4x2_t p00 = vld2q_f32(line0);
            for (int j = 0; j < block_w; j++)
            {
                float32x4x2_t p00_new = vld2q_f32(line0 + 8);
                float32x4_t sum0 = vaddq_f32(p00.val[0], p00.val[1]);
                float32x4_t p01 = vextq_f32(p00.val[0], p00_new.val[0], 1);
                sum0 = vaddq_f32(sum0, p01);

                sum0 = vmulq_n_f32(sum0, 0.3333333f);
                vst1q_f32(out_ptr, sum0);

                p00 = p00_new;

                line0 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4; j < outw; j++)
            {
                *out_ptr = (line0[0] + line0[1] + line0[2]) * 0.3333333f;
                out_ptr++;
                line0 += 2;
            }
            if (pad_w1 == 1)
            {
                *out_ptr = (line0[0] + line0[1]) * 0.5f;
                out_ptr++;
            }
            else if (pad_w1 == 2)
            {
                *out_ptr = line0[0];
                out_ptr++;
            }
        }
    }
}

static void max_3x3s2(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                      int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)

{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if (pad_w1 > 0)
    {
        outw--;
    }
    if (pad_h1 > 0)
    {
        outh--;
    }
    int block_w = outw >> 2;
    int remain_w = inw - outw * 2;

    for (int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        const float* line1 = line0 + inw;
        const float* line2 = line1 + inw;
        float* out_ptr = output + c * out_hw;
        for (int i = 0; i < outh; i++)
        {
            float32x4x2_t p00 = vld2q_f32(line0);
            float32x4x2_t p10 = vld2q_f32(line1);
            float32x4x2_t p20 = vld2q_f32(line2);
            for (int j = 0; j < block_w; j++)
            {
                /*
                p00     = [1,2,3,4,5,6,7,8]
                p00.val[0]=[1,3,5,7]

                max0    = [2,4,6,8]
                p00_new = [9,10,11,12,13,14,15,16]
                p01     = [3,5,7,9]
                max0=max(max0,p01)=[3,5,7,9]
                */
                float32x4x2_t p00_new = vld2q_f32(line0 + 8);
                float32x4_t max0 = vmaxq_f32(p00.val[0], p00.val[1]);
                float32x4_t p01 = vextq_f32(p00.val[0], p00_new.val[0], 1);
                max0 = vmaxq_f32(max0, p01);

                float32x4x2_t p10_new = vld2q_f32(line1 + 8);
                float32x4_t max1 = vmaxq_f32(p10.val[0], p10.val[1]);
                float32x4_t p11 = vextq_f32(p10.val[0], p10_new.val[0], 1);
                max1 = vmaxq_f32(max1, p11);

                float32x4x2_t p20_new = vld2q_f32(line2 + 8);
                float32x4_t max2 = vmaxq_f32(p20.val[0], p20.val[1]);
                float32x4_t p21 = vextq_f32(p20.val[0], p20_new.val[0], 1);
                max2 = vmaxq_f32(max2, p21);

                max0 = vmaxq_f32(vmaxq_f32(max0, max1), max2);
                vst1q_f32(out_ptr, max0);

                p00 = p00_new;
                p10 = p10_new;
                p20 = p20_new;

                line0 += 8;
                line1 += 8;
                line2 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4; j < outw; j++)
            {
                float max0 = fmax(fmax(line0[0], line0[1]), line0[2]);
                float max1 = fmax(fmax(line1[0], line1[1]), line1[2]);
                float max2 = fmax(fmax(line2[0], line2[1]), line2[2]);
                *out_ptr = fmax(fmax(max0, max1), max2);

                out_ptr++;
                line0 += 2;
                line1 += 2;
                line2 += 2;
            }
            if (pad_w1 == 1)
            {
                float max0 = fmax(fmax(line0[0], line0[1]), fmax(line1[0], line1[1]));
                *out_ptr = fmax(fmax(line2[0], line2[1]), max0);
                out_ptr++;
            }
            line0 += remain_w + inw;
            line1 += remain_w + inw;
            line2 += remain_w + inw;
        }
        if (pad_h1 == 1)
        {
            float32x4x2_t p00 = vld2q_f32(line0);
            float32x4x2_t p10 = vld2q_f32(line1);
            for (int j = 0; j < block_w; j++)
            {
                float32x4x2_t p00_new = vld2q_f32(line0 + 8);
                float32x4_t max0 = vmaxq_f32(p00.val[0], p00.val[1]);
                float32x4_t p01 = vextq_f32(p00.val[0], p00_new.val[0], 1);
                max0 = vmaxq_f32(max0, p01);

                float32x4x2_t p10_new = vld2q_f32(line1 + 8);
                float32x4_t max1 = vmaxq_f32(p10.val[0], p10.val[1]);
                float32x4_t p11 = vextq_f32(p10.val[0], p10_new.val[0], 1);
                max1 = vmaxq_f32(max1, p11);

                vst1q_f32(out_ptr, vmaxq_f32(max0, max1));

                p00 = p00_new;
                p10 = p10_new;

                line0 += 8;
                line1 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4; j < outw; j++)
            {
                float max0 = fmax(fmax(line0[0], line0[1]), line0[2]);
                float max1 = fmax(fmax(line1[0], line1[1]), line1[2]);

                *out_ptr = fmax(max0, max1);

                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if (pad_w1 == 1)
            {
                *out_ptr = fmax(fmax(line0[0], line0[1]), fmax(line1[0], line1[1]));
                out_ptr++;
            }
        }
    }
}

static void avg_2x2s2_p1(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                         int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if (inw % 2 == 0)
        outw--;
    if (inh % 2 == 0)
        outh--;
    int block_w = (outw - 1) >> 2;
    int remain_w = inw - outw * 2 + 1;

    for (int c = 0; c < inc; c++)
    {
        const float* line00 = input + c * in_hw;
        float* out_ptr = output + c * out_hw;
        // h begin
        if (is_caffe == 0)
            *out_ptr = line00[0];
        else
            *out_ptr = line00[0] * 0.25f;
        out_ptr++;
        line00++;
        for (int j = 0; j < block_w; j++)
        {
            float32x4_t p00 = vld1q_f32(line00);
            float32x4_t p01 = vld1q_f32(line00 + 4);
#ifdef __aarch64__
            float32x4_t sum0 = vpaddq_f32(p00, p01);
#else
            float32x2_t sum0_1 = vpadd_f32(vget_low_f32(p00), vget_high_f32(p00));
            float32x2_t sum0_2 = vpadd_f32(vget_low_f32(p01), vget_high_f32(p01));
            float32x4_t sum0 = vcombine_f32(sum0_1, sum0_2);
#endif

            if (is_caffe == 0)
                sum0 = vmulq_n_f32(sum0, 0.5f);
            else
                sum0 = vmulq_n_f32(sum0, 0.25f);
            vst1q_f32(out_ptr, sum0);
            line00 += 8;
            out_ptr += 4;
        }
        for (int j = block_w * 4 + 1; j < outw; j++)
        {
            if (is_caffe == 0)
                *out_ptr = (line00[0] + line00[1]) * 0.5f;
            else
                *out_ptr = (line00[0] + line00[1]) * 0.25f;
            out_ptr++;
            line00 += 2;
        }
        if (inw % 2 == 0)
        {
            if (is_caffe == 0)
                *out_ptr = line00[0];
            else
                *out_ptr = line00[0] * 0.25f;
            out_ptr++;
        }
        line00 += remain_w;

        // h center
        const float* line0 = line00;
        const float* line1 = line0 + inw;
        for (int i = 1; i < outh; i++)
        {
            // w begin
            if (is_caffe == 0)
                *out_ptr = (line0[0] + line1[0]) * 0.5f;
            else
                *out_ptr = (line0[0] + line1[0]) * 0.25f;
            out_ptr++;
            line0++;
            line1++;
            // w center
            for (int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p10 = vld1q_f32(line1);
                float32x4_t sum0 = vaddq_f32(p00, p10);

                float32x4_t p01 = vld1q_f32(line0 + 4);
                float32x4_t p11 = vld1q_f32(line1 + 4);
                float32x4_t sum1 = vaddq_f32(p01, p11);

#ifdef __aarch64__
                float32x4_t _sum = vpaddq_f32(sum0, sum1);
#else
                float32x2_t sum0_1 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                float32x2_t sum0_2 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                float32x4_t _sum = vcombine_f32(sum0_1, sum0_2);
#endif
                _sum = vmulq_n_f32(_sum, 0.25f);
                vst1q_f32(out_ptr, _sum);

                out_ptr += 4;
                line0 += 8;
                line1 += 8;
            }
            for (int j = block_w * 4 + 1; j < outw; j++)
            {
                *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1]) * 0.25f;
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            // w end
            if (inw % 2 == 0)
            {
                if (is_caffe == 0)
                    *out_ptr = (line0[0] + line1[0]) * 0.5f;
                else
                    *out_ptr = (line0[0] + line1[0]) * 0.25f;
                out_ptr++;
            }
            line0 += remain_w + inw;
            line1 += remain_w + inw;
        }

        // h end
        if (inh % 2 == 0)
        {
            if (is_caffe == 0)
                *out_ptr = line0[0];
            else
                *out_ptr = line0[0] * 0.25f;
            out_ptr++;
            line0++;
            for (int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p01 = vld1q_f32(line0 + 4);
#ifdef __aarch64__
                float32x4_t _sum = vpaddq_f32(p00, p01);
#else
                float32x2_t sum0_1 = vpadd_f32(vget_low_f32(p00), vget_high_f32(p00));
                float32x2_t sum0_2 = vpadd_f32(vget_low_f32(p01), vget_high_f32(p01));
                float32x4_t _sum = vcombine_f32(sum0_1, sum0_2);
#endif
                if (is_caffe == 0)
                    _sum = vmulq_n_f32(_sum, 0.5f);
                else
                    _sum = vmulq_n_f32(_sum, 0.25f);
                vst1q_f32(out_ptr, _sum);

                out_ptr += 4;
                line0 += 8;
            }
            for (int j = block_w * 4 + 1; j < outw; j++)
            {
                if (is_caffe == 0)
                    *out_ptr = (line0[0] + line0[1]) * 0.5f;
                else
                    *out_ptr = (line0[0] + line0[1]) * 0.25f;
                out_ptr++;
                line0 += 2;
            }
            if (inw % 2 == 0)
            {
                if (is_caffe == 0)
                    *out_ptr = line0[0];
                else
                    *out_ptr = line0[0] * 0.25f;
            }
        }
    }
}

static void max_2x2s2_p1(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                         int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if (inw % 2 == 0)
        outw--;
    if (inh % 2 == 0)
        outh--;
    int block_w = (outw - 1) >> 2;
    int remain_w = inw - outw * 2 + 1;

    for (int c = 0; c < inc; c++)
    {
        const float* line00 = input + c * in_hw;
        float* out_ptr = output + c * out_hw;
        // h begin
        *out_ptr = line00[0];
        out_ptr++;
        line00++;
        for (int j = 0; j < block_w; j++)
        {
            float32x4_t p00 = vld1q_f32(line00);
            float32x4_t p01 = vld1q_f32(line00 + 4);
#ifdef __aarch64__
            float32x4_t _max = vpmaxq_f32(p00, p01);
#else
            float32x2_t max0_1 = vpmax_f32(vget_low_f32(p00), vget_high_f32(p00));
            float32x2_t max0_2 = vpmax_f32(vget_low_f32(p01), vget_high_f32(p01));
            float32x4_t _max = vcombine_f32(max0_1, max0_2);
#endif
            vst1q_f32(out_ptr, _max);

            out_ptr += 4;
            line00 += 8;
        }
        for (int j = block_w * 4 + 1; j < outw; j++)
        {
            *out_ptr = fmax(line00[0], line00[1]);
            out_ptr++;
            line00 += 2;
        }
        if (inw % 2 == 0)
        {
            *out_ptr = line00[0];
            out_ptr++;
        }
        line00 += remain_w;

        // h center
        const float* line0 = line00;
        const float* line1 = line0 + inw;
        for (int i = 1; i < outh; i++)
        {
            // w begin
            *out_ptr = fmax(line0[0], line1[0]);
            out_ptr++;
            line0++;
            line1++;
            // w center
            for (int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p10 = vld1q_f32(line1);

                float32x4_t p01 = vld1q_f32(line0 + 4);
                float32x4_t p11 = vld1q_f32(line1 + 4);

#ifdef __aarch64__
                float32x4_t max0 = vmaxq_f32(p00, p10);
                float32x4_t max1 = vmaxq_f32(p01, p11);
                float32x4_t _max = vpmaxq_f32(max0, max1);
#else
                float32x2_t max0_1 = vpmax_f32(vget_low_f32(p00), vget_low_f32(p10));
                float32x2_t max0_2 = vpmax_f32(vget_high_f32(p00), vget_high_f32(p10));
                max0_1 = vpmax_f32(max0_1, max0_2);
                float32x2_t max1_1 = vpmax_f32(vget_low_f32(p01), vget_low_f32(p11));
                float32x2_t max1_2 = vpmax_f32(vget_high_f32(p01), vget_high_f32(p11));
                max1_1 = vpmax_f32(max1_1, max1_2);

                float32x4_t _max = vcombine_f32(max0_1, max1_1);
#endif

                vst1q_f32(out_ptr, _max);

                out_ptr += 4;
                line0 += 8;
                line1 += 8;
            }
            for (int j = block_w * 4 + 1; j < outw; j++)
            {
                float32x2_t p1 = vld1_f32(line0);
                float32x2_t p2 = vld1_f32(line1);
                float32x2_t _max = vmax_f32(p1, p2);
                *out_ptr = fmax(_max[0], _max[1]);
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            // w end
            if (inw % 2 == 0)
            {
                *out_ptr = fmax(line0[0], line1[0]);
                out_ptr++;
            }
            line0 += remain_w + inw;
            line1 += remain_w + inw;
        }
        // h end
        if (inh % 2 == 0)
        {
            *out_ptr = line0[0];
            out_ptr++;
            line0++;
            for (int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p01 = vld1q_f32(line0 + 4);
#ifdef __aarch64__
                float32x4_t _max = vpmaxq_f32(p00, p01);
#else
                float32x2_t max0_1 = vpmax_f32(vget_low_f32(p00), vget_high_f32(p00));
                float32x2_t max0_2 = vpmax_f32(vget_low_f32(p01), vget_high_f32(p01));
                float32x4_t _max = vcombine_f32(max0_1, max0_2);
#endif
                vst1q_f32(out_ptr, _max);

                out_ptr += 4;
                line0 += 8;
            }
            for (int j = block_w * 4 + 1; j < outw; j++)
            {
                *out_ptr = fmax(line0[0], line0[1]);
                out_ptr++;
                line0 += 2;
            }
            if (inw % 2 == 0)
            {
                *out_ptr = line0[0];
            }
        }
    }
}

static void max_3x3s2_p1(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                         int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)

{
    // TLOG_ERR("max_3x3s2_p1\n");
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if (is_caffe == 1 || inw % 2 == 1)
        outw--;
    if (is_caffe == 1 || inh % 2 == 1)
        outh--;
    int block_w = (outw - 1) >> 2;
    int remain_w = inw - outw * 2 + 1;

    for (int c = 0; c < inc; c++)
    {
        const float* line1 = input + c * in_hw;
        const float* line2 = line1 + inw;
        float* out_ptr = output + c * out_hw;

        // h begin ---------------------------------------
        *out_ptr = fmax(fmax(line1[0], line1[1]), fmax(line2[0], line2[1]));
        out_ptr++;
        line1 += 1;
        line2 += 1;

        float32x4x2_t p10 = vld2q_f32(line1);
        float32x4x2_t p20 = vld2q_f32(line2);
        for (int j = 0; j < block_w; j++)
        {
            float32x4x2_t p10_new = vld2q_f32(line1 + 8);
            float32x4_t max1 = vmaxq_f32(p10.val[0], p10.val[1]);
            float32x4_t p11 = vextq_f32(p10.val[0], p10_new.val[0], 1);
            max1 = vmaxq_f32(max1, p11);

            float32x4x2_t p20_new = vld2q_f32(line2 + 8);
            float32x4_t max2 = vmaxq_f32(p20.val[0], p20.val[1]);
            float32x4_t p21 = vextq_f32(p20.val[0], p20_new.val[0], 1);
            max2 = vmaxq_f32(max2, p21);

            max1 = vmaxq_f32(max1, max2);
            vst1q_f32(out_ptr, max1);

            p10 = p10_new;
            p20 = p20_new;

            line1 += 8;
            line2 += 8;
            out_ptr += 4;
        }
        for (int j = block_w * 4 + 1; j < outw; j++)
        {
            float max1 = fmax(fmax(line1[0], line1[1]), line1[2]);
            float max2 = fmax(fmax(line2[0], line2[1]), line2[2]);
            *out_ptr = fmax(max1, max2);

            out_ptr++;
            line1 += 2;
            line2 += 2;
        }
        if (inw % 2 == 1)
        {
            *out_ptr = fmax(fmax(line1[0], line1[1]), fmax(line2[0], line2[1]));
            out_ptr++;
        }
        else if (is_caffe == 1 && inw % 2 == 0)
        {
            *out_ptr = fmax(line1[0], line2[0]);
            out_ptr++;
        }
        line1 += remain_w;
        line2 += remain_w;

        // h center ---------------------------------------
        const float* line0 = line1;
        line1 = line2;
        line2 = line1 + inw;
        for (int i = 1; i < outh; i++)
        {
            // left
            float max0 = fmax(fmax(line1[0], line1[1]), fmax(line2[0], line2[1]));
            *out_ptr = fmax(fmax(line0[0], line0[1]), max0);
            out_ptr++;
            line0 += 1;
            line1 += 1;
            line2 += 1;
            // mid
            float32x4x2_t p00 = vld2q_f32(line0);
            float32x4x2_t p10 = vld2q_f32(line1);
            float32x4x2_t p20 = vld2q_f32(line2);
            for (int j = 0; j < block_w; j++)
            {
                float32x4x2_t p00_new = vld2q_f32(line0 + 8);
                float32x4_t max0 = vmaxq_f32(p00.val[0], p00.val[1]);
                float32x4_t p01 = vextq_f32(p00.val[0], p00_new.val[0], 1);
                max0 = vmaxq_f32(max0, p01);

                float32x4x2_t p10_new = vld2q_f32(line1 + 8);
                float32x4_t max1 = vmaxq_f32(p10.val[0], p10.val[1]);
                float32x4_t p11 = vextq_f32(p10.val[0], p10_new.val[0], 1);
                max1 = vmaxq_f32(max1, p11);

                float32x4x2_t p20_new = vld2q_f32(line2 + 8);
                float32x4_t max2 = vmaxq_f32(p20.val[0], p20.val[1]);
                float32x4_t p21 = vextq_f32(p20.val[0], p20_new.val[0], 1);
                max2 = vmaxq_f32(max2, p21);

                max0 = vmaxq_f32(vmaxq_f32(max0, max1), max2);
                vst1q_f32(out_ptr, max0);

                p00 = p00_new;
                p10 = p10_new;
                p20 = p20_new;

                line0 += 8;
                line1 += 8;
                line2 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4 + 1; j < outw; j++)
            {
                float max0 = fmax(fmax(line0[0], line0[1]), line0[2]);
                float max1 = fmax(fmax(line1[0], line1[1]), line1[2]);
                float max2 = fmax(fmax(line2[0], line2[1]), line2[2]);
                *out_ptr = fmax(fmax(max0, max1), max2);

                out_ptr++;
                line0 += 2;
                line1 += 2;
                line2 += 2;
            }
            if (inw % 2 == 1)
            {
                max0 = fmax(fmax(line1[0], line1[1]), fmax(line2[0], line2[1]));
                *out_ptr = fmax(fmax(line0[0], line0[1]), max0);
                out_ptr++;
            }
            else if (inw % 2 == 0 && is_caffe == 1)
            {
                *out_ptr = fmax(fmax(line0[0], line1[0]), line2[0]);
                out_ptr++;
            }
            line0 += inw + remain_w;
            line1 += inw + remain_w;
            line2 += inw + remain_w;
        }

        // h end ------------------------------------------
        if (inh % 2 == 1)
        {
            *out_ptr = fmax(fmax(line1[0], line1[1]), fmax(line0[0], line0[1]));
            out_ptr++;
            line0 += 1;
            line1 += 1;

            float32x4x2_t p00 = vld2q_f32(line0);
            float32x4x2_t p10 = vld2q_f32(line1);
            for (int j = 0; j < block_w; j++)
            {
                float32x4x2_t p00_new = vld2q_f32(line0 + 8);
                float32x4_t max0 = vmaxq_f32(p00.val[0], p00.val[1]);
                float32x4_t p01 = vextq_f32(p00.val[0], p00_new.val[0], 1);
                max0 = vmaxq_f32(max0, p01);

                float32x4x2_t p10_new = vld2q_f32(line1 + 8);
                float32x4_t max1 = vmaxq_f32(p10.val[0], p10.val[1]);
                float32x4_t p11 = vextq_f32(p10.val[0], p10_new.val[0], 1);
                max1 = vmaxq_f32(max1, p11);

                max0 = vmaxq_f32(max0, max1);
                vst1q_f32(out_ptr, max0);

                p00 = p00_new;
                p10 = p10_new;

                line0 += 8;
                line1 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4 + 1; j < outw; j++)
            {
                float max0 = fmax(fmax(line0[0], line0[1]), line0[2]);
                float max1 = fmax(fmax(line1[0], line1[1]), line1[2]);
                *out_ptr = fmax(max0, max1);

                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if (inw % 2 == 1)
            {
                *out_ptr = fmax(fmax(line1[0], line1[1]), fmax(line0[0], line0[1]));
                out_ptr++;
            }
            else if (inw % 2 == 0 && is_caffe == 1)
            {
                *out_ptr = fmax(line0[0], line1[0]);
                out_ptr++;
            }
        }
        else if (inh % 2 == 0 && is_caffe == 1)
        {
            *out_ptr = fmax(line0[0], line0[1]);
            out_ptr++;
            line0 += 1;

            float32x4x2_t p00 = vld2q_f32(line0);
            for (int j = 0; j < block_w; j++)
            {
                float32x4x2_t p00_new = vld2q_f32(line0 + 8);
                float32x4_t max0 = vmaxq_f32(p00.val[0], p00.val[1]);
                float32x4_t p01 = vextq_f32(p00.val[0], p00_new.val[0], 1);
                max0 = vmaxq_f32(max0, p01);

                vst1q_f32(out_ptr, max0);

                p00 = p00_new;

                line0 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4 + 1; j < outw; j++)
            {
                *out_ptr = fmax(fmax(line0[0], line0[1]), line0[2]);

                out_ptr++;
                line0 += 2;
            }
            if (inw % 2 == 1)
            {
                *out_ptr = fmax(line0[0], line0[1]);
                out_ptr++;
            }
            else if (inw % 2 == 0)
            {
                *out_ptr = line0[0];
            }
        }
    }
}

static void avg_3x3s2_p1(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                         int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if (is_caffe == 1 || inw % 2 == 1)
        outw--;
    if (is_caffe == 1 || inh % 2 == 1)
        outh--;
    int block_w = (outw - 1) >> 2;
    int remain_w = inw - outw * 2 + 1;

    for (int c = 0; c < inc; c++)
    {
        const float* line1 = input + c * in_hw;
        const float* line2 = line1 + inw;
        float* out_ptr = output + c * out_hw;

        // h begin ---------------------------------------
        if (is_caffe == 0)
            *out_ptr = (line1[0] + line1[1] + line2[0] + line2[1]) * 0.25f;
        else
            *out_ptr = (line1[0] + line1[1] + line2[0] + line2[1]) * 0.11111111f;
        out_ptr++;
        line1 += 1;
        line2 += 1;

        float32x4x2_t p10 = vld2q_f32(line1);
        float32x4x2_t p20 = vld2q_f32(line2);
        for (int j = 0; j < block_w; j++)
        {
            float32x4x2_t p10_new = vld2q_f32(line1 + 8);
            float32x4_t sum1 = vaddq_f32(p10.val[0], p10.val[1]);
            float32x4_t p11 = vextq_f32(p10.val[0], p10_new.val[0], 1);
            sum1 = vaddq_f32(sum1, p11);

            float32x4x2_t p20_new = vld2q_f32(line2 + 8);
            float32x4_t sum2 = vaddq_f32(p20.val[0], p20.val[1]);
            float32x4_t p21 = vextq_f32(p20.val[0], p20_new.val[0], 1);
            sum2 = vaddq_f32(sum2, p21);

            sum1 = vaddq_f32(sum1, sum2);
            if (is_caffe == 0)
                sum1 = vmulq_n_f32(sum1, 0.16666667f);
            else
                sum1 = vmulq_n_f32(sum1, 0.11111111f);
            vst1q_f32(out_ptr, sum1);

            p10 = p10_new;
            p20 = p20_new;

            line1 += 8;
            line2 += 8;
            out_ptr += 4;
        }
        for (int j = block_w * 4 + 1; j < outw; j++)
        {
            if (is_caffe == 0)
                *out_ptr = (line1[0] + line1[1] + line1[2] + line2[0] + line2[1] + line2[2]) * 0.16666667f;
            else
                *out_ptr = (line1[0] + line1[1] + line1[2] + line2[0] + line2[1] + line2[2]) * 0.11111111f;
            out_ptr++;
            line1 += 2;
            line2 += 2;
        }
        if (inw % 2 == 1)
        {
            if (is_caffe == 0)
                *out_ptr = (line1[0] + line1[1] + line2[0] + line2[1]) * 0.25f;
            else
                *out_ptr = (line1[0] + line1[1] + line2[0] + line2[1]) * 0.11111111f;
            out_ptr++;
        }
        else if (inw % 2 == 0 && is_caffe == 1)
        {
            *out_ptr = (line1[0] + line2[0]) * 0.16666667f;
            out_ptr++;
        }
        line1 += remain_w;
        line2 += remain_w;

        // h center ---------------------------------------
        const float* line0 = line1;
        line1 = line2;
        line2 = line1 + inw;
        for (int i = 1; i < outh; i++)
        {
            // left
            if (is_caffe == 0)
                *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * 0.16666667f;
            else
                *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * 0.11111111f;
            out_ptr++;
            line0 += 1;
            line1 += 1;
            line2 += 1;
            // mid
            float32x4x2_t p00 = vld2q_f32(line0);
            float32x4x2_t p10 = vld2q_f32(line1);
            float32x4x2_t p20 = vld2q_f32(line2);
            for (int j = 0; j < block_w; j++)
            {
                float32x4x2_t p00_new = vld2q_f32(line0 + 8);
                float32x4_t sum0 = vaddq_f32(p00.val[0], p00.val[1]);
                float32x4_t p01 = vextq_f32(p00.val[0], p00_new.val[0], 1);
                sum0 = vaddq_f32(sum0, p01);

                float32x4x2_t p10_new = vld2q_f32(line1 + 8);
                float32x4_t sum1 = vaddq_f32(p10.val[0], p10.val[1]);
                float32x4_t p11 = vextq_f32(p10.val[0], p10_new.val[0], 1);
                sum1 = vaddq_f32(sum1, p11);

                float32x4x2_t p20_new = vld2q_f32(line2 + 8);
                float32x4_t sum2 = vaddq_f32(p20.val[0], p20.val[1]);
                float32x4_t p21 = vextq_f32(p20.val[0], p20_new.val[0], 1);
                sum2 = vaddq_f32(sum2, p21);

                sum0 = vaddq_f32(vaddq_f32(sum0, sum1), sum2);
                sum0 = vmulq_n_f32(sum0, 0.11111111f);
                vst1q_f32(out_ptr, sum0);

                p00 = p00_new;
                p10 = p10_new;
                p20 = p20_new;

                line0 += 8;
                line1 += 8;
                line2 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4 + 1; j < outw; j++)
            {
                *out_ptr = (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2] + line2[0] + line2[1] + line2[2]) * 0.11111111f;
                out_ptr++;
                line0 += 2;
                line1 += 2;
                line2 += 2;
            }
            // end
            if (inw % 2 == 1)
            {
                if (is_caffe == 0)
                    *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * 0.16666667f;
                else
                    *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * 0.11111111f;
                out_ptr++;
            }
            else if (inw % 2 == 0 && is_caffe == 1)
            {
                *out_ptr = (line0[0] + line1[0] + line2[0]) * 0.16666667f;
                out_ptr++;
            }
            line0 += remain_w + inw;
            line1 += remain_w + inw;
            line2 += remain_w + inw;
        }
        // h  end-------------------------------
        if (inh % 2 == 1)
        {
            if (is_caffe == 0)
                *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1]) * 0.25f;
            else
                *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1]) * 0.11111111f;
            out_ptr++;
            line0 += 1;
            line1 += 1;

            float32x4x2_t p00 = vld2q_f32(line0);
            float32x4x2_t p10 = vld2q_f32(line1);
            for (int j = 0; j < block_w; j++)
            {
                float32x4x2_t p00_new = vld2q_f32(line0 + 8);
                float32x4_t sum0 = vaddq_f32(p00.val[0], p00.val[1]);
                float32x4_t p01 = vextq_f32(p00.val[0], p00_new.val[0], 1);
                sum0 = vaddq_f32(sum0, p01);

                float32x4x2_t p10_new = vld2q_f32(line1 + 8);
                float32x4_t sum1 = vaddq_f32(p10.val[0], p10.val[1]);
                float32x4_t p11 = vextq_f32(p10.val[0], p10_new.val[0], 1);
                sum1 = vaddq_f32(sum1, p11);

                sum0 = vaddq_f32(sum0, sum1);
                if (is_caffe == 0)
                    sum0 = vmulq_n_f32(sum0, 0.16666667f);
                else
                    sum0 = vmulq_n_f32(sum0, 0.11111111f);
                vst1q_f32(out_ptr, sum0);

                p00 = p00_new;
                p10 = p10_new;

                line0 += 8;
                line1 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4 + 1; j < outw; j++)
            {
                if (is_caffe == 0)
                    *out_ptr = (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2]) * 0.16666667f;
                else
                    *out_ptr = (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2]) * 0.11111111f;
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if (inw % 2 == 1)
            {
                if (is_caffe == 0)
                    *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1]) * 0.25f;
                else
                    *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1]) * 0.11111111f;
                out_ptr++;
            }
            else if (inw % 2 == 0 && is_caffe == 1)
            {
                *out_ptr = (line0[0] + line1[0]) * 0.16666667f;
                out_ptr++;
            }
        }
        else if (inw % 2 == 0 && is_caffe == 1)
        {
            *out_ptr = (line0[0] + line0[1]) * 0.16666667f;
            out_ptr++;
            line0 += 1;

            float32x4x2_t p00 = vld2q_f32(line0);
            for (int j = 0; j < block_w; j++)
            {
                float32x4x2_t p00_new = vld2q_f32(line0 + 8);
                float32x4_t sum0 = vaddq_f32(p00.val[0], p00.val[1]);
                float32x4_t p01 = vextq_f32(p00.val[0], p00_new.val[0], 1);
                sum0 = vaddq_f32(sum0, p01);

                sum0 = vmulq_n_f32(sum0, 0.16666667f);
                vst1q_f32(out_ptr, sum0);

                p00 = p00_new;

                line0 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4 + 1; j < outw; j++)
            {
                *out_ptr = (line0[0] + line0[1] + line0[2]) * 0.16666667f;
                out_ptr++;
                line0 += 2;
            }
            if (inw % 2 == 1)
            {
                *out_ptr = (line0[0] + line0[1]) * 0.16666667f;
                out_ptr++;
            }
            else if (inw % 2 == 0)
            {
                *out_ptr = line0[0] * 0.25f;
                out_ptr++;
            }
        }
    }
}

static void max_3x3s1_p1(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                         int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    //     TLOG_ERR("max_3x3s1_p1\n");
    int in_hw = inw * inh;

    int mid_w = inw - 2;
    int mid_h = inh - 2;

    for (int c = 0; c < inc; c++)
    {
        const float* line1 = input + c * in_hw;
        const float* line2 = line1 + inw;

        float* out_ptr = output + c * in_hw;

        // h begin left----[line1+=0]-----------------------------------
        *out_ptr = fmax(fmax(line1[0], line1[1]), fmax(line2[0], line2[1]));
        out_ptr++;

        // h begin center----[line1+=1]----------------------------------
        for (int j = 0; j < mid_w; j++)
        {
            float max1 = fmax(fmax(line1[0], line1[1]), line1[2]);
            float max2 = fmax(fmax(line2[0], line2[1]), line2[2]);
            *out_ptr = fmax(max2, max1);
            out_ptr++;
            line1 += 1;
            line2 += 1;
        }
        // h begin right----[line1+=2]-----------------------------------
        *out_ptr = fmax(fmax(line1[0], line1[1]), fmax(line2[0], line2[1]));
        out_ptr++;
        line1 += 2;
        line2 += 2;

        // h center ---------------------------------------
        const float* line0 = input + c * in_hw;

        for (int i = 0; i < mid_h; i++)
        {
            // left
            float max0 = fmax(fmax(line1[0], line1[1]), fmax(line2[0], line2[1]));
            *out_ptr = fmax(fmax(line0[0], line0[1]), max0);
            out_ptr++;

            // mid
            for (int j = 0; j < mid_w; j++)
            {
                float max0 = fmax(fmax(line0[0], line0[1]), line0[2]);
                float max1 = fmax(fmax(line1[0], line1[1]), line1[2]);
                float max2 = fmax(fmax(line2[0], line2[1]), line2[2]);
                *out_ptr = fmax(fmax(max0, max1), max2);
                out_ptr++;
                line0 += 1;
                line1 += 1;
                line2 += 1;
            }
            max0 = fmax(fmax(line1[0], line1[1]), fmax(line2[0], line2[1]));
            *out_ptr = fmax(fmax(line0[0], line0[1]), max0);
            out_ptr++;
            line0 += 2;
            line1 += 2;
            line2 += 2;
        }

        // h end ------------------------------------------
        *out_ptr = fmax(fmax(line1[0], line1[1]), fmax(line0[0], line0[1]));
        out_ptr++;

        for (int j = 0; j < mid_w; j++)
        {
            float max0 = fmax(fmax(line0[0], line0[1]), line0[2]);
            float max1 = fmax(fmax(line1[0], line1[1]), line1[2]);

            *out_ptr = fmax(max0, max1);
            out_ptr++;
            line0 += 1;
            line1 += 1;
        }

        *out_ptr = fmax(fmax(line1[0], line1[1]), fmax(line0[0], line0[1]));
    }
}

static void avg_3x3s1_p1(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                         int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    //     TLOG_ERR("avg_3x3s1_p1\n");
    int in_hw = inw * inh;

    int mid_w = inw - 2;
    int mid_h = inh - 2;

    for (int c = 0; c < inc; c++)
    {
        const float* line1 = input + c * in_hw;
        const float* line2 = line1 + inw;

        float* out_ptr = output + c * in_hw;

        // h begin left----[line1+=0]-----------------------------------
        if (is_caffe == 0)
            *out_ptr = (line1[0] + line1[1] + line2[0] + line2[1]) * 0.25f;
        else
            *out_ptr = (line1[0] + line1[1] + line2[0] + line2[1]) * 0.11111111f;
        out_ptr++;

        // h begin center----[line1+=1]----------------------------------
        for (int j = 0; j < mid_w; j++)
        {
            if (is_caffe == 0)
                *out_ptr = (line1[0] + line1[1] + line1[2] + line2[0] + line2[1] + line2[2]) * 0.16666667f;
            else
                *out_ptr = (line1[0] + line1[1] + line1[2] + line2[0] + line2[1] + line2[2]) * 0.11111111f;
            out_ptr++;
            line1 += 1;
            line2 += 1;
        }
        // h begin right----[line1+=2]-----------------------------------
        if (is_caffe == 0)
            *out_ptr = (line1[0] + line1[1] + line2[0] + line2[1]) * 0.25f;
        else
            *out_ptr = (line1[0] + line1[1] + line2[0] + line2[1]) * 0.11111111f;
        out_ptr++;
        line1 += 2;
        line2 += 2;

        // h center ---------------------------------------
        const float* line0 = input + c * in_hw;

        for (int i = 0; i < mid_h; i++)
        {
            // left
            if (is_caffe == 0)
                *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * 0.16666667f;
            else
                *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * 0.11111111f;
            out_ptr++;

            // mid
            for (int j = 0; j < mid_w; j++)
            {
                *out_ptr = (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2] + line2[0] + line2[1] + line2[2]) * 0.11111111f;
                out_ptr++;
                line0 += 1;
                line1 += 1;
                line2 += 1;
            }
            if (is_caffe == 0)
                *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * 0.16666667f;
            else
                *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * 0.11111111f;
            out_ptr++;
            line0 += 2;
            line1 += 2;
            line2 += 2;
        }

        // h end ------------------------------------------
        if (is_caffe == 0)
            *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1]) * 0.25f;
        else
            *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1]) * 0.11111111f;
        out_ptr++;

        for (int j = 0; j < mid_w; j++)
        {
            if (is_caffe == 0)
                *out_ptr = (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2]) * 0.16666667f;
            else
                *out_ptr = (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2]) * 0.11111111f;
            out_ptr++;
            line0 += 1;
            line1 += 1;
        }

        if (is_caffe == 0)
            *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1]) * 0.25f;
        else
            *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1]) * 0.11111111f;
    }
}

static void avg_global(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                       int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    int in_hw = inw * inh;
    int block = in_hw >> 3;
    int tail = in_hw & ~7;

    for (int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        float* out_ptr = output + c;
        float sum = 0.f;
        for (int j = 0; j < block; j++)
        {
            float32x4_t p00 = vld1q_f32(line0);
            float32x4_t p01 = vld1q_f32(line0 + 4);
            p00 = vaddq_f32(p00, p01);
            // p00=vpaddq_f32(p00,p00);
            // sum+=(p00[0]+p00[1]);
            sum += (p00[0] + p00[1] + p00[2] + p00[3]);
            line0 += 8;
        }
        for (int j = tail; j < in_hw; j++)
        {
            sum += line0[0];
            line0++;
        }
        *out_ptr = sum / in_hw;
    }
}

static void max_global(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                       int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    int in_hw = inw * inh;
    int block = in_hw >> 3;
    int tail = in_hw & ~7;

    for (int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        float* out_ptr = output + c;
        float32x4_t p00 = vld1q_f32(line0);
        float32x4_t res = p00;
        for (int j = 0; j < block; j++)
        {
            float32x4_t p00 = vld1q_f32(line0);
            float32x4_t p01 = vld1q_f32(line0 + 4);
            float32x4_t max0 = vmaxq_f32(p00, p01);
            res = vmaxq_f32(res, max0);
            line0 += 8;
        }
        float max_ = fmax(fmax(res[0], res[1]), fmax(res[2], res[3]));
        for (int j = tail; j < in_hw; j++)
        {
            max_ = fmax(max_, line0[0]);
            line0++;
        }
        *out_ptr = max_;
    }
}

int pooling_kernel_perf_prerun(struct tensor* input, struct tensor* out, struct pool_param* param)
{
    int pool_size = POOL_GENERIC;

    /* global pooling */
    if (param->global)
    {
        if (param->pool_method == POOL_AVG)
            param->funct = (pooling_kernel_t)avg_global;
        else if (param->pool_method == POOL_MAX)
            param->funct = (pooling_kernel_t)max_global;

        assert(param->funct != NULL);
        return 0;
    }

    /* general pooling */
    if (param->stride_h == 2 && param->stride_w == 2)
    {
        if (param->kernel_h == 2 && param->kernel_w == 2)
            pool_size = POOL_K2S2;
        else if (param->kernel_h == 3 && param->kernel_w == 3)
            pool_size = POOL_K3S2;
    }
    else if (param->stride_h == 1 && param->stride_w == 1)
    {
        if (param->kernel_h == 3 && param->kernel_w == 3)
            pool_size = POOL_K3S1;
    }

    /* general max pooling, k2s2, k2k2p1, k3s1p1, k3s2, k3s2p1 */
    if (param->pool_method == POOL_MAX)
    {
        if ((param->pad_h0 == param->pad_w0) && (param->pad_h1 == param->pad_w1))
        {
            if (param->pad_h0 == 0)
            {
                if (pool_size == POOL_K2S2)
                    param->funct = (pooling_kernel_t)max_2x2s2;
                else if (pool_size == POOL_K3S2)
                    param->funct = (pooling_kernel_t)max_3x3s2;
            }
            else if (param->pad_h0 == 1)
            {
                if (pool_size == POOL_K2S2)
                    param->funct = (pooling_kernel_t)max_2x2s2_p1;
                else if (pool_size == POOL_K3S2)
                    param->funct = (pooling_kernel_t)max_3x3s2_p1;
                else if (pool_size == POOL_K3S1)
                    param->funct = (pooling_kernel_t)max_3x3s1_p1;
            }
        }

        if (param->funct != NULL)
            return 0;
        else
        {
            TLOG_ERR("perf general max pooling func not be find\n");
            return -1;
        }
    }

    /* general avg pooling, k2s2, k2s2p1, k3s2, k3s2p1 */
    if (param->pool_method == POOL_AVG)
    {
        if ((param->pad_h0 == param->pad_w0) && (param->pad_h1 == param->pad_w1))
        {
            if (param->pad_h0 == 0 && param->pad_h1 == 0)
            {
                if (pool_size == POOL_K2S2)
                    param->funct = (pooling_kernel_t)avg_2x2s2;
                else if (pool_size == POOL_K3S2)
                    param->funct = (pooling_kernel_t)avg_3x3s2;
            }
            else if (param->pad_h0 == 1 && param->pad_h1 == 1)
            {
                if (pool_size == POOL_K2S2)
                    param->funct = (pooling_kernel_t)avg_2x2s2_p1;
                else if (pool_size == POOL_K3S2)
                    param->funct = (pooling_kernel_t)avg_3x3s2_p1;
                else if (pool_size == POOL_K3S1)
                    param->funct = (pooling_kernel_t)avg_3x3s1_p1;
            }
            else if (param->pad_h0 == 0 && param->pad_h1 == 1)
            {
                if (pool_size == POOL_K3S2)
                    param->funct = (pooling_kernel_t)avg_3x3s2;
            }
        }

        if (param->funct != NULL)
            return 0;
        else
        {
            TLOG_ERR("perf general avg pooling func not be find\n");
            return -1;
        }
    }

    TLOG_ERR("perf pooling func not be find\n");
    return -1;
}

int pooling_kernel_perf_run(struct tensor* input, struct tensor* output, struct pool_param* param, int num_thread)
{
    // TLOG_ERR("perf pooling_kernel_run\n");
    int is_caffe = param->caffe_flavor;
    pooling_kernel_t kernel = (pooling_kernel_t)(param->funct);

    int batch = input->dims[0];
    int c = input->dims[1];
    int in_h = input->dims[2];
    int in_w = input->dims[3];

    int out_h = output->dims[2];
    int out_w = output->dims[3];

    int img_size = c * in_h * in_w;
    int feature_size = c * out_h * out_w;

    for (int n = 0; n < batch; n++)
    {
        void* input_frame = input->data + n * img_size * input->elem_size;
        void* output_frame = output->data + n * feature_size * output->elem_size;

#pragma omp parallel for num_threads(num_thread)
        for (int ch = 0; ch < c; ch++)
        {
            void* cur_input = input_frame + ch * in_h * in_w * input->elem_size;
            void* cur_output = output_frame + ch * out_h * out_w * output->elem_size;
            kernel(cur_input, cur_output, 1, in_h, in_w, out_h, out_w, param->kernel_h, param->kernel_w,
                   param->stride_h, param->stride_w, param->pad_h0, param->pad_w0, param->pad_h1, param->pad_w1,
                   is_caffe);
        }
    }

    return 0;
}
