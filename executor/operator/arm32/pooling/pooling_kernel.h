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
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */
#ifndef __POOLING_KERNEL_H__
#define __POOLING_KERNEL_H__

#include <arm_neon.h>

/**
 * MaxPool_2x2: pooling for ksize=2x2,stride=2, pad=0(default pad=0)
 * @param[in]    input     input data (const float pointer)
 * @param[in]    output    output data (float pointer)
 * @param[in]    inc       input channel (int)
 * @param[in]    inh       input height (int)
 * @param[in]    inw       input width (int)
 * @param[in]    outh      output height (int)
 * @param[in]    outw      output width (int)
 * @param[in]    inc       input channel (int)
 * @param[in]    htail     htail=(inh-ksize_h)%stride_h (int)
 * @param[in]    wtail     wtail=(inw-ksize_w)%stride_w (int)
 * @return		None
 */
static void MaxPool_2x2s2(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int, int,
                          int, int, int, int, int pad_h1, int pad_w1, int)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if(pad_w1 > 0)
    {
        outw--;
    }
    if(pad_h1 > 0)
    {
        outh--;
    }
    int block_w = outw >> 2;
    int remain_w = inw - outw * 2;

    for(int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        const float* line1 = line0 + inw;
        float* out_ptr = output + c * out_hw;
        for(int i = 0; i < outh; i++)
        {
            for(int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p10 = vld1q_f32(line1);
                float32x2_t max0_1 = vpmax_f32(vget_low_f32(p00), vget_low_f32(p10));
                float32x2_t max0_2 = vpmax_f32(vget_high_f32(p00), vget_high_f32(p10));
                max0_1 = vpmax_f32(max0_1, max0_2);

                float32x4_t p01 = vld1q_f32(line0 + 4);
                float32x4_t p11 = vld1q_f32(line1 + 4);
                float32x2_t max1_1 = vpmax_f32(vget_low_f32(p01), vget_low_f32(p11));
                float32x2_t max1_2 = vpmax_f32(vget_high_f32(p01), vget_high_f32(p11));
                max1_1 = vpmax_f32(max1_1, max1_2);

                float32x4_t _max = vcombine_f32(max0_1, max1_1);
                vst1q_f32(out_ptr, _max);
                line0 += 8;
                line1 += 8;
                out_ptr += 4;
            }
            for(int j = block_w * 4; j < outw; j++)
            {
                float32x2_t p1 = vld1_f32(line0);
                float32x2_t p2 = vld1_f32(line1);
                float32x2_t _max = vmax_f32(p1, p2);
                //*out_ptr = std::max(_max[0], _max[1]);
                *out_ptr = std::max(vget_lane_f32(_max, 0), vget_lane_f32(_max, 1));
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if(pad_w1 > 0)
            {
                *out_ptr = std::max(line0[0], line1[0]);
                out_ptr++;
            }
            line0 += remain_w + inw;
            line1 += remain_w + inw;
        }
        if(pad_h1 > 0)
        {
            for(int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p01 = vld1q_f32(line0 + 4);

                float32x2_t max0_1 = vpmax_f32(vget_low_f32(p00), vget_high_f32(p00));
                float32x2_t max0_2 = vpmax_f32(vget_low_f32(p01), vget_high_f32(p01));
                p00 = vcombine_f32(max0_1, max0_2);

                vst1q_f32(out_ptr, p00);
                line0 += 8;
                out_ptr += 4;
            }
            for(int j = block_w * 4; j < outw; j++)
            {
                float32x2_t p1 = vld1_f32(line0);
                //*out_ptr = std::max(p1[0], p1[1]);
                *out_ptr = std::max(vget_lane_f32(p1, 0), vget_lane_f32(p1, 1));
                out_ptr++;
                line0 += 2;
            }
            if(pad_w1 > 0)
            {
                *out_ptr = line0[0];
                out_ptr++;
            }
        }
    }
}

static void AvgPool_2x2s2(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int, int,
                          int, int, int, int, int pad_h1, int pad_w1, int)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if(pad_w1 > 0)
    {
        outw--;
    }
    if(pad_h1 > 0)
    {
        outh--;
    }
    int block_w = outw >> 2;
    int remain_w = inw - outw * 2;

    for(int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        const float* line1 = line0 + inw;
        float* out_ptr = output + c * out_hw;
        for(int i = 0; i < outh; i++)
        {
            for(int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p10 = vld1q_f32(line1);
                float32x4_t sum0 = vaddq_f32(p00, p10);

                float32x4_t p01 = vld1q_f32(line0 + 4);
                float32x4_t p11 = vld1q_f32(line1 + 4);
                float32x4_t sum1 = vaddq_f32(p01, p11);

                float32x2_t sum0_1 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                float32x2_t sum0_2 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                sum0 = vcombine_f32(sum0_1, sum0_2);

                sum0 = vmulq_n_f32(sum0, 0.25f);
                vst1q_f32(out_ptr, sum0);
                line0 += 8;
                line1 += 8;
                out_ptr += 4;
            }
            for(int j = block_w * 4; j < outw; j++)
            {
                float32x2_t p1 = vld1_f32(line0);
                float32x2_t p2 = vld1_f32(line1);
                float32x2_t sum = vadd_f32(p1, p2);

                *out_ptr = (vget_lane_f32(sum, 0) + vget_lane_f32(sum, 1)) * 0.25f;
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if(pad_w1 > 0)
            {
                *out_ptr = (line0[0] + line1[0]) * 0.5f;
                out_ptr++;
            }
            line0 += remain_w + inw;
            line1 += remain_w + inw;
        }
        if(pad_h1 > 0)
        {
            for(int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p01 = vld1q_f32(line0 + 4);

                float32x2_t sum0_1 = vpadd_f32(vget_low_f32(p00), vget_high_f32(p00));
                float32x2_t sum0_2 = vpadd_f32(vget_low_f32(p01), vget_high_f32(p01));
                p00 = vcombine_f32(sum0_1, sum0_2);

                p00 = vmulq_n_f32(p00, 0.5f);
                vst1q_f32(out_ptr, p00);
                line0 += 8;
                out_ptr += 4;
            }
            for(int j = block_w * 4; j < outw; j++)
            {
                float32x2_t p1 = vld1_f32(line0);
                *out_ptr = (vget_lane_f32(p1, 0) + vget_lane_f32(p1, 1)) * 0.5f;
                out_ptr++;
                line0 += 2;
            }
            if(pad_w1 > 0)
            {
                *out_ptr = line0[0];
                out_ptr++;
            }
        }
    }
}

static void MaxPool_3x3s2(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int, int,
                          int, int, int, int, int pad_h1, int pad_w1, int)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if(pad_w1 > 0)
    {
        outw--;
    }
    if(pad_h1 > 0)
    {
        outh--;
    }
    int block_w = outw >> 2;
    int remain_w = inw - outw * 2;

    for(int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        const float* line1 = line0 + inw;
        const float* line2 = line1 + inw;
        float* out_ptr = output + c * out_hw;
        for(int i = 0; i < outh; i++)
        {
            float32x4x2_t p00 = vld2q_f32(line0);
            float32x4x2_t p10 = vld2q_f32(line1);
            float32x4x2_t p20 = vld2q_f32(line2);
            for(int j = 0; j < block_w; j++)
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
            for(int j = block_w * 4; j < outw; j++)
            {
                float max0 = std::max(std::max(line0[0], line0[1]), line0[2]);
                float max1 = std::max(std::max(line1[0], line1[1]), line1[2]);
                float max2 = std::max(std::max(line2[0], line2[1]), line2[2]);
                *out_ptr = std::max(std::max(max0, max1), max2);

                out_ptr++;
                line0 += 2;
                line1 += 2;
                line2 += 2;
            }
            if(pad_w1 > 0)
            {
                float max0 = std::max(std::max(line0[0], line0[1]), std::max(line1[0], line1[1]));
                *out_ptr = std::max(std::max(line2[0], line2[1]), max0);
                out_ptr++;
            }
            line0 += remain_w + inw;
            line1 += remain_w + inw;
            line2 += remain_w + inw;
        }
        if(pad_h1 > 0)
        {
            float32x4x2_t p00 = vld2q_f32(line0);
            float32x4x2_t p10 = vld2q_f32(line1);
            for(int j = 0; j < block_w; j++)
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
            for(int j = block_w * 4; j < outw; j++)
            {
                float max0 = std::max(std::max(line0[0], line0[1]), line0[2]);
                float max1 = std::max(std::max(line1[0], line1[1]), line1[2]);

                *out_ptr = std::max(max0, max1);

                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if(pad_w1 > 0)
            {
                *out_ptr = std::max(std::max(line0[0], line0[1]), std::max(line1[0], line1[1]));
                out_ptr++;
            }
        }
    }
}

static void AvgPool_3x3s2(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int, int,
                          int, int, int, int, int pad_h1, int pad_w1, int)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if(pad_w1 > 0)
    {
        outw--;
    }
    if(pad_h1 > 0)
    {
        outh--;
    }
    int block_w = outw >> 2;
    int remain_w = inw - outw * 2;

    for(int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        const float* line1 = line0 + inw;
        const float* line2 = line1 + inw;
        float* out_ptr = output + c * out_hw;
        for(int i = 0; i < outh; i++)
        {
            float32x4x2_t p00 = vld2q_f32(line0);
            float32x4x2_t p10 = vld2q_f32(line1);
            float32x4x2_t p20 = vld2q_f32(line2);
            for(int j = 0; j < block_w; j++)
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
            for(int j = block_w * 4; j < outw; j++)
            {
                *out_ptr =
                    (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2] + line2[0] + line2[1] + line2[2]) *
                    0.11111111f;
                out_ptr++;
                line0 += 2;
                line1 += 2;
                line2 += 2;
            }
            if(pad_w1 > 0)
            {
                *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * 0.16666667f;
                out_ptr++;
            }
            line0 += remain_w + inw;
            line1 += remain_w + inw;
            line2 += remain_w + inw;
        }
        if(pad_h1 > 0)
        {
            float32x4x2_t p00 = vld2q_f32(line0);
            float32x4x2_t p10 = vld2q_f32(line1);
            for(int j = 0; j < block_w; j++)
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
            for(int j = block_w * 4; j < outw; j++)
            {
                *out_ptr = (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2]) * 0.16666667f;
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if(pad_w1 > 0)
            {
                *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1]) * 0.25f;
                out_ptr++;
            }
        }
    }
}

static void MaxPool_2x2s2_pad1(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int,
                               int, int, int, int, int, int pad_h1, int pad_w1, int)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if(pad_w1 > 0)
        outw--;
    if(pad_h1 > 0)
        outh--;
    int block_w = (outw - 1) >> 2;
    int remain_w = inw - outw * 2 + 1;

    for(int c = 0; c < inc; c++)
    {
        const float* line00 = input + c * in_hw;
        float* out_ptr = output + c * out_hw;
        // h begin
        *out_ptr = line00[0];
        out_ptr++;
        line00++;

        for(int j = 0; j < block_w; j++)
        {
            float32x4_t p00 = vld1q_f32(line00);
            float32x4_t p01 = vld1q_f32(line00 + 4);
            float32x2_t max0_1 = vpmax_f32(vget_low_f32(p00), vget_high_f32(p00));
            float32x2_t max0_2 = vpmax_f32(vget_low_f32(p01), vget_high_f32(p01));

            float32x4_t _max = vcombine_f32(max0_1, max0_2);
            vst1q_f32(out_ptr, _max);
            line00 += 8;
            out_ptr += 4;
        }
        for(int j = block_w * 4 + 1; j < outw; j++)
        {
            *out_ptr = std::max(line00[0], line00[1]);
            out_ptr++;
            line00 += 2;
        }
        if(pad_w1 > 0)
        {
            *out_ptr = line00[0];
            out_ptr++;
        }
        line00 += remain_w;
        // h center
        const float* line0 = line00;
        const float* line1 = line0 + inw;
        for(int i = 1; i < outh; i++)
        {
            // w begin
            *out_ptr = std::max(line0[0], line1[0]);
            out_ptr++;
            line0++;
            line1++;
            // w center
            for(int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p10 = vld1q_f32(line1);
                float32x2_t max0_1 = vpmax_f32(vget_low_f32(p00), vget_low_f32(p10));
                float32x2_t max0_2 = vpmax_f32(vget_high_f32(p00), vget_high_f32(p10));
                max0_1 = vpmax_f32(max0_1, max0_2);

                float32x4_t p01 = vld1q_f32(line0 + 4);
                float32x4_t p11 = vld1q_f32(line1 + 4);
                float32x2_t max1_1 = vpmax_f32(vget_low_f32(p01), vget_low_f32(p11));
                float32x2_t max1_2 = vpmax_f32(vget_high_f32(p01), vget_high_f32(p11));
                max1_1 = vpmax_f32(max1_1, max1_2);

                float32x4_t _max = vcombine_f32(max0_1, max1_1);
                vst1q_f32(out_ptr, _max);
                line0 += 8;
                line1 += 8;
                out_ptr += 4;
            }
            for(int j = block_w * 4 + 1; j < outw; j++)
            {
                float32x2_t p1 = vld1_f32(line0);
                float32x2_t p2 = vld1_f32(line1);
                float32x2_t _max = vmax_f32(p1, p2);
                *out_ptr = std::max(vget_lane_f32(_max, 0), vget_lane_f32(_max, 1));
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            // w end
            if(pad_w1 > 0)
            {
                *out_ptr = std::max(line0[0], line1[0]);
                out_ptr++;
            }
            line0 += inw + remain_w;
            line1 += inw + remain_w;
        }

        // h end
        if(pad_h1 > 0)
        {
            *out_ptr = line0[0];
            out_ptr++;
            line0++;
            for(int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p01 = vld1q_f32(line0 + 4);
                float32x2_t max0_1 = vpmax_f32(vget_low_f32(p00), vget_high_f32(p00));
                float32x2_t max0_2 = vpmax_f32(vget_low_f32(p01), vget_high_f32(p01));

                float32x4_t _max = vcombine_f32(max0_1, max0_2);
                vst1q_f32(out_ptr, _max);
                line0 += 8;
                out_ptr += 4;
            }
            for(int j = block_w * 4 + 1; j < outw; j++)
            {
                *out_ptr = std::max(line0[0], line0[1]);
                out_ptr++;
                line0 += 2;
            }
            if(pad_w1 > 0)
            {
                *out_ptr = line0[0];
                out_ptr++;
            }
        }
    }
}

static void AvgPool_2x2s2_pad1(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int,
                               int, int, int, int, int, int pad_h1, int pad_w1, int is_caffe)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if(pad_w1 > 0)
        outw--;
    if(pad_h1 > 0)
        outh--;
    int block_w = (outw - 1) >> 2;
    int remain_w = inw - outw * 2 + 1;

    for(int c = 0; c < inc; c++)
    {
        const float* line00 = input + c * in_hw;
        float* out_ptr = output + c * out_hw;
        // h begin
        if(is_caffe == 0)
            *out_ptr = line00[0];
        else
            *out_ptr = line00[0] * 0.25f;
        out_ptr++;
        line00++;
        for(int j = 0; j < block_w; j++)
        {
            float32x4_t p00 = vld1q_f32(line00);
            float32x4_t p01 = vld1q_f32(line00 + 4);

            float32x2_t sum0_1 = vpadd_f32(vget_low_f32(p00), vget_high_f32(p00));
            float32x2_t sum0_2 = vpadd_f32(vget_low_f32(p01), vget_high_f32(p01));
            float32x4_t sum0 = vcombine_f32(sum0_1, sum0_2);

            if(is_caffe == 0)
                sum0 = vmulq_n_f32(sum0, 0.5f);
            else
                sum0 = vmulq_n_f32(sum0, 0.25f);
            vst1q_f32(out_ptr, sum0);
            line00 += 8;
            out_ptr += 4;
        }
        for(int j = block_w * 4 + 1; j < outw; j++)
        {
            if(is_caffe == 0)
                *out_ptr = (line00[0] + line00[1]) * 0.5f;
            else
                *out_ptr = (line00[0] + line00[1]) * 0.25f;
            out_ptr++;
            line00 += 2;
        }
        if(pad_w1 > 0)
        {
            if(is_caffe == 0)
                *out_ptr = line00[0];
            else
                *out_ptr = line00[0] * 0.25f;
            out_ptr++;
        }
        line00 += remain_w;
        // h center
        const float* line0 = line00;
        const float* line1 = line0 + inw;
        for(int i = 1; i < outh; i++)
        {
            // w begin
            if(is_caffe == 0)
                *out_ptr = (line0[0] + line1[0]) * 0.5f;
            else
                *out_ptr = (line0[0] + line1[0]) * 0.25f;
            out_ptr++;
            line0++;
            line1++;
            // w center
            for(int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p10 = vld1q_f32(line1);
                float32x4_t sum0 = vaddq_f32(p00, p10);

                float32x4_t p01 = vld1q_f32(line0 + 4);
                float32x4_t p11 = vld1q_f32(line1 + 4);
                float32x4_t sum1 = vaddq_f32(p01, p11);

                float32x2_t sum0_1 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                float32x2_t sum0_2 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                sum0 = vcombine_f32(sum0_1, sum0_2);

                sum0 = vmulq_n_f32(sum0, 0.25f);
                vst1q_f32(out_ptr, sum0);
                line0 += 8;
                line1 += 8;
                out_ptr += 4;
            }
            for(int j = block_w * 4 + 1; j < outw; j++)
            {
                float32x2_t p1 = vld1_f32(line0);
                float32x2_t p2 = vld1_f32(line1);
                float32x2_t sum = vadd_f32(p1, p2);

                *out_ptr = (vget_lane_f32(sum, 0) + vget_lane_f32(sum, 1)) * 0.25f;
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            // w end
            if(pad_w1 > 0)
            {
                if(is_caffe == 0)
                    *out_ptr = (line0[0] + line1[0]) * 0.5f;
                else
                    *out_ptr = (line0[0] + line1[0]) * 0.25;
                out_ptr++;
                line0++;
                line1++;
            }
            line0 += inw + remain_w;
            line1 += inw + remain_w;
        }

        // h end
        if(pad_h1 > 0)
        {
            if(is_caffe == 0)
                *out_ptr = line0[0];
            else
                *out_ptr = line0[0] * 0.25;
            out_ptr++;
            line0++;
            for(int j = 0; j < block_w; j++)
            {
                float32x4_t p00 = vld1q_f32(line0);
                float32x4_t p01 = vld1q_f32(line0 + 4);

                float32x2_t sum0_1 = vpadd_f32(vget_low_f32(p00), vget_high_f32(p00));
                float32x2_t sum0_2 = vpadd_f32(vget_low_f32(p01), vget_high_f32(p01));
                float32x4_t sum0 = vcombine_f32(sum0_1, sum0_2);

                if(is_caffe == 0)
                    sum0 = vmulq_n_f32(sum0, 0.5f);
                else
                    sum0 = vmulq_n_f32(sum0, 0.25f);
                vst1q_f32(out_ptr, sum0);
                line0 += 8;
                out_ptr += 4;
            }
            for(int j = block_w * 4 + 1; j < outw; j++)
            {
                if(is_caffe == 0)
                    *out_ptr = (line0[0] + line0[1]) * 0.5f;
                else
                    *out_ptr = (line0[0] + line0[1]) * 0.25f;
                out_ptr++;
                line0 += 2;
            }
            if(pad_w1 > 0)
            {
                if(is_caffe == 0)
                    *out_ptr = line0[0];
                else
                    *out_ptr = line0[0] * 0.25f;
                out_ptr++;
            }
        }
    }
}

static void MaxPool_3x3s2_pad1(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int,
                               int, int, int, int, int, int pad_h1, int pad_w1, int)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if(pad_w1 > 0)
    {
        outw--;
    }
    if(pad_h1 > 0)
    {
        outh--;
    }
    int block_w = (outw - 1) >> 2;
    int remain_w = inw - outw * 2 + 1;

    for(int c = 0; c < inc; c++)
    {
        const float* line1 = input + c * in_hw;
        const float* line2 = line1 + inw;
        float* out_ptr = output + c * out_hw;

        // h begin ---------------------------------------
        *out_ptr = std::max(std::max(line1[0], line1[1]), std::max(line2[0], line2[1]));
        out_ptr++;
        line1 += 1;
        line2 += 1;

        float32x4x2_t p10 = vld2q_f32(line1);
        float32x4x2_t p20 = vld2q_f32(line2);
        for(int j = 0; j < block_w; j++)
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
        for(int j = block_w * 4 + 1; j < outw; j++)
        {
            float max1 = std::max(std::max(line1[0], line1[1]), line1[2]);
            float max2 = std::max(std::max(line2[0], line2[1]), line2[2]);
            *out_ptr = std::max(max1, max2);

            out_ptr++;
            line1 += 2;
            line2 += 2;
        }
        if(pad_w1 == 1)
        {
            *out_ptr = std::max(std::max(line1[0], line1[1]), std::max(line2[0], line2[1]));
            out_ptr++;
        }
        else if(pad_w1 == 2)
        {
            *out_ptr = std::max(line1[0], line2[0]);
            out_ptr++;
        }
        line1 += remain_w;
        line2 += remain_w;

        // h center ---------------------------------------
        const float* line0 = line1;
        line1 = line2;
        line2 = line1 + inw;
        for(int i = 1; i < outh; i++)
        {
            // left
            float max0 = std::max(std::max(line1[0], line1[1]), std::max(line2[0], line2[1]));
            *out_ptr = std::max(std::max(line0[0], line0[1]), max0);
            out_ptr++;
            line0 += 1;
            line1 += 1;
            line2 += 1;
            // mid
            float32x4x2_t p00 = vld2q_f32(line0);
            float32x4x2_t p10 = vld2q_f32(line1);
            float32x4x2_t p20 = vld2q_f32(line2);
            for(int j = 0; j < block_w; j++)
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
            for(int j = block_w * 4 + 1; j < outw; j++)
            {
                float max0 = std::max(std::max(line0[0], line0[1]), line0[2]);
                float max1 = std::max(std::max(line1[0], line1[1]), line1[2]);
                float max2 = std::max(std::max(line2[0], line2[1]), line2[2]);
                *out_ptr = std::max(std::max(max0, max1), max2);

                out_ptr++;
                line0 += 2;
                line1 += 2;
                line2 += 2;
            }
            if(pad_w1 == 1)
            {
                float max0 = std::max(std::max(line0[0], line0[1]), std::max(line1[0], line1[1]));
                *out_ptr = std::max(std::max(line2[0], line2[1]), max0);
                out_ptr++;
            }
            else if(pad_w1 == 2)
            {
                *out_ptr = std::max(std::max(line0[0], line1[0]), line2[0]);
                out_ptr++;
            }
            line0 += inw + remain_w;
            line1 += inw + remain_w;
            line2 += inw + remain_w;
        }

        // h end ------------------------------------------
        if(pad_h1 == 1)
        {
            *out_ptr = std::max(std::max(line1[0], line1[1]), std::max(line0[0], line0[1]));
            out_ptr++;
            line0 += 1;
            line1 += 1;

            float32x4x2_t p00 = vld2q_f32(line0);
            float32x4x2_t p10 = vld2q_f32(line1);
            for(int j = 0; j < block_w; j++)
            {
                float32x4x2_t p00_new = vld2q_f32(line0 + 8);
                float32x4_t max0 = vmaxq_f32(p00.val[0], p00.val[1]);
                float32x4_t p01 = vextq_f32(p00.val[0], p00_new.val[0], 1);
                max0 = vmaxq_f32(max0, p01);

                float32x4x2_t p10_new = vld2q_f32(line1 + 8);
                float32x4_t max1 = vmaxq_f32(p10.val[0], p10.val[1]);
                float32x4_t p11 = vextq_f32(p10.val[0], p10_new.val[0], 1);
                max1 = vmaxq_f32(max1, p11);

                max1 = vmaxq_f32(max1, max0);
                vst1q_f32(out_ptr, max1);

                p00 = p00_new;
                p10 = p10_new;

                line0 += 8;
                line1 += 8;
                out_ptr += 4;
            }
            for(int j = block_w * 4 + 1; j < outw; j++)
            {
                float max0 = std::max(std::max(line0[0], line0[1]), line0[2]);
                float max1 = std::max(std::max(line1[0], line1[1]), line1[2]);
                *out_ptr = std::max(max1, max0);

                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if(pad_w1 == 1)
            {
                *out_ptr = std::max(std::max(line1[0], line1[1]), std::max(line0[0], line0[1]));
                out_ptr++;
            }
            else if(pad_w1 == 2)
            {
                *out_ptr = std::max(line1[0], line0[0]);
                out_ptr++;
            }
        }
        else if(pad_h1 == 2)
        {
            *out_ptr = std::max(line0[0], line0[1]);
            out_ptr++;
            line0 += 1;

            float32x4x2_t p00 = vld2q_f32(line0);
            for(int j = 0; j < block_w; j++)
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
            for(int j = block_w * 4 + 1; j < outw; j++)
            {
                *out_ptr = std::max(std::max(line0[0], line0[1]), line0[2]);

                out_ptr++;
                line0 += 2;
            }
            if(pad_w1 == 1)
            {
                *out_ptr = std::max(line0[0], line0[1]);
                out_ptr++;
            }
            else if(pad_w1 == 2)
            {
                *out_ptr = line1[0];
                out_ptr++;
            }
        }
    }
}

static void AvgPool_3x3s2_pad1(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int,
                               int, int, int, int, int, int pad_h1, int pad_w1, int is_caffe)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if(pad_w1 > 0)
        outw--;
    if(pad_h1 > 0)
        outh--;
    int block_w = (outw - 1) >> 2;
    int remain_w = inw - outw * 2 + 1;

    for(int c = 0; c < inc; c++)
    {
        const float* line1 = input + c * in_hw;
        const float* line2 = line1 + inw;
        float* out_ptr = output + c * out_hw;

        // h begin ---------------------------------------
        if(is_caffe == 0)
            *out_ptr = (line1[0] + line1[1] + line2[0] + line2[1]) * 0.25f;
        else
            *out_ptr = (line1[0] + line1[1] + line2[0] + line2[1]) * 0.11111111f;
        out_ptr++;
        line1 += 1;
        line2 += 1;

        float32x4x2_t p10 = vld2q_f32(line1);
        float32x4x2_t p20 = vld2q_f32(line2);
        for(int j = 0; j < block_w; j++)
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
            if(is_caffe == 0)
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
        for(int j = block_w * 4 + 1; j < outw; j++)
        {
            if(is_caffe == 0)
                *out_ptr = (line1[0] + line1[1] + line1[2] + line2[0] + line2[1] + line2[2]) * 0.16666667f;
            else
                *out_ptr = (line1[0] + line1[1] + line1[2] + line2[0] + line2[1] + line2[2]) * 0.11111111f;
            out_ptr++;
            line1 += 2;
            line2 += 2;
        }
        if(pad_w1 == 1)
        {
            if(is_caffe == 0)
                *out_ptr = (line1[0] + line1[1] + line2[0] + line2[1]) * 0.25f;
            else
                *out_ptr = (line1[0] + line1[1] + line2[0] + line2[1]) * 0.11111111f;
            out_ptr++;
        }
        else if(pad_w1 == 2)
        {
            if(is_caffe == 0)
                *out_ptr = (line1[0] + line2[0]) * 0.5;
            else
                *out_ptr = (line1[0] + line2[0]) * 0.16666667f;
            out_ptr++;
        }
        line1 += remain_w;
        line2 += remain_w;

        // h center ---------------------------------------
        const float* line0 = line1;
        line1 = line2;
        line2 = line1 + inw;
        for(int i = 0; i < outh; i++)
        {
            // left
            if(is_caffe == 0)
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
            for(int j = 0; j < block_w; j++)
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
            for(int j = block_w * 4 + 1; j < outw; j++)
            {
                *out_ptr =
                    (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2] + line2[0] + line2[1] + line2[2]) *
                    0.11111111f;
                out_ptr++;
                line0 += 2;
                line1 += 2;
                line2 += 2;
            }
            if(pad_w1 == 1)
            {
                if(is_caffe == 0)
                    *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * 0.16666667f;
                else
                    *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1] + line2[0] + line2[1]) * 0.11111111f;

                out_ptr++;
            }
            else if(pad_w1 == 2)
            {
                if(is_caffe == 0)
                    *out_ptr = (line0[0] + line1[0] + line2[0]) * 0.33333333f;
                else
                    *out_ptr = (line0[0] + line1[0] + line2[0]) * 0.16666667f;
                out_ptr++;
            }
            line0 += inw + remain_w;
            line1 += inw + remain_w;
            line2 += inw + remain_w;
        }

        // h end ------------------------------------------
        if(pad_h1 == 1)
        {
            if(is_caffe == 0)
                *out_ptr = (line1[0] + line1[1] + line0[0] + line0[1]) * 0.25f;
            else
                *out_ptr = (line1[0] + line1[1] + line0[0] + line0[1]) * 0.11111111f;
            out_ptr++;
            line1 += 1;
            line0 += 1;

            float32x4x2_t p00 = vld2q_f32(line0);
            float32x4x2_t p10 = vld2q_f32(line1);
            for(int j = 0; j < block_w; j++)
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
                if(is_caffe == 0)
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
            for(int j = block_w * 4 + 1; j < outw; j++)
            {
                if(is_caffe == 0)
                    *out_ptr = (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2]) * 0.16666667f;
                else
                    *out_ptr = (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2]) * 0.11111111f;
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if(pad_w1 == 1)
            {
                if(is_caffe == 0)
                    *out_ptr = (line1[0] + line1[1] + line0[0] + line0[1]) * 0.25f;
                else
                    *out_ptr = (line1[0] + line1[1] + line0[0] + line0[1]) * 0.11111111f;
                out_ptr++;
            }
            else if(pad_w1 == 2)
            {
                if(is_caffe == 0)
                    *out_ptr = (line1[0] + line0[0]) * 0.5f;
                else
                    *out_ptr = (line1[0] + line0[0]) * 0.16666667f;
                out_ptr++;
            }
        }
        else if(pad_h1 == 2)
        {
            if(is_caffe == 0)
                *out_ptr = (line1[0] + line0[0]) * 0.5f;
            else
                *out_ptr = (line0[0] + line0[1]) * 0.16666667f;
            out_ptr++;
            line0 += 1;

            float32x4x2_t p00 = vld2q_f32(line0);
            for(int j = 0; j < block_w; j++)
            {
                float32x4x2_t p00_new = vld2q_f32(line0 + 8);
                float32x4_t sum0 = vaddq_f32(p00.val[0], p00.val[1]);
                float32x4_t p01 = vextq_f32(p00.val[0], p00_new.val[0], 1);
                sum0 = vaddq_f32(sum0, p01);

                if(is_caffe == 0)
                    sum0 = vmulq_n_f32(sum0, 0.33333333f);
                else
                    sum0 = vmulq_n_f32(sum0, 0.16666667f);
                vst1q_f32(out_ptr, sum0);

                p00 = p00_new;

                line0 += 8;
                out_ptr += 4;
            }
            for(int j = block_w * 4 + 1; j < outw; j++)
            {
                if(is_caffe == 0)
                    *out_ptr = (line0[0] + line0[1] + line0[2]) * 0.33333333f;
                else
                    *out_ptr = (line0[0] + line0[1] + line0[2]) * 0.16666667f;
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if(pad_w1 == 1)
            {
                if(is_caffe == 0)
                    *out_ptr = (line0[0] + line0[1]) * 0.5f;
                else
                    *out_ptr = (line0[0] + line0[1]) * 0.16666667f;
                out_ptr++;
            }
            else if(pad_w1 == 2)
            {
                if(is_caffe == 0)
                    *out_ptr = line0[0];
                else
                    *out_ptr = line0[0] * 0.25f;
                out_ptr++;
            }
        }
    }
}

static void MaxPool_3x3s1_pad1(const float* input, float* output, int inc, int inh, int inw, int /*outh*/, int /*outw*/,
                               int, int, int, int, int, int, int pad_h1, int pad_w1, int)
{
    int in_hw = inw * inh;

    int mid_w = inw - 2;
    int mid_h = inh - 2;

    for(int c = 0; c < inc; c++)
    {
        const float* line1 = input + c * in_hw;

        const float* line2 = line1 + inw;
        float* out_ptr = output + c * in_hw;

        // h begin left----[line1+=0]-----------------------------------
        *out_ptr = std::max(std::max(line1[0], line1[1]), std::max(line2[0], line2[1]));
        out_ptr++;

        // h begin center----[line1+=1]----------------------------------
        for(int j = 0; j < mid_w; j++)
        {
            float max1 = std::max(std::max(line1[0], line1[1]), line1[2]);
            float max2 = std::max(std::max(line2[0], line2[1]), line2[2]);
            *out_ptr = std::max(max2, max1);
            out_ptr++;
            line1 += 1;
            line2 += 1;
        }
        // h begin right----[line1+=2]-----------------------------------
        *out_ptr = std::max(std::max(line1[0], line1[1]), std::max(line2[0], line2[1]));
        out_ptr++;
        line1 += 2;
        line2 += 2;

        // h center ---------------------------------------
        const float* line0 = input + c * in_hw;

        for(int i = 0; i < mid_h; i++)
        {
            // left
            float max0 = std::max(std::max(line1[0], line1[1]), std::max(line2[0], line2[1]));
            *out_ptr = std::max(std::max(line0[0], line0[1]), max0);
            out_ptr++;

            // mid
            for(int j = 0; j < mid_w; j++)
            {
                float max0 = std::max(std::max(line0[0], line0[1]), line0[2]);
                float max1 = std::max(std::max(line1[0], line1[1]), line1[2]);
                float max2 = std::max(std::max(line2[0], line2[1]), line2[2]);
                *out_ptr = std::max(std::max(max0, max1), max2);
                out_ptr++;
                line0 += 1;
                line1 += 1;
                line2 += 1;
            }

            max0 = std::max(std::max(line1[0], line1[1]), std::max(line2[0], line2[1]));
            *out_ptr = std::max(std::max(line0[0], line0[1]), max0);
            out_ptr++;
            line0 += 2;
            line1 += 2;
            line2 += 2;
        }

        // h end ------------------------------------------
        *out_ptr = std::max(std::max(line1[0], line1[1]), std::max(line0[0], line0[1]));
        out_ptr++;

        for(int j = 0; j < mid_w; j++)
        {
            float max0 = std::max(std::max(line0[0], line0[1]), line0[2]);
            float max1 = std::max(std::max(line1[0], line1[1]), line1[2]);

            *out_ptr = std::max(max0, max1);
            out_ptr++;
            line0 += 1;
            line1 += 1;
        }

        *out_ptr = std::max(std::max(line1[0], line1[1]), std::max(line0[0], line0[1]));
        out_ptr++;
    }
}

// TODO: parallel in channel
static void Global_AvgPool(const float* input, float* output, int inc, int inh, int inw, int /*outh*/, int /*outw*/,
                           int, int, int, int, int, int, int, int, int)
{
    int in_hw = inw * inh;
    int block = in_hw >> 3;
    int tail = in_hw & ~7;

    for(int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        float* out_ptr = output + c;
        float sum = 0.f;
        for(int j = 0; j < block; j++)
        {
            float32x4_t p00 = vld1q_f32(line0);
            float32x4_t p01 = vld1q_f32(line0 + 4);
            p00 = vaddq_f32(p00, p01);
            sum += (vgetq_lane_f32(p00, 0) + vgetq_lane_f32(p00, 1) + vgetq_lane_f32(p00, 2) + vgetq_lane_f32(p00, 3));
            line0 += 8;
        }
        for(int j = tail; j < in_hw; j++)
        {
            sum += line0[0];
            line0++;
        }
        *out_ptr = sum / in_hw;
    }
}

// TODO: parallel in channel
static void Global_MaxPool(const float* input, float* output, int inc, int inh, int inw, int /*outh*/, int /*outw*/,
                           int, int, int, int, int, int, int, int, int)
{
    int in_hw = inw * inh;
    int block = in_hw >> 3;
    int tail = in_hw & ~7;

    for(int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        float* out_ptr = output + c;
        float32x4_t p00 = vld1q_f32(line0);
        float32x4_t res = p00;
        for(int j = 0; j < block; j++)
        {
            float32x4_t p00 = vld1q_f32(line0);
            float32x4_t p01 = vld1q_f32(line0 + 4);
            float32x4_t max0 = vmaxq_f32(p00, p01);
            res = vmaxq_f32(res, max0);
            line0 += 8;
        }
        float max_ = std::max(std::max(vgetq_lane_f32(res, 0), vgetq_lane_f32(res, 1)), std::max(vgetq_lane_f32(res, 2), vgetq_lane_f32(res, 3)));
        for(int j = tail; j < in_hw; j++)
        {
            max_ = std::max(max_, line0[0]);
            line0++;
        }
        *out_ptr = max_;
    }
}

static void Generic_MaxPool(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                            int k_w, int stride_h, int stride_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int)
{
    int in_hw = inh * inw;
    int out_hw = outh * outw;
    for(int c = 0; c < inc; c++)
    {
        int c_skip = c * in_hw;
        int oc_skip = c * out_hw;

        for(int ph = 0; ph < outh; ph++)
        {
            int h_start = ph * stride_h - pad_h0;
            int h_end = std::min(h_start + k_h, inh);
            h_start = std::max(h_start, 0);

            for(int pw = 0; pw < outw; pw++)
            {
                int w_start = pw * stride_w - pad_w0;
                int w_end = std::min(w_start + k_w, inw);
                w_start = std::max(w_start, 0);

                const int out_index = oc_skip + ph * outw + pw;
                output[out_index] = input[c_skip + h_start * inw + w_start];
                for(int h = h_start; h < h_end; h++)
                {
                    for(int w = w_start; w < w_end; w++)
                    {
                        int in_index = c_skip + h * inw + w;

                        if(input[in_index] > output[out_index])
                        {
                            output[out_index] = input[in_index];
                        }
                    }
                }    // end ksize_h,ksize_w
            }
        }
    }
}

static void Generic_AvgPool(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                            int k_w, int stride_h, int stride_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1,
                            int is_caffe)
{
    int in_hw = inh * inw;
    int out_hw = outh * outw;
    for(int c = 0; c < inc; c++)
    {
        int c_skip = c * in_hw;
        int oc_skip = c * out_hw;

        for(int ph = 0; ph < outh; ph++)
        {
            for(int pw = 0; pw < outw; pw++)
            {
                int h_start = ph * stride_h - pad_h0;
                int w_start = pw * stride_w - pad_w0;
                int h_end, w_end, pool_size;

                if(is_caffe == 0)
                {
                    h_end = std::min(h_start + k_h, inh);
                    w_end = std::min(w_start + k_w, inw);

                    h_start = std::max(h_start, 0);
                    w_start = std::max(w_start, 0);
                    pool_size = (h_end - h_start) * (w_end - w_start);
                }
                else
                {
                    h_end = std::min(h_start + k_h, inh + pad_h0);
                    w_end = std::min(w_start + k_w, inw + pad_w0);
                    pool_size = (h_end - h_start) * (w_end - w_start);

                    h_start = std::max(h_start, 0);
                    w_start = std::max(w_start, 0);
                    h_end = std::min(h_end, inh);
                    w_end = std::min(w_end, inw);
                }

                const int out_index = oc_skip + ph * outw + pw;
                output[out_index] = 0.f;
                for(int h = h_start; h < h_end; h++)
                {
                    for(int w = w_start; w < w_end; w++)
                    {
                        output[out_index] += input[c_skip + h * inw + w];
                    }
                }    // end ksize_h,ksize_w
                output[out_index] /= pool_size;
            }
        }
    }
}
#endif
