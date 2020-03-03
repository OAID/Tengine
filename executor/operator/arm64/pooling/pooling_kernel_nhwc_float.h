
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
 * Author: haoluo@openailab.com
 */

#ifndef __POOLING_KERNEL_NHWC_FP32_H__
#define __POOLING_KERNEL_NHWC_FP32_H__

#include <arm_neon.h>
#include <cfloat>

void eltwise_max(float* out, const float* in, int c)
{
    int i = 0;
    for(i = 0; i + 3 < c; i += 4)
    {
        float32x4_t in_4 = vld1q_f32(in + i);
        float32x4_t out_4 = vld1q_f32(out + i);
        float32x4_t max_4 = vmaxq_f32(in_4, out_4);
        vst1q_f32(out + i, max_4);
    }

    for(; i < c; i++)
        out[i] = std::max(out[i], in[i]);
}

void eltwise_add(float* out, const float* in, int c)
{
    int i = 0;
    for(i = 0; i + 3 < c; i += 4)
    {
        float32x4_t in_4 = vld1q_f32(in + i);
        float32x4_t out_4 = vld1q_f32(out + i);
        out_4 = vaddq_f32(in_4, out_4);
        vst1q_f32(out + i, out_4);
    }

    for(; i < c; i++)
        out[i] += in[i];
}

void eltwise_mul(float* out, int c, float scale)
{
    int i = 0;
    float32x4_t scale_4 = vdupq_n_f32(scale);
    for(i = 0; i + 3 < c; i += 4)
    {
        float32x4_t out_4 = vld1q_f32(out + i);
        out_4 = vmulq_f32(out_4, scale_4);
        vst1q_f32(out + i, out_4);
    }

    for(; i < c; i++)
        out[i] *= scale;
}

void eltwise_set_zero(float* out, int c)
{
    float zero = FLT_MIN;

    int i = 0;
    float32x4_t zero_4 = vdupq_n_f32(zero);
    for(i = 0; i + 3 < c; i += 4)
    {
        vst1q_f32(out + i, zero_4);
    }

    for(; i < c; i++)
        out[i] = zero;
}

void eltwise_set_min(float* out, int c)
{
    float min = -FLT_MAX;

    int i = 0;
    float32x4_t min_4 = vdupq_n_f32(min);
    for(i = 0; i + 3 < c; i += 4)
    {
        vst1q_f32(out + i, min_4);
    }

    for(; i < c; i++)
        out[i] = min;
}

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

// global
static void Global_AvgPool_nhwc_float(const float* input, float* output, int inc, int inh, int inw, int, int, int, int,
                                      int, int, int, int, int, int, int, int start_c, int end_c)
{
    int calc_size = end_c - start_c;
    float* cur_out = output + start_c;
    eltwise_set_zero(cur_out, calc_size);

    float scale = 1.0 / (inh * inw);
    for(int i_h = 0; i_h < inh; i_h++)
    {
        for(int i_w = 0; i_w < inw; i_w++)
        {
            const float* cur_input = input + (i_h * inw + i_w) * inc + start_c;
            eltwise_add(cur_out, cur_input, calc_size);
        }
    }
    eltwise_mul(cur_out, calc_size, scale);
}

static void Global_MaxPool_nhwc_float(const float* input, float* output, int inc, int inh, int inw, int, int, int, int,
                                      int, int, int, int, int, int, int, int start_c, int end_c)
{
    int calc_size = end_c - start_c;
    float* cur_out = output + start_c;
    eltwise_set_min(cur_out, calc_size);
    for(int i_h = 0; i_h < inh; i_h++)
    {
        for(int i_w = 0; i_w < inw; i_w++)
        {
            const float* cur_input = input + (i_h * inw + i_w) * inc + start_c;
            eltwise_max(cur_out, cur_input, calc_size);
        }
    }
}
#if 0
// 2x2s2
static void AvgPool_2x2s2_nhwc_float(const float* input, float* output, int inc, int inh, int inw, int outh, int outw,
                                     int, int, int, int, int, int, int pad_h1, int pad_w1, int, int, int)
{
    if(pad_w1 > 0)
    {
        outw--;
    }
    if(pad_h1 > 0)
    {
        outh--;
    }

    int block_c = inc >> 2;
    int remain_w = inw - outw * 2;
    // 4 channels as a block(parallel)
    for(int k = 0; k < block_c; k++)
    {
        const float* p0 = input + k * 4;
        const float* p1 = p0 + inw * inc;
        float* o_ptr = output + k * 4;
        // lines that no need to pad
        for(int i = 0; i < outh; i++)
        {
            // collumns that no need to pad
            for(int j = 0; j < outw; j++)
            {
                float32x4_t d0_4c = vld1q_f32(p0);
                float32x4_t d1_4c = vld1q_f32(p0 + inc);
                float32x4_t d2_4c = vld1q_f32(p1);
                float32x4_t d3_4c = vld1q_f32(p1 + inc);
                float32x4_t sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), vaddq_f32(d2_4c, d3_4c));
                float32x4_t sca_4c = vdupq_n_f32(0.25);
                float32x4_t sum_4c_n = vmulq_f32(sum_4c, sca_4c);
                vst1q_f32(o_ptr, sum_4c_n);
                o_ptr += inc;
                p0 += 2 * inc;
                p1 += 2 * inc;
            }
            // the pad column
            if(pad_w1 > 0)
            {
                float32x4_t d0_4c = vld1q_f32(p0);
                float32x4_t d1_4c = vld1q_f32(p1);
                float32x4_t sum_4c = vaddq_f32(d0_4c, d1_4c);
                float32x4_t sca_4c = vdupq_n_f32(0.50);
                float32x4_t sum_4c_n = vmulq_f32(sum_4c, sca_4c);
                vst1q_f32(o_ptr, sum_4c_n);
                o_ptr += inc;
            }
            p0 += remain_w * inc + inw * inc;
            p1 += remain_w * inc + inw * inc;
        }
        // the pad line
        if(pad_h1 > 0)
        {
            // columns that no need to pad
            for(int j = 0; j < outw; j++)
            {
                float32x4_t d0_4c = vld1q_f32(p0);
                float32x4_t d1_4c = vld1q_f32(p0 + inc);
                float32x4_t sum_4c = vaddq_f32(d0_4c, d1_4c);
                float32x4_t sca_4c = vdupq_n_f32(0.50);
                float32x4_t sum_4c_n = vmulq_f32(sum_4c, sca_4c);
                vst1q_f32(o_ptr, sum_4c_n);
                o_ptr += inc;
                p0 += 2 * inc;
            }
            // the pad column
            if(pad_w1 > 0)
            {
                float32x4_t d0_4c = vld1q_f32(p0);
                vst1q_f32(o_ptr, d0_4c);
            }
        }
    }
    // remain channels(sequence)
    for(int k = 4 * block_c; k < inc; k++)
    {
        const float* p0 = input + k;
        const float* p1 = p0 + inw * inc;
        float* o_ptr = output + k;
        // lines that no need to pad
        for(int i = 0; i < outh; i++)
        {
            // columns that no need to pad
            for(int j = 0; j < outw; j++)
            {
                float d0_1c = p0[0];
                float d1_1c = (p0 + inc)[0];
                float d2_1c = p1[0];
                float d3_1c = (p1 + inc)[0];
                float sum_1c = (d0_1c + d1_1c + d2_1c + d3_1c) / 4;
                *o_ptr = sum_1c;
                o_ptr += inc;
                p0 += 2 * inc;
                p1 += 2 * inc;
            }
            // the pad column
            if(pad_w1 > 0)
            {
                float d0_1c = p0[0];
                float d1_1c = p1[0];
                float sum_1c = (d0_1c + d1_1c) > 1;
                *o_ptr = sum_1c;
                o_ptr += inc;
            }
            p0 += remain_w * inc + inw * inc;
            p1 += remain_w * inc + inw * inc;
        }
        // the pad line
        if(pad_h1 > 0)
        {
            // columns that no need to pad
            for(int j = 0; j < outw; j++)
            {
                float d0_1c = p0[0];
                float d1_1c = (p0 + inc)[0];
                float sum_1c = (d0_1c + d1_1c) > 1;
                *o_ptr = sum_1c;
                o_ptr += inc;
                p0 += 2 * inc;
            }
            // the pad column
            if(pad_w1 > 0)
            {
                float d0_1c = p0[0];
                *o_ptr = d0_1c;
            }
        }
    }
}

// 2x2s2_pad1
static void AvgPool_2x2s2_pad1_nhwc_float(const float* input, float* output, int inc, int inh, int inw, int outh,
                                          int outw, int, int, int, int, int, int, int pad_h1, int pad_w1, int, int, int)
{
    if(pad_w1 > 0)
    {
        outw--;
    }
    if(pad_h1 > 0)
    {
        outh--;
    }

    int block_c = inc >> 2;
    int remain_w = inw - outw * 2 + 1;
    // 4 channels as a block(parallel)
    for(int k = 0; k < block_c; k++)
    {
        const float* p0 = input + k * 4;
        const float* p1 = p0 + inw * inc;
        float* o_ptr = output + k * 4;
        // the first line and middle lines
        for(int i = 0; i < outh; i++)
        {
            // the first line that need to pad
            if(i == 0)
            {
                // the first column that need to pad
                {
                    float32x4_t d0_4c = vld1q_f32(p0);
                    vst1q_f32(o_ptr, d0_4c);
                    o_ptr += inc;
                    p0 += inc;
                }
                // the middle columns that no need to pad
                for(int j = 1; j < outw; j++)
                {
                    float32x4_t d0_4c = vld1q_f32(p0);
                    float32x4_t d1_4c = vld1q_f32(p0 + inc);
                    float32x4_t sum_4c = vaddq_f32(d0_4c, d1_4c);
                    float32x4_t sca_4c = vdupq_n_f32(0.50);
                    float32x4_t sum_4c_n = vmulq_f32(sum_4c, sca_4c);
                    vst1q_f32(o_ptr, sum_4c_n);
                    o_ptr += inc;
                    p0 += 2 * inc;
                }
                // the last column that need to pad
                if(pad_w1 > 0)
                {
                    float32x4_t d0_4c = vld1q_f32(p0);
                    vst1q_f32(o_ptr, d0_4c);
                    o_ptr += inc;
                }
                p0 += remain_w * inc;
            }
            // the midddle lines that no need to pad
            else
            {
                p1 = p0 + inw * inc;
                // the first colunm that need to pad
                {
                    float32x4_t d0_4c = vld1q_f32(p0);
                    float32x4_t d1_4c = vld1q_f32(p1);
                    float32x4_t sum_4c = vaddq_f32(d0_4c, d1_4c);
                    float32x4_t sca_4c = vdupq_n_f32(0.50);
                    float32x4_t sum_4c_n = vmulq_f32(sum_4c, sca_4c);
                    vst1q_f32(o_ptr, sum_4c_n);
                    o_ptr += inc;
                    p0 += inc;
                    p1 += inc;
                }
                // the middle columns that no need to pad
                for(int j = 1; j < outw; j++)
                {
                    float32x4_t d0_4c = vld1q_f32(p0);
                    float32x4_t d1_4c = vld1q_f32(p0 + inc);
                    float32x4_t d2_4c = vld1q_f32(p1);
                    float32x4_t d3_4c = vld1q_f32(p1 + inc);
                    float32x4_t sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), vaddq_f32(d2_4c, d3_4c));
                    float32x4_t sca_4c = vdupq_n_f32(0.25);
                    float32x4_t sum_4c_n = vmulq_f32(sum_4c, sca_4c);
                    vst1q_f32(o_ptr, sum_4c_n);
                    o_ptr += inc;
                    p0 += 2 * inc;
                    p1 += 2 * inc;
                }
                // the last column that need to pad
                if(pad_w1 > 0)
                {
                    float32x4_t d0_4c = vld1q_f32(p0);
                    float32x4_t d1_4c = vld1q_f32(p1);
                    float32x4_t sum_4c = vaddq_f32(d0_4c, d1_4c);
                    float32x4_t sca_4c = vdupq_n_f32(0.50);
                    float32x4_t sum_4c_n = vmulq_f32(sum_4c, sca_4c);
                    vst1q_f32(o_ptr, sum_4c_n);
                    o_ptr += inc;
                }
                p0 += remain_w * inc + inw * inc;
                p1 += remain_w * inc + inw * inc;
            }
        }
        // the last line that need to pad
        if(pad_h1 > 0)
        {
            // the first column that need to pad
            {
                float32x4_t d0_4c = vld1q_f32(p0);
                vst1q_f32(o_ptr, d0_4c);
                o_ptr += inc;
                p0 += inc;
            }
            // the middle columns that no need to pad
            for(int j = 1; j < outw; j++)
            {
                float32x4_t d0_4c = vld1q_f32(p0);
                float32x4_t d1_4c = vld1q_f32(p0 + inc);
                float32x4_t sum_4c = vaddq_f32(d0_4c, d1_4c);
                float32x4_t sca_4c = vdupq_n_f32(0.50);
                float32x4_t sum_4c_n = vmulq_f32(sum_4c, sca_4c);
                vst1q_f32(o_ptr, sum_4c_n);
                o_ptr += inc;
                p0 += 2 * inc;
            }
            // the last column that need to pad
            if(pad_w1 > 0)
            {
                float32x4_t d0_4c = vld1q_f32(p0);
                vst1q_f32(o_ptr, d0_4c);
            }
        }
    }
    // remain channels(sequence)
    for(int k = 4 * block_c; k < inc; k++)
    {
        const float* p0 = input + k;
        const float* p1 = p0 + inw * inc;
        float* o_ptr = output + k;
        // the first line and middle lines
        for(int i = 0; i < outh; i++)
        {
            // the first line that need to pad
            if(i == 0)
            {
                // the first column that need to pad
                {
                    float d0_1c = p0[0];
                    *o_ptr = d0_1c;
                    o_ptr += inc;
                    p0 += inc;
                }
                // the middle columns that no need to pad
                for(int j = 1; j < outw; j++)
                {
                    float d0_1c = p0[0];
                    float d1_1c = (p0 + inc)[0];
                    float sum_1c = (d0_1c + d1_1c) > 1;
                    *o_ptr = sum_1c;
                    o_ptr += inc;
                    p0 += 2 * inc;
                }
                // the last column that need to pad
                if(pad_w1 > 0)
                {
                    float d0_1c = p0[0];
                    *o_ptr = d0_1c;
                    o_ptr += inc;
                }
                p0 += remain_w * inc;
            }
            // the middle lines that no need to pad
            else
            {
                p1 = p0 + inw * inc;
                // the first column that need to pad
                {
                    float d0_1c = p0[0];
                    float d1_1c = p1[0];
                    float sum_1c = (d0_1c + d1_1c) > 1;
                    *o_ptr = sum_1c;
                    o_ptr += inc;
                    p0 += inc;
                    p1 += inc;
                }
                // the middle columns that no need to pad
                for(int j = 1; j < outw; j++)
                {
                    float d0_1c = p0[0];
                    float d1_1c = (p0 + inc)[0];
                    float d2_1c = p1[0];
                    float d3_1c = (p1 + inc)[0];
                    float sum_1c = (d0_1c + d1_1c + d2_1c + d3_1c) / 4;
                    *o_ptr = sum_1c;
                    o_ptr += inc;
                    p0 += 2 * inc;
                    p1 += 2 * inc;
                }
                // the last column that need to pad
                if(pad_w1 > 0)
                {
                    float d0_1c = p0[0];
                    float d1_1c = p1[0];
                    float sum_1c = (d0_1c + d1_1c) > 1;
                    *o_ptr = sum_1c;
                    o_ptr += inc;
                }
                p0 += remain_w * inc + inw * inc;
                p1 += remain_w * inc + inw * inc;
            }
        }
        // the last line that need to pad
        if(pad_h1 > 0)
        {
            // the first column that need to pad
            {
                float d0_1c = p0[0];
                *o_ptr = d0_1c;
                o_ptr += inc;
                p0 += inc;
            }
            // the middle columns that no need to pad
            for(int j = 1; j < outw; j++)
            {
                float d0_1c = p0[0];
                float d1_1c = (p0 + inc)[0];
                float sum_1c = (d0_1c + d1_1c) > 1;
                *o_ptr = sum_1c;
                o_ptr += inc;
                p0 += 2 * inc;
            }
            // the last column that need to pad
            if(pad_w1 > 0)
            {
                float d0_1c = p0[0];
                *o_ptr = d0_1c;
            }
        }
    }
}

// 3x3s2
static void AvgPool_3x3s2_nhwc_float(const float* input, float* output, int inc, int inh, int inw, int outh, int outw,
                                     int, int, int, int, int, int, int pad_h1, int pad_w1, int, int, int)
{
    if(pad_w1 > 0)
    {
        outw--;
    }
    if(pad_h1 > 0)
    {
        outh--;
    }

    int block_c = inc >> 2;
    int remain_w = inw - outw * 2;
    float scale9 = 0.11111111f;
    float scale6 = 0.16666667f;
    // 4 channels as a block(parallel)
    for(int k = 0; k < block_c; k++)
    {
        const float* p0 = input + k * 4;
        const float* p1 = p0 + inw * inc;
        const float* p2 = p1 + inw * inc;
        float* o_ptr = output + k * 4;
        // float o_tmp = 0;
        // the lines that no need to pad
        for(int i = 0; i < outh; i++)
        {
            float32x4_t d0_4c = vld1q_f32(p0);
            float32x4_t d1_4c = vld1q_f32(p1);
            float32x4_t d2_4c = vld1q_f32(p2);
            p0 += inc;
            p1 += inc;
            p2 += inc;
            // the columns that no need to pad
            for(int j = 0; j < outw; j++)
            {
                float32x4_t sum_4c = vdupq_n_f32(0);
                for(int s = 0; s < 2; s++)
                {
                    sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), vaddq_f32(sum_4c, d2_4c));
                    d0_4c = vld1q_f32(p0);
                    d1_4c = vld1q_f32(p1);
                    d2_4c = vld1q_f32(p2);
                    p0 += inc;
                    p1 += inc;
                    p2 += inc;
                }
                sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), vaddq_f32(sum_4c, d2_4c));
                float32x4_t sca_4c = vdupq_n_f32(scale9);
                sum_4c = vmulq_f32(sum_4c, sca_4c);
                vst1q_f32(o_ptr, sum_4c);
                o_ptr += inc;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,0))) * scale9;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,1))) * scale9;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,2))) * scale9;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,3))) * scale9;
                //*o_ptr++ = (float)o_tmp;
                // o_ptr = o_ptr - 4 + inc;
            }
            // the last column that need to pad
            if(pad_w1 > 0)
            {
                float32x4_t sum_4c = vdupq_n_f32(0);
                sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), vaddq_f32(sum_4c, d2_4c));
                d0_4c = vld1q_f32(p0);
                d1_4c = vld1q_f32(p1);
                d2_4c = vld1q_f32(p2);
                sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), vaddq_f32(sum_4c, d2_4c));
                float32x4_t sca_4c = vdupq_n_f32(scale6);
                sum_4c = vmulq_f32(sum_4c, sca_4c);
                vst1q_f32(o_ptr, sum_4c);
                o_ptr += inc;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,0))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,1))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,2))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,3))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_ptr = o_ptr - 4 + inc;
            }
            p0 += remain_w * inc + inw * inc;
            p1 += remain_w * inc + inw * inc;
            p2 += remain_w * inc + inw * inc;
        }
        // the last line that need to pad
        if(pad_h1 > 0)
        {
            float32x4_t d0_4c = vld1q_f32(p0);
            float32x4_t d1_4c = vld1q_f32(p1);
            p0 += inc;
            p1 += inc;
            // the columns that no need to pad
            for(int j = 0; j < outw; j++)
            {
                float32x4_t sum_4c = vdupq_n_f32(0);
                for(int s = 0; s < 2; s++)
                {
                    sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c);
                    d0_4c = vld1q_f32(p0);
                    d1_4c = vld1q_f32(p1);
                    p0 += inc;
                    p1 += inc;
                }
                sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c);
                float32x4_t sca_4c = vdupq_n_f32(scale6);
                sum_4c = vmulq_f32(sum_4c, sca_4c);
                vst1q_f32(o_ptr, sum_4c);
                o_ptr += inc;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,0))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,1))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,2))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,3))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_ptr = o_ptr - 4 + inc;
            }
            // the last column that need to pad
            if(pad_w1 > 0)
            {
                float32x4_t sum_4c = vdupq_n_f32(0);
                sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c);
                d0_4c = vld1q_f32(p0);
                d1_4c = vld1q_f32(p1);
                sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c);
                float32x4_t sca_4c = vdupq_n_f32(0.25);
                float32x4_t sum_4c_c = vmulq_f32(sum_4c, sca_4c);
                vst1q_f32(o_ptr, sum_4c_c);
                o_ptr += inc;
            }
        }
    }
    // remain channels(sequence)
    for(int k = 4 * block_c; k < inc; k++)
    {
        const float* p0 = input + k;
        const float* p1 = p0 + inw * inc;
        const float* p2 = p1 + inw * inc;
        float* o_ptr = output + k;
        // the lines that no need to pad
        for(int i = 0; i < outh; i++)
        {
            float d0_1c = p0[0];
            float d1_1c = p1[0];
            float d2_1c = p2[0];
            p0 += inc;
            p1 += inc;
            p2 += inc;
            // the columns that no need to pad
            for(int j = 0; j < outw; j++)
            {
                float sum_1c = 0;
                for(int s = 0; s < 2; s++)
                {
                    sum_1c = (d0_1c + d1_1c + d2_1c + sum_1c);
                    d0_1c = p0[0];
                    d1_1c = p1[0];
                    d2_1c = p2[0];
                    p0 += inc;
                    p1 += inc;
                    p2 += inc;
                }
                sum_1c = (d0_1c + d1_1c + d2_1c + sum_1c);
                *o_ptr = ( float )sum_1c * scale9;
                o_ptr += inc;
            }
            // the last column that need to pad
            if(pad_w1 > 0)
            {
                float sum_1c = 0;
                sum_1c = (d0_1c + d1_1c + d2_1c + sum_1c);
                d0_1c = p0[0];
                d1_1c = p1[0];
                d2_1c = p2[0];
                sum_1c = (d0_1c + d1_1c + d2_1c + sum_1c);
                *o_ptr = ( float )sum_1c * scale6;
                o_ptr += inc;
            }
            p0 += remain_w * inc + inw * inc;
            p1 += remain_w * inc + inw * inc;
            p2 += remain_w * inc + inw * inc;
        }
        // the last line that need to pad
        if(pad_h1 > 0)
        {
            float d0_1c = p0[0];
            float d1_1c = p1[0];
            p0 += inc;
            p1 += inc;
            // the columns that no need to pad
            for(int j = 0; j < outw; j++)
            {
                float sum_1c = 0;
                for(int s = 0; s < 2; s++)
                {
                    sum_1c = (d0_1c + d1_1c + sum_1c);
                    d0_1c = p0[0];
                    d1_1c = p1[0];
                    p0 += inc;
                    p1 += inc;
                }
                sum_1c = (d0_1c + d1_1c + sum_1c);
                *o_ptr = ( float )sum_1c * scale6;
                o_ptr += inc;
            }
            // the last column that need to pad
            if(pad_w1 > 0)
            {
                float sum_1c = 0;
                sum_1c = (d0_1c + d1_1c + sum_1c);
                d0_1c = p0[0];
                d1_1c = p1[0];
                sum_1c = (d0_1c + d1_1c + sum_1c);
                *o_ptr = ( float )sum_1c / 4;
                o_ptr += inc;
            }
        }
    }
}

// 3x3s2_pad1
static void AvgPool_3x3s2_pad1_nhwc_float(const float* input, float* output, int inc, int inh, int inw, int outh,
                                          int outw, int, int, int, int, int, int, int pad_h1, int pad_w1, int, int, int)
{
    if(pad_w1 > 0)
    {
        outw--;
    }
    if(pad_h1 > 0)
    {
        outh--;
    }

    int block_c = inc >> 2;
    int remain_w = inw - outw * 2;
    float scale9 = 0.11111111f;
    float scale6 = 0.16666667f;
    // 4 channels as a block(parallel)
    for(int k = 0; k < block_c; k++)
    {
        const float* p0 = input + k * 4;
        const float* p1 = p0 + inw * inc;
        const float* p2 = p1 + inw * inc;
        float* o_ptr = output + k * 4;
        // float o_tmp = 0;
        // the first line that need to pad
        {
            // the first column that need to pad
            float32x4_t sum_4c_tl = vdupq_n_f32(0);
            float32x4_t d0_4c = vld1q_f32(p0);
            float32x4_t d1_4c = vld1q_f32(p1);
            sum_4c_tl = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c_tl);
            p0 += inc;
            p1 += inc;
            d0_4c = vld1q_f32(p0);
            d1_4c = vld1q_f32(p1);
            sum_4c_tl = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c_tl);
            p0 += inc;
            p1 += inc;
            float32x4_t sca_4c = vdupq_n_f32(0.25);
            float32x4_t sum_4c_c = vmulq_f32(sum_4c_tl, sca_4c);
            vst1q_f32(o_ptr, sum_4c_c);
            o_ptr += inc;
            // the middle columns that no need to pad
            for(int j = 1; j < outw; j++)
            {
                float32x4_t sum_4c = vdupq_n_f32(0);
                for(int s = 0; s < 2; s++)
                {
                    sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c);
                    d0_4c = vld1q_f32(p0);
                    d1_4c = vld1q_f32(p1);
                    p0 += inc;
                    p1 += inc;
                }
                sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c);
                float32x4_t sca_4c = vdupq_n_f32(scale6);
                sum_4c = vmulq_f32(sum_4c, sca_4c);
                vst1q_f32(o_ptr, sum_4c);
                o_ptr += inc;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,0))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,1))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,2))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,3))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_ptr = o_ptr - 4 + inc;
            }
            // the last column that need to pad
            if(pad_w1 > 0)
            {
                float32x4_t sum_4c_tr = vdupq_n_f32(0);
                sum_4c_tr = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c_tr);
                d0_4c = vld1q_f32(p0);
                d1_4c = vld1q_f32(p1);
                sum_4c_tr = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c_tr);
                float32x4_t sca_4c = vdupq_n_f32(0.25);
                float32x4_t sum_4c_c = vmulq_f32(sum_4c_tr, sca_4c);
                vst1q_f32(o_ptr, sum_4c_c);
                o_ptr += inc;
            }
            p0 += remain_w * inc + inw * inc;
            p1 += remain_w * inc + inw * inc;
        }
        // the middle lines that no need to pad
        for(int i = 1; i < outh; i++)
        {
            // the first column that need to pad
            p2 = p1 + inw * inc;
            float32x4_t sum_4c_ml = vdupq_n_f32(0);
            float32x4_t d0_4c = vld1q_f32(p0);
            float32x4_t d1_4c = vld1q_f32(p1);
            float32x4_t d2_4c = vld1q_f32(p2);
            sum_4c_ml = vaddq_f32(vaddq_f32(d0_4c, d1_4c), vaddq_f32(sum_4c_ml, d2_4c));
            p0 += inc;
            p1 += inc;
            p2 += inc;
            d0_4c = vld1q_f32(p0);
            d1_4c = vld1q_f32(p1);
            d2_4c = vld1q_f32(p2);
            sum_4c_ml = vaddq_f32(vaddq_f32(d0_4c, d1_4c), vaddq_f32(sum_4c_ml, d2_4c));
            p0 += inc;
            p1 += inc;
            p2 += inc;
            float32x4_t sca_4c = vdupq_n_f32(scale6);
            sum_4c_ml = vmulq_f32(sum_4c_ml, sca_4c);
            vst1q_f32(o_ptr, sum_4c_ml);
            o_ptr += inc;
            // o_tmp = ((float)(vgetq_lane_f32(sum_4c_ml,0))) * scale6;
            //*o_ptr++ = (float)o_tmp;
            // o_tmp = ((float)(vgetq_lane_f32(sum_4c_ml,1))) * scale6;
            //*o_ptr++ = (float)o_tmp;
            // o_tmp = ((float)(vgetq_lane_f32(sum_4c_ml,2))) * scale6;
            //*o_ptr++ = (float)o_tmp;
            // o_tmp = ((float)(vgetq_lane_f32(sum_4c_ml,3))) * scale6;
            //*o_ptr++ = (float)o_tmp;
            // o_ptr = o_ptr - 4 + inc;
            // the middle columns that no need to pad
            for(int j = 1; j < outw; j++)
            {
                float32x4_t sum_4c = vdupq_n_f32(0);
                for(int s = 0; s < 2; s++)
                {
                    sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), vaddq_f32(sum_4c, d2_4c));
                    d0_4c = vld1q_f32(p0);
                    d1_4c = vld1q_f32(p1);
                    d2_4c = vld1q_f32(p2);
                    p0 += inc;
                    p1 += inc;
                    p2 += inc;
                }
                sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), vaddq_f32(sum_4c, d2_4c));
                float32x4_t sca_4c = vdupq_n_f32(scale9);
                sum_4c = vmulq_f32(sum_4c, sca_4c);
                vst1q_f32(o_ptr, sum_4c);
                o_ptr += inc;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,0))) * scale9;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,1))) * scale9;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,2))) * scale9;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,3))) * scale9;
                //*o_ptr++ = (float)o_tmp;
                // o_ptr = o_ptr - 4 + inc;
            }
            // the last column that need to pad
            if(pad_w1 > 0)
            {
                float32x4_t sum_4c = vdupq_n_f32(0);
                sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), vaddq_f32(sum_4c, d2_4c));
                d0_4c = vld1q_f32(p0);
                d1_4c = vld1q_f32(p1);
                d2_4c = vld1q_f32(p2);
                sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), vaddq_f32(sum_4c, d2_4c));
                float32x4_t sca_4c = vdupq_n_f32(scale6);
                sum_4c = vmulq_f32(sum_4c, sca_4c);
                vst1q_f32(o_ptr, sum_4c);
                o_ptr += inc;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,0))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,1))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,2))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,3))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_ptr = o_ptr - 4 + inc;
            }
            p0 += remain_w * inc + inw * inc;
            p1 += remain_w * inc + inw * inc;
            p2 += remain_w * inc + inw * inc;
        }
        // the last line that need to pad
        if(pad_h1 > 0)
        {
            // the first column that need to pad
            float32x4_t sum_4c_bl = vdupq_n_f32(0);
            float32x4_t d0_4c = vld1q_f32(p0);
            float32x4_t d1_4c = vld1q_f32(p1);
            sum_4c_bl = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c_bl);
            p0 += inc;
            p1 += inc;
            d0_4c = vld1q_f32(p0);
            d1_4c = vld1q_f32(p1);
            sum_4c_bl = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c_bl);
            p0 += inc;
            p1 += inc;
            float32x4_t sca_4c = vdupq_n_f32(0.25);
            float32x4_t sum_4c_c = vmulq_f32(sum_4c_bl, sca_4c);
            vst1q_f32(o_ptr, sum_4c_c);
            o_ptr += inc;
            // the middle columns that no need to pad
            for(int j = 1; j < outw; j++)
            {
                float32x4_t sum_4c = vdupq_n_f32(0);
                for(int s = 0; s < 2; s++)
                {
                    sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c);
                    d0_4c = vld1q_f32(p0);
                    d1_4c = vld1q_f32(p1);
                    p0 += inc;
                    p1 += inc;
                }
                sum_4c = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c);
                float32x4_t sca_4c = vdupq_n_f32(scale6);
                sum_4c = vmulq_f32(sum_4c, sca_4c);
                vst1q_f32(o_ptr, sum_4c);
                o_ptr += inc;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,0))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,1))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,2))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_tmp = ((float)(vgetq_lane_f32(sum_4c,3))) * scale6;
                //*o_ptr++ = (float)o_tmp;
                // o_ptr = o_ptr - 4 + inc;
            }
            // the last column that need to pad
            if(pad_w1 > 0)
            {
                float32x4_t sum_4c_br = vdupq_n_f32(0);
                sum_4c_br = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c_br);
                d0_4c = vld1q_f32(p0);
                d1_4c = vld1q_f32(p1);
                sum_4c_br = vaddq_f32(vaddq_f32(d0_4c, d1_4c), sum_4c_br);
                float32x4_t sca_4c = vdupq_n_f32(0.25);
                float32x4_t sum_4c_c = vmulq_f32(sum_4c_br, sca_4c);
                vst1q_f32(o_ptr, sum_4c_c);
            }
        }
    }
    // remain channels(sequence)
    for(int k = 4 * block_c; k < inc; k++)
    {
        const float* p0 = input + k;
        const float* p1 = p0 + inw * inc;
        const float* p2 = p1 + inw * inc;
        float* o_ptr = output + k;
        // the first line that need to pad
        {
            // the first column that need to pad
            float d0_1c = p0[0];
            float d1_1c = p1[0];
            float sum_1c = (d0_1c + d1_1c);
            p0 += inc;
            p1 += inc;
            d0_1c = p0[0];
            d1_1c = p1[0];
            sum_1c = (d0_1c + d1_1c + sum_1c);
            p0 += inc;
            p1 += inc;
            *o_ptr = ( float )sum_1c / 4;
            o_ptr += inc;
            // the middle columns that no need to pad
            for(int j = 1; j < outw; j++)
            {
                sum_1c = 0;
                for(int s = 0; s < 2; s++)
                {
                    sum_1c = (d0_1c + d1_1c + sum_1c);
                    d0_1c = p0[0];
                    d1_1c = p1[0];
                    p0 += inc;
                    p1 += inc;
                }
                sum_1c = (d0_1c + d1_1c + sum_1c);
                *o_ptr = ( float )sum_1c * scale6;
                o_ptr += inc;
            }
            // the last column that need to pad
            if(pad_w1 > 0)
            {
                float sum_1c = 0;
                sum_1c = (d0_1c + d1_1c + sum_1c);
                d0_1c = p0[0];
                d1_1c = p1[0];
                sum_1c = (d0_1c + d1_1c + sum_1c);
                *o_ptr = ( float )sum_1c / 4;
                o_ptr += inc;
            }
            p0 += remain_w * inc + inw * inc;
            p1 += remain_w * inc + inw * inc;
        }
        // the middle lines that need to pad
        for(int i = 1; i < outh; i++)
        {
            // the first column that need to pad
            p2 = p1 + inw * inc;
            float d0_1c = p0[0];
            float d1_1c = p1[0];
            float d2_1c = p2[0];
            float sum_1c = (d0_1c + d1_1c + d2_1c);
            p0 += inc;
            p1 += inc;
            p2 += inc;
            d0_1c = p0[0];
            d1_1c = p1[0];
            d2_1c = p2[0];
            sum_1c = (d0_1c + d1_1c + d2_1c + sum_1c);
            p0 += inc;
            p1 += inc;
            p2 += inc;
            *o_ptr = ( float )sum_1c * scale6;
            o_ptr += inc;
            // the middle columns that no need to pad
            for(int j = 1; j < outw; j++)
            {
                float sum_1c = 0;
                for(int s = 0; s < 2; s++)
                {
                    sum_1c = (d0_1c + d1_1c + d2_1c + sum_1c);
                    d0_1c = p0[0];
                    d1_1c = p1[0];
                    d2_1c = p2[0];
                    p0 += inc;
                    p1 += inc;
                    p2 += inc;
                }
                sum_1c = (d0_1c + d1_1c + d2_1c + sum_1c);
                *o_ptr = ( float )sum_1c * scale9;
                o_ptr += inc;
            }
            // the last column that need to pad
            if(pad_w1 > 0)
            {
                float sum_1c = 0;
                sum_1c = (d0_1c + d1_1c + d2_1c + sum_1c);
                d0_1c = p0[0];
                d1_1c = p1[0];
                d2_1c = p2[0];
                sum_1c = (d0_1c + d1_1c + d2_1c + sum_1c);
                *o_ptr = ( float )sum_1c * scale6;
                o_ptr += inc;
            }
            p0 += remain_w * inc + inw * inc;
            p1 += remain_w * inc + inw * inc;
            p2 += remain_w * inc + inw * inc;
        }
        // the last line that need to pad
        if(pad_h1 > 0)
        {
            // the first column that need to pad
            float d0_1c = p0[0];
            float d1_1c = p1[0];
            float sum_1c = (d0_1c + d1_1c);
            p0 += inc;
            p1 += inc;
            d0_1c = p0[0];
            d1_1c = p1[0];
            sum_1c = (d0_1c + d1_1c + sum_1c);
            p0 += inc;
            p1 += inc;
            *o_ptr = ( float )sum_1c / 4;
            o_ptr += inc;
            // the middle columns that no need to pad
            for(int j = 1; j < outw; j++)
            {
                float sum_1c = 0;
                for(int s = 0; s < 2; s++)
                {
                    sum_1c = (d0_1c + d1_1c + sum_1c);
                    d0_1c = p0[0];
                    d1_1c = p1[0];
                    p0 += inc;
                    p1 += inc;
                }
                sum_1c = (d0_1c + d1_1c + sum_1c);
                *o_ptr = ( float )sum_1c * scale6;
                o_ptr += inc;
            }
            // the last column that need to pad
            if(pad_w1 > 0)
            {
                float sum_1c = 0;
                sum_1c = (d0_1c + d1_1c + sum_1c);
                d0_1c = p0[0];
                d1_1c = p1[0];
                sum_1c = (d0_1c + d1_1c + sum_1c);
                *o_ptr = ( float )sum_1c / 4;
                o_ptr += inc;
            }
        }
    }
}
#endif
// avgpool_3x3s1_pad1
static void AvgPool_3x3s1_pad1_nhwc_float(const float* input, float* output, int inc, int inh, int inw, int, int, int,
                                          int, int, int, int, int, int, int, int is_caffe, int start_c, int end_c)
{
    int calc_size = end_c - start_c;
    for(int i = 0; i < inh * inw; i++)
    {
        float* cur_out = output + i * inc + start_c;
        eltwise_set_zero(cur_out, calc_size);
    }
    for(int ih = 0; ih < inh; ih++)
        for(int iw = 0; iw < inw; iw++)
        {
            int h_start = std::max(ih - 1, 0);
            int h_end = std::min(ih + 2, inh);
            int w_start = std::max(iw - 1, 0);
            int w_end = std::min(iw + 2, inw);
            const float* cur_input = input + (ih * inw + iw) * inc + start_c;
            for(int ph = h_start; ph < h_end; ++ph)
            {
                for(int pw = w_start; pw < w_end; ++pw)
                {
                    float* cur_out = output + (ph * inw + pw) * inc + start_c;
                    eltwise_add(cur_out, cur_input, calc_size);
                }
            }
        }
    for(int oh = 0; oh < inh; oh++)
        for(int ow = 0; ow < inw; ow++)
        {
            float scale = 0.11111111f;
            if(is_caffe)
            {
            }
            else if(oh == 0)
            {
                if(ow == 0 || ow == inw - 1)
                    scale = 0.25f;
                else
                    scale = 0.16666667f;
            }
            else if(oh == inh - 1)
            {
                if(ow == 0 || ow == inw - 1)
                    scale = 0.25f;
                else
                    scale = 0.16666667f;
            }
            else
            {
                if(ow == 0 || ow == inw - 1)
                    scale = 0.16666667f;
            }
            float* cur_out = output + (oh * inw + ow) * inc + start_c;
            eltwise_mul(cur_out, calc_size, scale);
        }
}

// generic pool
static void Generic_MaxPool_nhwc_float(const float* input, float* output, int inc, int inh, int inw, int outh, int outw,
                                       int k_h, int k_w, int stride_h, int stride_w, int pad_h0, int pad_w0, int pad_h1,
                                       int pad_w1, int, int start_c, int end_c)
{
    int calc_size = end_c - start_c;
    for(int i = 0; i < outh * outw; i++)
    {
        float* cur_out = output + i * inc + start_c;
        eltwise_set_min(cur_out, calc_size);
    }
    for(int i_h = 0; i_h < inh; i_h++)
    {
        for(int i_w = 0; i_w < inw; i_w++)
        {
            const float* cur_input = input + (i_h * inw + i_w) * inc + start_c;
            int h_start = (i_h + pad_h0) < k_h ? 0 : (i_h + pad_h0 - k_h) / stride_h + 1;
            int h_end = std::min((i_h + pad_h0) / stride_h + 1, outh);
            int w_start = (i_w + pad_w0) < k_w ? 0 : (i_w + pad_w0 - k_w) / stride_w + 1;
            int w_end = std::min((i_w + pad_w0) / stride_w + 1, outw);
            for(int ph = h_start; ph < h_end; ++ph)
            {
                for(int pw = w_start; pw < w_end; ++pw)
                {
                    float* cur_out = output + (ph * outw + pw) * inc + start_c;
                    eltwise_max(cur_out, cur_input, calc_size);
                }
            }
        }
    }
}

static void Generic_AvgPool_nhwc_float(const float* input, float* output, int inc, int inh, int inw, int outh, int outw,
                                       int k_h, int k_w, int stride_h, int stride_w, int pad_h0, int pad_w0, int pad_h1,
                                       int pad_w1, int is_caffe, int start_c, int end_c)
{
    int calc_size = end_c - start_c;
    for(int i = 0; i < outh * outw; i++)
    {
        float* cur_out = output + i * inc + start_c;
        eltwise_set_zero(cur_out, calc_size);
    }
    for(int c = start_c; c < end_c; c++)
    {
        int c_skip = c;
        int oc_skip = c;

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

                const int out_index = oc_skip + ph * outw * inc + pw * inc;
                float o_tmp = 0.f;
                for(int h = h_start; h < h_end; h++)
                {
                    for(int w = w_start; w < w_end; w++)
                    {
                        o_tmp += ( float )(input[c_skip + h * inw * inc + w * inc]);
                    }
                }    // end ksize_h,ksize_w
                output[out_index] = ( float )(o_tmp / pool_size);
            }
        }
    }
}
#endif
