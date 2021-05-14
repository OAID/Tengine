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
 * Author: 1091545398@qq.com
 */

#include "pooling_param.h"

#include "graph/tensor.h"
#include "utility/log.h"

#include <emmintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

#define POOL_GENERIC 0
#define POOL_K2S2 1
#define POOL_K3S2 2
#define POOL_K3S1 3


typedef void (*pooling_kernel_t)(const void* input, void* output, int inc, int inh, int inw, int outh, int outw, int,
                                 int, int, int, int, int, int pad_h1, int pad_w1, int);


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
            int j = 0;
            for (; j + 4 < mid_w; j += 2)
            {
                __m128 r0 = _mm_loadu_ps(line0);
                __m128 r1 = _mm_loadu_ps(line1);
                __m128 r2 = _mm_loadu_ps(line2);

                __m128 max_truct = _mm_max_ps(_mm_max_ps(r0, r1), r2);
                float* max0 = (float*)(&max_truct);

                *out_ptr = fmax(fmax(max0[0], max0[1]), max0[2]);
                out_ptr++;
                *out_ptr = fmax(fmax(max0[1], max0[2]), max0[3]);
                out_ptr++;
                line0 += 2;
                line1 += 2;
                line2 += 2;
            }

            for (; j < mid_w; j++)
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

    const __m128 scalar_025 = _mm_set1_ps(0.25);
    const __m128 scalar_05 = _mm_set1_ps(0.5);

    for (int c = 0; c < inc; c++)
    {
        const float* line00 = input + c * in_hw;
        float* out_ptr = output + c * out_hw;
        // h begin
        if (is_caffe == 0)
            *out_ptr = line00[0];
        else
            *out_ptr = line00[0] * 0.25;

        out_ptr++;
        line00++;
        for (int j = 0; j < block_w; j++)
        {
            __m128 p00 = _mm_loadu_ps(line00);
            __m128 p01 = _mm_loadu_ps(line00 + 4);

            __m128 r00 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(2, 0, 2, 0));
            __m128 r01 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(3, 1, 3, 1));

            __m128 _sum = _mm_add_ps(r00, r01);

            if (is_caffe == 0)
                _sum = _mm_mul_ps(_sum, scalar_05);
            else
                _sum = _mm_mul_ps(_sum, scalar_025);
            _mm_storeu_ps(out_ptr, _sum);

            out_ptr += 4;
            line00 += 8;
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
                *out_ptr = (line0[0] + line1[0]) * 0.25;
            else
                *out_ptr = (line0[0] + line1[0]) * 0.25;
            out_ptr++;
            line0++;
            line1++;
            // w center
            for (int j = 0; j < block_w; j++)
            {
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 4);
                __m128 p10 = _mm_loadu_ps(line1);
                __m128 p11 = _mm_loadu_ps(line1 + 4);

                __m128 r00 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(2, 0, 2, 0));
                __m128 r01 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(3, 1, 3, 1));
                __m128 r10 = _mm_shuffle_ps(p10, p11, _MM_SHUFFLE(2, 0, 2, 0));
                __m128 r11 = _mm_shuffle_ps(p10, p11, _MM_SHUFFLE(3, 1, 3, 1));

                __m128 sum0 = _mm_add_ps(r00, r01);
                __m128 sum1 = _mm_add_ps(r10, r11);
                __m128 _sum = _mm_add_ps(sum0, sum1);

                _mm_storeu_ps(out_ptr, _mm_mul_ps(_sum, scalar_025));

                out_ptr += 4;
                line0 += 8;
                line1 += 8;
            }
            for (int j = block_w * 4 + 1; j < outw; j++)
            {
                *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1]) * 0.25;
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            // w end
            if (inw % 2 == 0)
            {
                if (is_caffe == 0)
                    *out_ptr = (line0[0] + line1[0]) * 0.5;
                else
                    *out_ptr = (line0[0] + line1[0]) * 0.25;
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
                *out_ptr = line0[0] * 0.25;
            out_ptr++;
            line0++;
            for (int j = 0; j < block_w; j++)
            {
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 4);

                __m128 r00 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(2, 0, 2, 0));
                __m128 r01 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(3, 1, 3, 1));

                __m128 _sum = _mm_add_ps(r00, r01);
                if (is_caffe == 0)
                    _mm_storeu_ps(out_ptr, _mm_mul_ps(_sum, scalar_05));
                else
                    _mm_storeu_ps(out_ptr, _mm_mul_ps(_sum, scalar_025));
                out_ptr += 4;
                line0 += 8;
            }
            for (int j = block_w * 4 + 1; j < outw; j++)
            {
                if (is_caffe == 0)
                    *out_ptr = (line0[0] + line0[1]) * 0.5;
                else
                    *out_ptr = (line0[0] + line0[1]) * 0.25;
                out_ptr++;
                line0 += 2;
            }
            if (inw % 2 == 0)
            {
                if (is_caffe == 0)
                    *out_ptr = line0[0];
                else
                    *out_ptr = line0[0] * 0.25;
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
            __m128 p00 = _mm_loadu_ps(line00);
            __m128 p01 = _mm_loadu_ps(line00 + 4);

            __m128 r00 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(2, 0, 2, 0));
            __m128 r01 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(3, 1, 3, 1));

            __m128 _max = _mm_max_ps(r00, r01);
            _mm_storeu_ps(out_ptr, _max);

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
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 4);
                __m128 p10 = _mm_loadu_ps(line1);
                __m128 p11 = _mm_loadu_ps(line1 + 4);

                __m128 r00 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(2, 0, 2, 0));
                __m128 r01 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(3, 1, 3, 1));
                __m128 r10 = _mm_shuffle_ps(p10, p11, _MM_SHUFFLE(2, 0, 2, 0));
                __m128 r11 = _mm_shuffle_ps(p10, p11, _MM_SHUFFLE(3, 1, 3, 1));

                __m128 max0 = _mm_max_ps(r00, r01);
                __m128 max1 = _mm_max_ps(r10, r11);
                __m128 _max = _mm_max_ps(max0, max1);

                _mm_storeu_ps(out_ptr, _max);

                out_ptr += 4;
                line0 += 8;
                line1 += 8;
            }
            for (int j = block_w * 4 + 1; j < outw; j++)
            {
                float l0max = fmax(line0[0], line0[1]);
                float l1max = fmax(line1[0], line1[1]);
                *out_ptr = fmax(l0max, l1max);
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
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 4);

                __m128 r00 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(2, 0, 2, 0));
                __m128 r01 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(3, 1, 3, 1));

                __m128 _max = _mm_max_ps(r00, r01);
                _mm_storeu_ps(out_ptr, _max);

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

        for (int j = 0; j < block_w; j++)
        {
            __m128 p10 = _mm_loadu_ps(line1);
            __m128 p11 = _mm_loadu_ps(line1 + 1);
            __m128 p12 = _mm_loadu_ps(line1 + 2);

            __m128 p14 = _mm_loadu_ps(line1 + 4);
            __m128 p15 = _mm_loadu_ps(line1 + 5);
            __m128 p16 = _mm_loadu_ps(line1 + 6);

            __m128 max102 = _mm_max_ps(p10, p11);
            max102 = _mm_max_ps(max102, p12);
            __m128 max146 = _mm_max_ps(p14, p15);
            max146 = _mm_max_ps(max146, p16);

            __m128 max1 = _mm_shuffle_ps(max102, max146, _MM_SHUFFLE(2, 0, 2, 0));

            __m128 p20 = _mm_loadu_ps(line2);
            __m128 p21 = _mm_loadu_ps(line2 + 1);
            __m128 p22 = _mm_loadu_ps(line2 + 2);

            __m128 p24 = _mm_loadu_ps(line2 + 4);
            __m128 p25 = _mm_loadu_ps(line2 + 5);
            __m128 p26 = _mm_loadu_ps(line2 + 6);

            __m128 max202 = _mm_max_ps(p20, p21);
            max202 = _mm_max_ps(max202, p22);
            __m128 max246 = _mm_max_ps(p24, p25);
            max246 = _mm_max_ps(max246, p26);

            __m128 max2 = _mm_shuffle_ps(max202, max246, _MM_SHUFFLE(2, 0, 2, 0));

            _mm_storeu_ps(out_ptr, _mm_max_ps(max2, max1));

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
            for (int j = 0; j < block_w; j++)
            {
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 1);
                __m128 p02 = _mm_loadu_ps(line0 + 2);

                __m128 p04 = _mm_loadu_ps(line0 + 4);
                __m128 p05 = _mm_loadu_ps(line0 + 5);
                __m128 p06 = _mm_loadu_ps(line0 + 6);

                __m128 max002 = _mm_max_ps(p00, p01);
                max002 = _mm_max_ps(max002, p02);
                __m128 max046 = _mm_max_ps(p04, p05);
                max046 = _mm_max_ps(max046, p06);

                __m128 max0 = _mm_shuffle_ps(max002, max046, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 p10 = _mm_loadu_ps(line1);
                __m128 p11 = _mm_loadu_ps(line1 + 1);
                __m128 p12 = _mm_loadu_ps(line1 + 2);

                __m128 p14 = _mm_loadu_ps(line1 + 4);
                __m128 p15 = _mm_loadu_ps(line1 + 5);
                __m128 p16 = _mm_loadu_ps(line1 + 6);

                __m128 max102 = _mm_max_ps(p10, p11);
                max102 = _mm_max_ps(max102, p12);
                __m128 max146 = _mm_max_ps(p14, p15);
                max146 = _mm_max_ps(max146, p16);

                __m128 max1 = _mm_shuffle_ps(max102, max146, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 p20 = _mm_loadu_ps(line2);
                __m128 p21 = _mm_loadu_ps(line2 + 1);
                __m128 p22 = _mm_loadu_ps(line2 + 2);

                __m128 p24 = _mm_loadu_ps(line2 + 4);
                __m128 p25 = _mm_loadu_ps(line2 + 5);
                __m128 p26 = _mm_loadu_ps(line2 + 6);

                __m128 max202 = _mm_max_ps(p20, p21);
                max202 = _mm_max_ps(max202, p22);
                __m128 max246 = _mm_max_ps(p24, p25);
                max246 = _mm_max_ps(max246, p26);

                __m128 max2 = _mm_shuffle_ps(max202, max246, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 max = _mm_max_ps(_mm_max_ps(max0, max1), max2);
                _mm_storeu_ps(out_ptr, max);

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

            for (int j = 0; j < block_w; j++)
            {
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 1);
                __m128 p02 = _mm_loadu_ps(line0 + 2);

                __m128 p04 = _mm_loadu_ps(line0 + 4);
                __m128 p05 = _mm_loadu_ps(line0 + 5);
                __m128 p06 = _mm_loadu_ps(line0 + 6);

                __m128 max002 = _mm_max_ps(p00, p01);
                max002 = _mm_max_ps(max002, p02);
                __m128 max046 = _mm_max_ps(p04, p05);
                max046 = _mm_max_ps(max046, p06);

                __m128 max0 = _mm_shuffle_ps(max002, max046, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 p10 = _mm_loadu_ps(line1);
                __m128 p11 = _mm_loadu_ps(line1 + 1);
                __m128 p12 = _mm_loadu_ps(line1 + 2);

                __m128 p14 = _mm_loadu_ps(line1 + 4);
                __m128 p15 = _mm_loadu_ps(line1 + 5);
                __m128 p16 = _mm_loadu_ps(line1 + 6);

                __m128 max102 = _mm_max_ps(p10, p11);
                max102 = _mm_max_ps(max102, p12);
                __m128 max146 = _mm_max_ps(p14, p15);
                max146 = _mm_max_ps(max146, p16);

                __m128 max1 = _mm_shuffle_ps(max102, max146, _MM_SHUFFLE(2, 0, 2, 0));

                _mm_storeu_ps(out_ptr, _mm_max_ps(max0, max1));

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

            for (int j = 0; j < block_w; j++)
            {
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 1);
                __m128 p02 = _mm_loadu_ps(line0 + 2);

                __m128 p04 = _mm_loadu_ps(line0 + 4);
                __m128 p05 = _mm_loadu_ps(line0 + 5);
                __m128 p06 = _mm_loadu_ps(line0 + 6);

                __m128 max002 = _mm_max_ps(p00, p01);
                max002 = _mm_max_ps(max002, p02);
                __m128 max046 = _mm_max_ps(p04, p05);
                max046 = _mm_max_ps(max046, p06);

                __m128 max0 = _mm_shuffle_ps(max002, max046, _MM_SHUFFLE(2, 0, 2, 0));

                _mm_store_ps(out_ptr, max0);

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
            for (int j = 0; j < block_w; j++)
            {
                /*
                p00     = [1,2,3,4,5,6,7,8]
                p00.val[0]=[1,3,5,7]

                max0    = [2,4,6,8]
                p00_new = [9,10,11,12,13,14,15,16]
                p01     = [3,5,7,9]
                max0=fmax(max0,p01)=[3,5,7,9]
                */

                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 1);
                __m128 p02 = _mm_loadu_ps(line0 + 2);

                __m128 p04 = _mm_loadu_ps(line0 + 4);
                __m128 p05 = _mm_loadu_ps(line0 + 5);
                __m128 p06 = _mm_loadu_ps(line0 + 6);

                __m128 max002 = _mm_max_ps(p00, p01);
                max002 = _mm_max_ps(max002, p02);
                __m128 max046 = _mm_max_ps(p04, p05);
                max046 = _mm_max_ps(max046, p06);

                __m128 max0 = _mm_shuffle_ps(max002, max046, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 p10 = _mm_loadu_ps(line1);
                __m128 p11 = _mm_loadu_ps(line1 + 1);
                __m128 p12 = _mm_loadu_ps(line1 + 2);

                __m128 p14 = _mm_loadu_ps(line1 + 4);
                __m128 p15 = _mm_loadu_ps(line1 + 5);
                __m128 p16 = _mm_loadu_ps(line1 + 6);

                __m128 max102 = _mm_max_ps(p10, p11);
                max102 = _mm_max_ps(max102, p12);
                __m128 max146 = _mm_max_ps(p14, p15);
                max146 = _mm_max_ps(max146, p16);

                __m128 max1 = _mm_shuffle_ps(max102, max146, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 p20 = _mm_loadu_ps(line2);
                __m128 p21 = _mm_loadu_ps(line2 + 1);
                __m128 p22 = _mm_loadu_ps(line2 + 2);

                __m128 p24 = _mm_loadu_ps(line2 + 4);
                __m128 p25 = _mm_loadu_ps(line2 + 5);
                __m128 p26 = _mm_loadu_ps(line2 + 6);

                __m128 max202 = _mm_max_ps(p20, p21);
                max202 = _mm_max_ps(max202, p22);
                __m128 max246 = _mm_max_ps(p24, p25);
                max246 = _mm_max_ps(max246, p26);

                __m128 max2 = _mm_shuffle_ps(max202, max246, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 max = _mm_max_ps(_mm_max_ps(max0, max1), max2);
                _mm_storeu_ps(out_ptr, max);

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
            for (int j = 0; j < block_w; j++)
            {
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 1);
                __m128 p02 = _mm_loadu_ps(line0 + 2);

                __m128 p04 = _mm_loadu_ps(line0 + 4);
                __m128 p05 = _mm_loadu_ps(line0 + 5);
                __m128 p06 = _mm_loadu_ps(line0 + 6);

                __m128 max002 = _mm_max_ps(p00, p01);
                max002 = _mm_max_ps(max002, p02);
                __m128 max046 = _mm_max_ps(p04, p05);
                max046 = _mm_max_ps(max046, p06);

                __m128 max0 = _mm_shuffle_ps(max002, max046, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 p10 = _mm_loadu_ps(line1);
                __m128 p11 = _mm_loadu_ps(line1 + 1);
                __m128 p12 = _mm_loadu_ps(line1 + 2);

                __m128 p14 = _mm_loadu_ps(line1 + 4);
                __m128 p15 = _mm_loadu_ps(line1 + 5);
                __m128 p16 = _mm_loadu_ps(line1 + 6);

                __m128 max102 = _mm_max_ps(p10, p11);
                max102 = _mm_max_ps(max102, p12);
                __m128 max146 = _mm_max_ps(p14, p15);
                max146 = _mm_max_ps(max146, p16);

                __m128 max1 = _mm_shuffle_ps(max102, max146, _MM_SHUFFLE(2, 0, 2, 0));

                _mm_storeu_ps(out_ptr, _mm_max_ps(max0, max1));

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

    const __m128 scalar_025 = _mm_set1_ps(0.25);
    const __m128 scalar_05 = _mm_set1_ps(0.5);

    for (int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        const float* line1 = line0 + inw;
        float* out_ptr = output + c * out_hw;

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < block_w; j++)
            {
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 4);
                __m128 p10 = _mm_loadu_ps(line1);
                __m128 p11 = _mm_loadu_ps(line1 + 4);

                __m128 r00 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(2, 0, 2, 0));
                __m128 r01 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(3, 1, 3, 1));
                __m128 r10 = _mm_shuffle_ps(p10, p11, _MM_SHUFFLE(2, 0, 2, 0));
                __m128 r11 = _mm_shuffle_ps(p10, p11, _MM_SHUFFLE(3, 1, 3, 1));

                __m128 sum0 = _mm_add_ps(r00, r01);
                __m128 sum1 = _mm_add_ps(r10, r11);
                __m128 sum = _mm_add_ps(sum0, sum1);

                _mm_storeu_ps(out_ptr, _mm_mul_ps(sum, scalar_025));
                line0 += 8;
                line1 += 8;
                out_ptr += 4;
            }

            for (int j = block_w * 4; j < outw; j++)
            {
                *out_ptr = (line0[0] + line0[1] + line1[0] + line1[1]) * 0.25f;

                out_ptr++;
                line0 += 2;
                line1 += 2;
            }

            if (pad_w1 > 0)
            {
                *out_ptr = (line0[0] + line1[0]) * 0.5;
                out_ptr++;
            }
            line0 += remain_w + inw;
            line1 += remain_w + inw;
        }

        if (pad_h1 > 0)
        {
            for (int j = 0; j < block_w; j++)
            {
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 4);

                __m128 r00 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(2, 0, 2, 0));
                __m128 r01 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(3, 1, 3, 1));

                __m128 sum = _mm_add_ps(r00, r01);
                _mm_storeu_ps(out_ptr, _mm_mul_ps(sum, scalar_05));

                line0 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4; j < outw; j++)
            {
                *out_ptr = (line0[0] + line0[1]) * 0.5;
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
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 4);
                __m128 p10 = _mm_loadu_ps(line1);
                __m128 p11 = _mm_loadu_ps(line1 + 4);

                __m128 r00 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(2, 0, 2, 0));
                __m128 r01 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(3, 1, 3, 1));
                __m128 r10 = _mm_shuffle_ps(p10, p11, _MM_SHUFFLE(2, 0, 2, 0));
                __m128 r11 = _mm_shuffle_ps(p10, p11, _MM_SHUFFLE(3, 1, 3, 1));

                __m128 max0 = _mm_max_ps(r00, r01);
                __m128 max1 = _mm_max_ps(r10, r11);
                __m128 _max = _mm_max_ps(max0, max1);

                _mm_storeu_ps(out_ptr, _max);
                line0 += 8;
                line1 += 8;
                out_ptr += 4;
            }

            for (int j = block_w * 4; j < outw; j++)
            {
                float l0max = fmax(line0[0], line0[1]);
                float l1max = fmax(line1[0], line1[1]);
                *out_ptr = fmax(l0max, l1max);

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
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 4);

                __m128 r00 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(2, 0, 2, 0));
                __m128 r01 = _mm_shuffle_ps(p00, p01, _MM_SHUFFLE(3, 1, 3, 1));

                __m128 _max = _mm_max_ps(r00, r01);
                _mm_storeu_ps(out_ptr, _max);

                line0 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4; j < outw; j++)
            {
                *out_ptr = fmax(line0[0], line0[1]);
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
    __m128 scalar_011 = _mm_set1_ps(0.11111111f);
    __m128 scalar_016 = _mm_set1_ps(0.16666667f);
    __m128 scalar_033 = _mm_set1_ps(0.3333333f);

    for (int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        const float* line1 = line0 + inw;
        const float* line2 = line1 + inw;
        float* out_ptr = output + c * out_hw;
        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < block_w; j++)
            {
                /*
                p00     = [1,2,3,4,5,6,7,8]
                p00.val[0]=[1,3,5,7]

                max0    = [2,4,6,8]
                p00_new = [9,10,11,12,13,14,15,16]
                p01     = [3,5,7,9]
                max0=fmax(max0,p01)=[3,5,7,9]
                */

                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 1);
                __m128 p02 = _mm_loadu_ps(line0 + 2);

                __m128 p04 = _mm_loadu_ps(line0 + 4);
                __m128 p05 = _mm_loadu_ps(line0 + 5);
                __m128 p06 = _mm_loadu_ps(line0 + 6);

                __m128 sum002 = _mm_add_ps(p00, p01);
                sum002 = _mm_add_ps(sum002, p02);
                __m128 sum046 = _mm_add_ps(p04, p05);
                sum046 = _mm_add_ps(sum046, p06);

                __m128 sum0 = _mm_shuffle_ps(sum002, sum046, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 p10 = _mm_loadu_ps(line1);
                __m128 p11 = _mm_loadu_ps(line1 + 1);
                __m128 p12 = _mm_loadu_ps(line1 + 2);

                __m128 p14 = _mm_loadu_ps(line1 + 4);
                __m128 p15 = _mm_loadu_ps(line1 + 5);
                __m128 p16 = _mm_loadu_ps(line1 + 6);

                __m128 sum102 = _mm_add_ps(p10, p11);
                sum102 = _mm_add_ps(sum102, p12);
                __m128 sum146 = _mm_add_ps(p14, p15);
                sum146 = _mm_add_ps(sum146, p16);

                __m128 sum1 = _mm_shuffle_ps(sum102, sum146, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 p20 = _mm_loadu_ps(line2);
                __m128 p21 = _mm_loadu_ps(line2 + 1);
                __m128 p22 = _mm_loadu_ps(line2 + 2);

                __m128 p24 = _mm_loadu_ps(line2 + 4);
                __m128 p25 = _mm_loadu_ps(line2 + 5);
                __m128 p26 = _mm_loadu_ps(line2 + 6);

                __m128 sum202 = _mm_add_ps(p20, p21);
                sum202 = _mm_add_ps(sum202, p22);
                __m128 sum246 = _mm_max_ps(p24, p25);
                sum246 = _mm_add_ps(sum246, p26);

                __m128 sum2 = _mm_shuffle_ps(sum202, sum246, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 avg = _mm_mul_ps(_mm_add_ps(_mm_add_ps(sum0, sum1), sum2), scalar_011);
                _mm_storeu_ps(out_ptr, avg);

                line0 += 8;
                line1 += 8;
                line2 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4; j < outw; j++)
            {
                float sum =
                    line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2] + line2[0] + line2[1] + line2[2];
                *out_ptr = sum * 0.11111111f;

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
            for (int j = 0; j < block_w; j++)
            {
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 1);
                __m128 p02 = _mm_loadu_ps(line0 + 2);

                __m128 p04 = _mm_loadu_ps(line0 + 4);
                __m128 p05 = _mm_loadu_ps(line0 + 5);
                __m128 p06 = _mm_loadu_ps(line0 + 6);

                __m128 sum002 = _mm_add_ps(p00, p01);
                sum002 = _mm_add_ps(sum002, p02);
                __m128 sum046 = _mm_add_ps(p04, p05);
                sum046 = _mm_add_ps(sum046, p06);

                __m128 sum0 = _mm_shuffle_ps(sum002, sum046, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 p10 = _mm_loadu_ps(line1);
                __m128 p11 = _mm_loadu_ps(line1 + 1);
                __m128 p12 = _mm_loadu_ps(line1 + 2);

                __m128 p14 = _mm_loadu_ps(line1 + 4);
                __m128 p15 = _mm_loadu_ps(line1 + 5);
                __m128 p16 = _mm_loadu_ps(line1 + 6);

                __m128 sum102 = _mm_add_ps(p10, p11);
                sum102 = _mm_add_ps(sum102, p12);
                __m128 sum146 = _mm_add_ps(p14, p15);
                sum146 = _mm_add_ps(sum146, p16);

                __m128 sum1 = _mm_shuffle_ps(sum102, sum146, _MM_SHUFFLE(2, 0, 2, 0));

                _mm_storeu_ps(out_ptr, _mm_mul_ps(_mm_add_ps(sum0, sum1), scalar_016));

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
            for (int j = 0; j < block_w; j++)
            {
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 1);
                __m128 p02 = _mm_loadu_ps(line0 + 2);

                __m128 p04 = _mm_loadu_ps(line0 + 4);
                __m128 p05 = _mm_loadu_ps(line0 + 5);
                __m128 p06 = _mm_loadu_ps(line0 + 6);

                __m128 sum002 = _mm_add_ps(p00, p01);
                sum002 = _mm_add_ps(sum002, p02);
                __m128 sum046 = _mm_add_ps(p04, p05);
                sum046 = _mm_add_ps(sum046, p06);

                __m128 sum0 = _mm_shuffle_ps(sum002, sum046, _MM_SHUFFLE(2, 0, 2, 0));

                _mm_storeu_ps(out_ptr, _mm_mul_ps(sum0, scalar_033));

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

    __m128 scalar_011 = _mm_set1_ps(0.11111111f);
    __m128 scalar_016 = _mm_set1_ps(0.16666667f);
    __m128 scalar_033 = _mm_set1_ps(0.3333333f);

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
        for (int j = 0; j < block_w; j++)
        {
            __m128 p00 = _mm_loadu_ps(line1);
            __m128 p01 = _mm_loadu_ps(line1 + 1);
            __m128 p02 = _mm_loadu_ps(line1 + 2);

            __m128 p04 = _mm_loadu_ps(line1 + 4);
            __m128 p05 = _mm_loadu_ps(line1 + 5);
            __m128 p06 = _mm_loadu_ps(line1 + 6);

            __m128 sum002 = _mm_add_ps(p00, p01);
            sum002 = _mm_add_ps(sum002, p02);
            __m128 sum046 = _mm_add_ps(p04, p05);
            sum046 = _mm_add_ps(sum046, p06);

            __m128 sum0 = _mm_shuffle_ps(sum002, sum046, _MM_SHUFFLE(2, 0, 2, 0));

            __m128 p10 = _mm_loadu_ps(line2);
            __m128 p11 = _mm_loadu_ps(line2 + 1);
            __m128 p12 = _mm_loadu_ps(line2 + 2);

            __m128 p14 = _mm_loadu_ps(line2 + 4);
            __m128 p15 = _mm_loadu_ps(line2 + 5);
            __m128 p16 = _mm_loadu_ps(line2 + 6);

            __m128 sum102 = _mm_add_ps(p10, p11);
            sum102 = _mm_add_ps(sum102, p12);
            __m128 sum146 = _mm_add_ps(p14, p15);
            sum146 = _mm_add_ps(sum146, p16);

            __m128 sum1 = _mm_shuffle_ps(sum102, sum146, _MM_SHUFFLE(2, 0, 2, 0));
            if (is_caffe == 0)
                sum1 = _mm_mul_ps(_mm_add_ps(sum1, sum0), scalar_016);
            else
                sum1 = _mm_mul_ps(_mm_add_ps(sum1, sum0), scalar_011);
            _mm_storeu_ps(out_ptr, sum1);

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
            for (int j = 0; j < block_w; j++)
            {
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 1);
                __m128 p02 = _mm_loadu_ps(line0 + 2);

                __m128 p04 = _mm_loadu_ps(line0 + 4);
                __m128 p05 = _mm_loadu_ps(line0 + 5);
                __m128 p06 = _mm_loadu_ps(line0 + 6);

                __m128 sum002 = _mm_add_ps(p00, p01);
                sum002 = _mm_add_ps(sum002, p02);
                __m128 sum046 = _mm_add_ps(p04, p05);
                sum046 = _mm_add_ps(sum046, p06);

                __m128 sum0 = _mm_shuffle_ps(sum002, sum046, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 p10 = _mm_loadu_ps(line1);
                __m128 p11 = _mm_loadu_ps(line1 + 1);
                __m128 p12 = _mm_loadu_ps(line1 + 2);

                __m128 p14 = _mm_loadu_ps(line1 + 4);
                __m128 p15 = _mm_loadu_ps(line1 + 5);
                __m128 p16 = _mm_loadu_ps(line1 + 6);

                __m128 sum102 = _mm_add_ps(p10, p11);
                sum102 = _mm_add_ps(sum102, p12);
                __m128 sum146 = _mm_add_ps(p14, p15);
                sum146 = _mm_add_ps(sum146, p16);

                __m128 sum1 = _mm_shuffle_ps(sum102, sum146, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 p20 = _mm_loadu_ps(line2);
                __m128 p21 = _mm_loadu_ps(line2 + 1);
                __m128 p22 = _mm_loadu_ps(line2 + 2);

                __m128 p24 = _mm_loadu_ps(line2 + 4);
                __m128 p25 = _mm_loadu_ps(line2 + 5);
                __m128 p26 = _mm_loadu_ps(line2 + 6);

                __m128 sum202 = _mm_add_ps(p20, p21);
                sum202 = _mm_add_ps(sum202, p22);
                __m128 sum246 = _mm_max_ps(p24, p25);
                sum246 = _mm_add_ps(sum246, p26);

                __m128 sum2 = _mm_shuffle_ps(sum202, sum246, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 avg = _mm_mul_ps(_mm_add_ps(_mm_add_ps(sum0, sum1), sum2), scalar_011);
                _mm_storeu_ps(out_ptr, avg);

                line0 += 8;
                line1 += 8;
                line2 += 8;
                out_ptr += 4;
            }
            for (int j = block_w * 4 + 1; j < outw; j++)
            {
                *out_ptr =
                    (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2] + line2[0] + line2[1] + line2[2]) *
                    0.11111111f;
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

            for (int j = 0; j < block_w; j++)
            {
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 1);
                __m128 p02 = _mm_loadu_ps(line0 + 2);

                __m128 p04 = _mm_loadu_ps(line0 + 4);
                __m128 p05 = _mm_loadu_ps(line0 + 5);
                __m128 p06 = _mm_loadu_ps(line0 + 6);

                __m128 sum002 = _mm_add_ps(p00, p01);
                sum002 = _mm_add_ps(sum002, p02);
                __m128 sum046 = _mm_add_ps(p04, p05);
                sum046 = _mm_add_ps(sum046, p06);

                __m128 sum0 = _mm_shuffle_ps(sum002, sum046, _MM_SHUFFLE(2, 0, 2, 0));

                __m128 p10 = _mm_loadu_ps(line1);
                __m128 p11 = _mm_loadu_ps(line1 + 1);
                __m128 p12 = _mm_loadu_ps(line1 + 2);

                __m128 p14 = _mm_loadu_ps(line1 + 4);
                __m128 p15 = _mm_loadu_ps(line1 + 5);
                __m128 p16 = _mm_loadu_ps(line1 + 6);

                __m128 sum102 = _mm_add_ps(p10, p11);
                sum102 = _mm_add_ps(sum102, p12);
                __m128 sum146 = _mm_add_ps(p14, p15);
                sum146 = _mm_add_ps(sum146, p16);

                __m128 sum1 = _mm_shuffle_ps(sum102, sum146, _MM_SHUFFLE(2, 0, 2, 0));
                if (is_caffe == 0)
                    sum1 = _mm_mul_ps(_mm_add_ps(sum1, sum0), scalar_016);
                else
                    sum1 = _mm_mul_ps(_mm_add_ps(sum1, sum0), scalar_011);
                _mm_storeu_ps(out_ptr, sum1);

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

            for (int j = 0; j < block_w; j++)
            {
                __m128 p00 = _mm_loadu_ps(line0);
                __m128 p01 = _mm_loadu_ps(line0 + 1);
                __m128 p02 = _mm_loadu_ps(line0 + 2);

                __m128 p04 = _mm_loadu_ps(line0 + 4);
                __m128 p05 = _mm_loadu_ps(line0 + 5);
                __m128 p06 = _mm_loadu_ps(line0 + 6);

                __m128 sum002 = _mm_add_ps(p00, p01);
                sum002 = _mm_add_ps(sum002, p02);
                __m128 sum046 = _mm_add_ps(p04, p05);
                sum046 = _mm_add_ps(sum046, p06);

                __m128 sum0 = _mm_shuffle_ps(sum002, sum046, _MM_SHUFFLE(2, 0, 2, 0));

                _mm_storeu_ps(out_ptr, _mm_mul_ps(sum0, scalar_016));

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
            __m128 p00 = _mm_loadu_ps(line0);
            __m128 p01 = _mm_loadu_ps(line0 + 4);
            p00 = _mm_add_ps(p00, p01);

            float* p00_ptr = (float*)(&p00);

            sum += (p00_ptr[0] + p00_ptr[1] + p00_ptr[2] + p00_ptr[3]);
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
        __m128 p00 = _mm_loadu_ps(line0);
        __m128 res = p00;
        for (int j = 0; j < block; j++)
        {
            __m128 p00 = _mm_loadu_ps(line0);
            __m128 p01 = _mm_loadu_ps(line0 + 4);
            __m128 max0 = _mm_max_ps(p00, p01);
            res = _mm_max_ps(res, max0);
            line0 += 8;
        }

        float* res_ptr = (float*)(&res);

        float max_ = fmax(fmax(res_ptr[0], res_ptr[1]), fmax(res_ptr[2], res_ptr[3]));
        for (int j = tail; j < in_hw; j++)
        {
            max_ = fmax(max_, line0[0]);
            line0++;
        }
        *out_ptr = max_;
    }
}

static int pooling_kernel_perf_prerun(struct tensor* input, struct tensor* out, struct pool_param* param)
{
    int pool_size = POOL_GENERIC;

    /* global pooling */
    if (param->global)
    {
        if (param->pool_method == POOL_AVG)
            param->funct = ( pooling_kernel_t )avg_global;
        else if (param->pool_method == POOL_MAX)
            param->funct = ( pooling_kernel_t )max_global;

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
                    param->funct = ( pooling_kernel_t )max_2x2s2;
                else if (pool_size == POOL_K3S2)
                    param->funct = ( pooling_kernel_t )max_3x3s2;
            }
            else if (param->pad_h0 == 1)
            {
                if (pool_size == POOL_K2S2)
                    param->funct = ( pooling_kernel_t )max_2x2s2_p1;
                else if (pool_size == POOL_K3S2)
                    param->funct = ( pooling_kernel_t )max_3x3s2_p1;
                else if (pool_size == POOL_K3S1)
                    param->funct = ( pooling_kernel_t )max_3x3s1_p1;
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
                    param->funct = ( pooling_kernel_t )avg_2x2s2;
                else if (pool_size == POOL_K3S2)
                    param->funct = ( pooling_kernel_t )avg_3x3s2;
            }
            else if (param->pad_h0 == 1 && param->pad_h1 == 1)
            {
                if (pool_size == POOL_K2S2)
                    param->funct = ( pooling_kernel_t )avg_2x2s2_p1;
                else if (pool_size == POOL_K3S2)
                    param->funct = ( pooling_kernel_t )avg_3x3s2_p1;
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

static int pooling_kernel_perf_run(struct tensor* input, struct tensor* output, struct pool_param* param, int num_thread)
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
        void* input_frame = (uint8_t*)input->data + n * img_size * input->elem_size;
        void* output_frame = (uint8_t*)output->data + n * feature_size * output->elem_size;

#pragma omp parallel for num_threads(num_thread)
        for (int ch = 0; ch < c; ch++)
        {
            void* cur_input = (uint8_t*)input_frame + ch * in_h * in_w * input->elem_size;
            void* cur_output = (uint8_t*)output_frame + ch * out_h * out_w * output->elem_size;
            kernel(cur_input, cur_output, 1, in_h, in_w, out_h, out_w, param->kernel_h, param->kernel_w,
                   param->stride_h, param->stride_w, param->pad_h0, param->pad_w0, param->pad_h1, param->pad_w1,
                   is_caffe);
        }
    }

    return 0;
}
