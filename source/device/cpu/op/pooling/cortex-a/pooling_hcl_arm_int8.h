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
#include <string.h>

#include <arm_neon.h>

#define POOL_GENERIC 0
#define POOL_K2S2    1
#define POOL_K3S2    2
#define POOL_K3S1    3

static inline int8_t arm_max_int8(int8_t a, int8_t b)
{
    if (a > b)
        return a;
    else
        return b;
}

static inline int8_t arm_min_int8(int8_t a, int8_t b)
{
    if (a > b)
        return b;
    else
        return a;
}

typedef void (*pooling_kernel_int8_t)(const void* input, void* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                                      int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe, float in_scale, float out_scale);

static void pad_0_align_2D_int8(int8_t* dst, int8_t* src, int m, int n, int m_align, int n_align, int pad_h, int pad_w)
{
    int i;
    if (n >= n_align && m >= m_align)
    {
        memcpy(dst, src, m * n * sizeof(int8_t));
        return;
    }
    for (i = 0; i < m; ++i)
    {
        memcpy(dst + (i + pad_h) * n_align + pad_w, src + i * n, n * sizeof(int8_t));
    }
}

// pad 0 in right and down side on 3D
static void pad_0_align_3D_int8(int8_t* dst, int8_t* src, int m, int n, int m_align, int n_align, int c, int pad_h, int pad_w)
{
    int i;
    if (n >= n_align && m >= m_align)
    {
        memcpy(dst, src, c * m * n * sizeof(int8_t));
        return;
    }
    for (i = 0; i < c; ++i)
    {
        pad_0_align_2D_int8(dst + i * m_align * n_align, src + i * m * n, m, n, m_align, n_align, pad_h, pad_w);
    }
}

static void delete_0_2D_int8(int8_t* dst, int8_t* src, int m_align, int n_align, int m, int n, int pad_h, int pad_w)
{
    int i;
    if (n >= n_align && m >= m_align)
    {
        memcpy(dst, src, m * n * sizeof(int8_t));
        return;
    }
    for (i = 0; i < m; ++i)
    {
        memcpy(dst + i * n, src + (i + pad_h) * n_align + pad_w, n * sizeof(int8_t));
    }
}

// pad 0 in right and down side on 3D
static void delete_0_3D_int8(int8_t* dst, int8_t* src, int m_align, int n_align, int m, int n, int c, int pad_h, int pad_w)
{
    int i;
    if (n >= n_align && m >= m_align)
    {
        memcpy(dst, src, c * m * n * sizeof(int8_t));
        return;
    }
    for (i = 0; i < c; ++i)
    {
        delete_0_2D_int8(dst + i * m * n, src + i * m_align * n_align, m_align, n_align, m, n, pad_h, pad_w);
    }
}

static void avg_2x2s2_int8(const int8_t* input, int8_t* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                           int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe, float in_scale, float out_scale)
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
    int block_w = outw >> 3;
    int remain_w = inw - outw * 2;
    int index = 0;

    for (int c = 0; c < inc; c++)
    {
        index = 0;
        const int8_t* line0 = input + c * in_hw;
        const int8_t* line1 = line0 + inw;
        int8_t* out_ptr = output + c * out_hw;
        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < block_w; j++)
            {
                int8x8_t p00 = vld1_s8(line0);
                int8x8_t p10 = vld1_s8(line1);
                int16x8_t sum0 = vaddl_s8(p00, p10);

                int8x8_t p01 = vld1_s8(line0 + 8);
                int8x8_t p11 = vld1_s8(line1 + 8);
                int16x8_t sum1 = vaddl_s8(p01, p11);
#ifdef __aarch64__
                /* pairwaise max */
                sum0 = vpaddq_s16(sum0, sum1);
                for (int n = 0; n < 8; n++)
                {
                    out_ptr[n] = (int8_t)round(sum0[n] / 4);
                }
#else
                /* pairwaise max */
                int32x4_t suml0 = vpaddlq_s16(sum0);
                int32x4_t suml1 = vpaddlq_s16(sum1);
                for (int n = 0; n < 4; n++)
                {
                    out_ptr[n] = (int8_t)round(suml0[n] / 4);
                    out_ptr[n + 1] = (int8_t)round(suml1[n] / 4);
                }
#endif
                line0 += 16;
                out_ptr = out_ptr + 8;
                index = index + 8;
            }
            index = block_w * 8;
            if (outw - index >= 4)
            {
                int8x8_t p00 = vld1_s8(line0);
                int8x8_t p10 = vld1_s8(line1);
                int16x8_t sum0 = vaddl_s8(p00, p10);
#ifdef __aarch64__
                /* pairwaise max */
                int16x8_t sum1 = {0};
                sum0 = vpaddq_s16(sum0, sum1);
                for (int n = 0; n < 4; n++)
                {
                    out_ptr[n] = (int8_t)round(sum0[n] / 4);
                }
#else
                /* pairwaise max */
                int32x4_t suml0 = vpaddlq_s16(sum0);
                for (int n = 0; n < 4; n++)
                {
                    out_ptr[n] = (int8_t)round(suml0[n] / 4);
                }
#endif
                line0 += 8;
                out_ptr = out_ptr + 4;
                index = index + 4;
            }
            for (; index < outw; index++)
            {
                *out_ptr = (int8_t)round((line0[0] + line0[1] + line1[0] + line1[1]) / 4);
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if (pad_w1 > 0)
            {
                *out_ptr = (int8_t)round((line0[0] + line1[0]) / 2);
                out_ptr++;
            }

            line0 += remain_w + inw;
            line1 += remain_w + inw;
        }
        if (pad_h1)
        {
            index = 0;
            for (int j = 0; j < block_w; j++)
            {
                int8x8_t p00 = vld1_s8(line0);
                int8x8_t p01 = vld1_s8(line0 + 8);

                int8x8_t p02 = {0};

                /* pairwaise max */
                int16x8_t sum0 = vaddl_s8(p00, p02);
                int16x8_t sum1 = vaddl_s8(p01, p02);
#ifdef __aarch64__
                sum0 = vpaddq_s16(sum0, sum1);
                for (int n = 0; n < 8; n++)
                {
                    out_ptr[n] = (int8_t)round(sum0[n] / 4);
                }
#else
                int32x4_t suml0 = vpaddlq_s16(sum0);
                int32x4_t suml1 = vpaddlq_s16(sum1);
                for (int n = 0; n < 4; n++)
                {
                    out_ptr[n] = (int8_t)round(suml0[n] / 4);
                    out_ptr[n + 1] = (int8_t)round(suml1[n] / 4);
                }
#endif
                line0 += 16;
                out_ptr = out_ptr + 8;
                index = index + 8;
            }
            index = block_w * 8;
            if (outw - index >= 4)
            {
                int8x8_t p00 = vld1_s8(line0);
                int8x8_t p01 = {0};
                int16x8_t sum0 = vaddl_s8(p00, p01);
#ifdef __aarch64__
                /* pairwaise max */
                int16x8_t sum1 = {0};
                sum0 = vpaddq_s16(sum0, sum1);
                for (int n = 0; n < 4; n++)
                {
                    out_ptr[n] = (int8_t)round(sum0[n] / 4);
                }
#else
                /* pairwaise max */
                int32x4_t suml0 = vpaddlq_s16(sum0);
                for (int n = 0; n < 4; n++)
                {
                    out_ptr[n] = (int8_t)round(suml0[n] / 4);
                }
#endif
                line0 += 8;
                out_ptr = out_ptr + 4;
                index = index + 4;
            }
            for (; index < outw; index++)
            {
                int sum0 = line0[0] + line0[1];
                *out_ptr = (int8_t)round((sum0) / 2);
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if (pad_w1 > 0)
            {
                *out_ptr = line0[0];
                out_ptr++;
            }
        }
    }
}

static void max_2x2s2_int8(const int8_t* input, int8_t* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                           int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe, float in_scale, float out_scale)
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
#ifdef __aarch64__
    int block_w = outw >> 4;
#else
    int block_w = outw >> 3;
#endif
    int remain_w = inw - outw * 2;
    int index = 0;
    for (int c = 0; c < inc; c++)
    {
        const int8_t* line0 = input + c * in_hw;
        const int8_t* line1 = line0 + inw;
        int8_t* out_ptr = output + c * out_hw;
        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < block_w; j++)
            {
#ifdef __aarch64__
                int8x16_t p00 = vld1q_s8(line0);
                int8x16_t p10 = vld1q_s8(line1);
                int8x16_t max0 = vmaxq_s8(p00, p10);

                int8x16_t p01 = vld1q_s8(line0 + 16);
                int8x16_t p11 = vld1q_s8(line1 + 16);
                int8x16_t max1 = vmaxq_s8(p01, p11);

                /* pairwaise max */
                int8x16_t _max = vpmaxq_s8(max0, max1);
                vst1q_s8(out_ptr, _max);
                line0 += 32;
                line1 += 32;
                out_ptr += 16;
            }
            index = block_w * 16;
#else
                int8x8_t p00 = vld1_s8(line0);
                int8x8_t p10 = vld1_s8(line1);
                int8x8_t max0 = vmax_s8(p00, p10);

                int8x8_t p01 = vld1_s8(line0 + 8);
                int8x8_t p11 = vld1_s8(line1 + 8);
                int8x8_t max1 = vmax_s8(p01, p11);

                /* pairwaise max */
                int8x8_t _max = vpmax_s8(max0, max1);
                vst1_s8(out_ptr, _max);
                line0 += 16;
                line1 += 16;
                out_ptr += 8;
            }
            index = block_w * 8;
#endif
            if (outw - index >= 8)
            {
                int8x8_t p00 = vld1_s8(line0);
                int8x8_t p10 = vld1_s8(line1);
                int8x8_t max0 = vmax_s8(p00, p10);

                int8x8_t p01 = vld1_s8(line0 + 8);
                int8x8_t p11 = vld1_s8(line1 + 8);
                int8x8_t max1 = vmax_s8(p01, p11);

                /* pairwaise max */
                int8x8_t _max = vpmax_s8(max0, max1);
                vst1_s8(out_ptr, _max);
                line0 += 16;
                line1 += 16;
                out_ptr = out_ptr + 8;
                index = index + 8;
            }
            if (outw - index >= 4)
            {
                int8x8_t p00 = vld1_s8(line0);
                int8x8_t p10 = vld1_s8(line1);
                int8x8_t max0 = vmax_s8(p00, p10);
                /* pairwaise max */
                int8x8_t max1 = {0};
                int8x8_t _max = vpmax_s8(max0, max1);

                out_ptr[0] = _max[0];
                out_ptr[1] = _max[1];
                out_ptr[2] = _max[2];
                out_ptr[3] = _max[3];

                line0 += 8;
                line1 += 8;
                out_ptr = out_ptr + 4;
                index = index + 4;
            }
            for (; index < outw; index++)
            {
                int8_t max0 = arm_max_int8(line0[0], line0[1]);
                int8_t max1 = arm_max_int8(line1[0], line1[1]);
                *out_ptr = arm_max_int8(max0, max1);

                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if (pad_w1 > 0)
            {
                *out_ptr = arm_max_int8(line0[0], line1[0]);
                out_ptr++;
            }
            line0 += remain_w + inw;
            line1 += remain_w + inw;
        }
        if (pad_h1 > 0)
        {
            for (int j = 0; j < block_w; j++)
            {
#ifdef __aarch64__
                int8x16_t p00 = vld1q_s8(line0);
                int8x16_t p01 = vld1q_s8(line0 + 16);

                /* pairwaise max */
                int8x16_t _max = vpmaxq_s8(p00, p01);
                vst1q_s8(out_ptr, _max);
                line0 += 32;
                out_ptr += 16;
            }
            index = block_w * 16;
#else
                int8x8_t p00 = vld1_s8(line0);
                int8x8_t p01 = vld1_s8(line0 + 8);

                /* pairwaise max */
                int8x8_t _max = vpmax_s8(p00, p01);
                vst1_s8(out_ptr, _max);
                line0 += 16;
                out_ptr += 8;
            }
            index = block_w * 8;
#endif
            if (outw - index >= 8)
            {
                int8x8_t p00 = vld1_s8(line0);
                int8x8_t p01 = vld1_s8(line0 + 8);

                /* pairwaise max */
                int8x8_t _max = vpmax_s8(p00, p01);
                vst1_s8(out_ptr, _max);
                line0 += 16;
                out_ptr = out_ptr + 8;
                index = index + 8;
            }
            if (outw - index >= 4)
            {
                int8x8_t p00 = vld1_s8(line0);
                /* pairwaise max */
                int8x8_t p01 = {0};
                int8x8_t _max = vpmax_s8(p00, p01);

                out_ptr[0] = _max[0];
                out_ptr[1] = _max[1];
                out_ptr[2] = _max[2];
                out_ptr[3] = _max[3];

                line0 += 8;
                out_ptr = out_ptr + 4;
                index = index + 4;
            }
            for (; index < outw; index++)
            {
                *out_ptr = arm_max_int8(line0[0], line0[1]);
                out_ptr++;
                line0 += 2;
            }
            if (pad_w1 > 0)
            {
                *out_ptr = arm_max_int8(line0[0], line1[0]);
                out_ptr++;
            }
        }
    }
}

static void avg_3x3s2_int8(const int8_t* input, int8_t* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                           int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe, float in_scale, float out_scale)
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
    int block_w = outw >> 3;
    int remain_w = inw - outw * 2;
    int index = 0;
    for (int c = 0; c < inc; c++)
    {
        const int8_t* line0 = input + c * in_hw;
        const int8_t* line1 = line0 + inw;
        const int8_t* line2 = line1 + inw;
        int8_t* out_ptr = output + c * out_hw;
        for (int i = 0; i < outh; i++)
        {
            index = 0;
            for (int j = 0; j < block_w; j++)
            {
                int8x8x2_t p00 = vld2_s8(line0);
                int8x8x2_t p10 = vld2_s8(line1);
                int8x8x2_t p20 = vld2_s8(line2);

                int8x8x2_t p00_new = vld2_s8(line0 + 16);
                int16x8_t sum0 = vaddl_s8(p00.val[0], p00.val[1]);
                int8x8_t p01 = vext_s8(p00.val[0], p00_new.val[0], 1);
                sum0 = vaddw_s8(sum0, p01);

                int8x8x2_t p10_new = vld2_s8(line1 + 16);
                sum0 = vaddw_s8(sum0, p10.val[0]);
                sum0 = vaddw_s8(sum0, p10.val[1]);
                int8x8_t p11 = vext_s8(p10.val[0], p10_new.val[0], 1);
                sum0 = vaddw_s8(sum0, p11);

                int8x8x2_t p20_new = vld2_s8(line2 + 16);
                sum0 = vaddw_s8(sum0, p20.val[0]);
                sum0 = vaddw_s8(sum0, p20.val[1]);
                int8x8_t p21 = vext_s8(p20.val[0], p20_new.val[0], 1);
                sum0 = vaddw_s8(sum0, p21);

                // sum0 = vadd_s8(vadd_s8(sum0, sum1), sum2);

                for (int n = 0; n < 8; n++)
                {
                    out_ptr[n] = (int8_t)round(sum0[n] / 9);
                }

                p00 = p00_new;
                p10 = p10_new;
                p20 = p20_new;

                line0 += 16;
                line1 += 16;
                line2 += 16;

                out_ptr += 8;
                index = index + 8;
            }

            for (; index < outw; index++)
            {
                int sum = (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2] + line2[0] + line2[1] + line2[2]);
                *out_ptr = (int8_t)round(sum / 9);
                out_ptr++;
                line0 += 2;
                line1 += 2;
                line2 += 2;
            }
            if (pad_w1 == 1)
            {
                int sum = (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2]);
                *out_ptr = (int8_t)round(sum / 6);
                out_ptr++;
            }
            else if (pad_w1 == 2)
            {
                int sum = (line0[0] + line1[0] + line2[0]);
                *out_ptr = (int8_t)round(sum / 6);
                out_ptr++;
            }
            line0 += remain_w + inw;
            line1 += remain_w + inw;
            line2 += remain_w + inw;
        }
        if (pad_h1 == 1)
        {
            index = 0;
            for (int j = 0; j < block_w; j++)
            {
                int8x8x2_t p00 = vld2_s8(line0);
                int8x8x2_t p10 = vld2_s8(line1);

                int8x8x2_t p00_new = vld2_s8(line0 + 16);
                int16x8_t sum0 = vaddl_s8(p00.val[0], p00.val[1]);
                int8x8_t p01 = vext_s8(p00.val[0], p00_new.val[0], 1);
                sum0 = vaddw_s8(sum0, p01);

                int8x8x2_t p10_new = vld2_s8(line1 + 16);
                sum0 = vaddw_s8(sum0, p10.val[0]);
                sum0 = vaddw_s8(sum0, p10.val[1]);
                int8x8_t p11 = vext_s8(p10.val[0], p10_new.val[0], 1);
                sum0 = vaddw_s8(sum0, p11);

                for (int n = 0; n < 8; n++)
                {
                    out_ptr[n] = (int8_t)round(sum0[n] / 6);
                }

                p00 = p00_new;
                p10 = p10_new;
                line0 += 16;
                line1 += 16;
                out_ptr += 8;
                index = index + 8;
            }
            for (; index < outw; index++)
            {
                int sum = (line0[0] + line0[1] + line0[2] + line1[0] + line1[1] + line1[2]);
                *out_ptr = (int8_t)round(sum / 6);
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if (pad_w1 == 1)
            {
                int sum = (line0[0] + line0[1] + line1[0] + line1[1]);
                *out_ptr = (int8_t)round(sum / 4);
                out_ptr++;
            }
            else if (pad_w1 == 2)
            {
                int sum = (line0[0] + line1[0]);
                *out_ptr = (int8_t)round(sum / 2);
                out_ptr++;
            }
        }
        else if (pad_h1 == 2)
        {
            index = 0;
            for (int j = 0; j < block_w; j++)
            {
                int8x8x2_t p00 = vld2_s8(line0);
                int8x8x2_t p00_new = vld2_s8(line0 + 16);
                int16x8_t sum0 = vaddl_s8(p00.val[0], p00.val[1]);
                int8x8_t p01 = vext_s8(p00.val[0], p00_new.val[0], 1);
                sum0 = vaddw_s8(sum0, p01);

                for (int n = 0; n < 8; n++)
                {
                    out_ptr[n] = (int8_t)round(sum0[n] / 3);
                }

                p00 = p00_new;
                line0 += 16;
                out_ptr += 8;
                index = index + 8;
            }
            for (; index < outw; index++)
            {
                *out_ptr = (int8_t)round((line0[0] + line0[1] + line0[2]) / 3);
                out_ptr++;
                line0 += 2;
            }
            if (pad_w1 == 1)
            {
                *out_ptr = (int8_t)round((line0[0] + line0[1]) / 2);
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

static void max_3x3s2_int8(const int8_t* input, int8_t* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                           int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe, float in_scale, float out_scale)
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
    int block_w = outw >> 4;
    int remain_w = inw - outw * 2;

    int index = 0;

    for (int c = 0; c < inc; c++)
    {
        const int8_t* line0 = input + c * in_hw;
        const int8_t* line1 = line0 + inw;
        const int8_t* line2 = line1 + inw;
        int8_t* out_ptr = output + c * out_hw;
        for (int i = 0; i < outh; i++)
        {
            int8x16x2_t p00 = vld2q_s8(line0);
            int8x16x2_t p10 = vld2q_s8(line1);
            int8x16x2_t p20 = vld2q_s8(line2);
            for (int j = 0; j < block_w; j++)
            {
                /*
                p00     = [1,2,3,4,5,6,7,8...]
                p00.val[0]=[1,3,5,7...]

                max0    = [2,4,6,8...]
                p00_new = [9,10,11,12,13,14,15,16...]
                p01     = [3,5,7,9...]
                max0=max(max0,p01)=[3,5,7,9]
                */
                int8x16x2_t p00_new = vld2q_s8(line0 + 32);
                int8x16_t max0 = vmaxq_s8(p00.val[0], p00.val[1]);
                int8x16_t p01 = vextq_s8(p00.val[0], p00_new.val[0], 1);
                max0 = vmaxq_s8(max0, p01);

                int8x16x2_t p10_new = vld2q_s8(line1 + 32);
                int8x16_t max1 = vmaxq_s8(p10.val[0], p10.val[1]);
                int8x16_t p11 = vextq_s8(p10.val[0], p10_new.val[0], 1);
                max1 = vmaxq_s8(max1, p11);

                int8x16x2_t p20_new = vld2q_s8(line2 + 32);
                int8x16_t max2 = vmaxq_s8(p20.val[0], p20.val[1]);
                int8x16_t p21 = vextq_s8(p20.val[0], p20_new.val[0], 1);
                max2 = vmaxq_s8(max2, p21);

                max0 = vmaxq_s8(vmaxq_s8(max0, max1), max2);
                vst1q_s8(out_ptr, max0);

                p00 = p00_new;
                p10 = p10_new;
                p20 = p20_new;

                line0 += 32;
                line1 += 32;
                line2 += 32;
                out_ptr += 16;
            }

            index = block_w * 16;

            if (outw - index > 8)
            {
                int8x8x2_t p00 = vld2_s8(line0);
                int8x8x2_t p10 = vld2_s8(line1);
                int8x8x2_t p20 = vld2_s8(line2);

                int8x8x2_t p00_new = vld2_s8(line0 + 16);
                int8x8_t max0 = vmax_s8(p00.val[0], p00.val[1]);
                int8x8_t p01 = vext_s8(p00.val[0], p00_new.val[0], 1);
                max0 = vmax_s8(max0, p01);

                int8x8x2_t p10_new = vld2_s8(line1 + 16);
                int8x8_t max1 = vmax_s8(p10.val[0], p10.val[1]);
                int8x8_t p11 = vext_s8(p10.val[0], p10_new.val[0], 1);
                max1 = vmax_s8(max1, p11);

                int8x8x2_t p20_new = vld2_s8(line2 + 16);
                int8x8_t max2 = vmax_s8(p20.val[0], p20.val[1]);
                int8x8_t p21 = vext_s8(p20.val[0], p20_new.val[0], 1);
                max2 = vmax_s8(max2, p21);

                max0 = vmax_s8(vmax_s8(max0, max1), max2);
                vst1_s8(out_ptr, max0);

                p00 = p00_new;
                p10 = p10_new;
                p20 = p20_new;

                line0 += 16;
                line1 += 16;
                line2 += 16;
                out_ptr += 8;
                index = index + 8;
            }
            for (; index < outw; index++)
            {
                int8_t max0 = arm_max_int8(arm_max_int8(line0[0], line0[1]), line0[2]);
                int8_t max1 = arm_max_int8(arm_max_int8(line1[0], line1[1]), line1[2]);
                int8_t max2 = arm_max_int8(arm_max_int8(line2[0], line2[1]), line2[2]);
                *out_ptr = arm_max_int8(arm_max_int8(max0, max1), max2);

                out_ptr++;
                line0 += 2;
                line1 += 2;
                line2 += 2;
            }
            if (pad_w1 == 1)
            {
                int8_t max0 = arm_max_int8(arm_max_int8(line0[0], line0[1]), arm_max_int8(line1[0], line1[1]));
                *out_ptr = arm_max_int8(arm_max_int8(line2[0], line2[1]), max0);
                out_ptr++;
            }
            line0 += remain_w + inw;
            line1 += remain_w + inw;
            line2 += remain_w + inw;
        }
        if (pad_h1 == 1)
        {
            int8x16x2_t p00 = vld2q_s8(line0);
            int8x16x2_t p10 = vld2q_s8(line1);
            for (int j = 0; j < block_w; j++)
            {
                int8x16x2_t p00_new = vld2q_s8(line0 + 32);
                int8x16_t max0 = vmaxq_s8(p00.val[0], p00.val[1]);
                int8x16_t p01 = vextq_s8(p00.val[0], p00_new.val[0], 1);
                max0 = vmaxq_s8(max0, p01);

                int8x16x2_t p10_new = vld2q_s8(line1 + 32);
                int8x16_t max1 = vmaxq_s8(p10.val[0], p10.val[1]);
                int8x16_t p11 = vextq_s8(p10.val[0], p10_new.val[0], 1);
                max1 = vmaxq_s8(max1, p11);

                max0 = vmaxq_s8(max0, max1);
                vst1q_s8(out_ptr, max0);

                p00 = p00_new;
                p10 = p10_new;

                line0 += 32;
                line1 += 32;
                out_ptr += 16;
            }

            index = block_w * 16;

            if (outw - index > 8)
            {
                int8x8x2_t p00 = vld2_s8(line0);
                int8x8x2_t p10 = vld2_s8(line1);

                int8x8x2_t p00_new = vld2_s8(line0 + 16);
                int8x8_t max0 = vmax_s8(p00.val[0], p00.val[1]);
                int8x8_t p01 = vext_s8(p00.val[0], p00_new.val[0], 1);
                max0 = vmax_s8(max0, p01);

                int8x8x2_t p10_new = vld2_s8(line1 + 16);
                int8x8_t max1 = vmax_s8(p10.val[0], p10.val[1]);
                int8x8_t p11 = vext_s8(p10.val[0], p10_new.val[0], 1);
                max1 = vmax_s8(max1, p11);

                max0 = vmax_s8(max0, max1);
                vst1_s8(out_ptr, max0);

                p00 = p00_new;
                p10 = p10_new;

                line0 += 16;
                line1 += 16;
                out_ptr += 8;
                index = index + 8;
            }
            for (; index < outw; index++)
            {
                int8_t max0 = arm_max_int8(arm_max_int8(line0[0], line0[1]), line0[2]);
                int8_t max1 = arm_max_int8(arm_max_int8(line1[0], line1[1]), line1[2]);
                *out_ptr = arm_max_int8(max0, max1);
                out_ptr++;
                line0 += 2;
                line1 += 2;
            }
            if (pad_w1 == 1)
            {
                *out_ptr = arm_max_int8(arm_max_int8(line0[0], line0[1]), arm_max_int8(line1[0], line1[1]));
                out_ptr++;
            }
        }
    }
}

static void avg_global_int8(const int8_t* input, int8_t* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                            int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe, float in_scale, float out_scale)
{
    int in_hw = inw * inh;
    int block = in_hw >> 4;

    for (int c = 0; c < inc; c++)
    {
        int index = 0;
        const int8_t* line0 = input + c * in_hw;
        int8_t* out_ptr = output + c;
        int sum = 0;
        for (int j = 0; j < block; j++)
        {
            int8x8_t p00 = vld1_s8(line0);
            int8x8_t p01 = vld1_s8(line0 + 8);
            int16x8_t pls = vaddl_s8(p00, p01);
            int32x4_t tmp = vpaddlq_s16(pls);
            sum += vgetq_lane_s32(tmp, 0) + vgetq_lane_s32(tmp, 1) + vgetq_lane_s32(tmp, 2) + vgetq_lane_s32(tmp, 3);
            line0 += 16;
        }
        index = block * 16;

        for (int j = index; j < in_hw; j++)
        {
            sum += line0[0];
            line0++;
        }
        float sum_fp32 = sum * in_scale;
        sum_fp32 = sum_fp32 / in_hw;
        int tmp = (int)round(sum_fp32 / out_scale);
        if (tmp > 127)
            tmp = 127;
        else if (tmp < -127)
            tmp = -127;

        *out_ptr = (int8_t)tmp; //round(sum / in_hw);
    }
}

static void max_global_int8(const int8_t* input, int8_t* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                            int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe, float in_scale, float out_scale)
{
    int in_hw = inw * inh;
    int block = in_hw >> 5;

    for (int c = 0; c < inc; c++)
    {
        int index = 0;
        const int8_t* line0 = input + c * in_hw;
        int8_t* out_ptr = output + c;

        int8x16_t p00 = vld1q_s8(line0);
        int8x16_t res = p00;
        for (int j = 0; j < block; j++)
        {
            int8x16_t p00 = vld1q_s8(line0);
            int8x16_t p01 = vld1q_s8(line0 + 16);
            int8x16_t max0 = vmaxq_s8(p00, p01);
            res = vmaxq_s8(res, max0);
            line0 += 32;
        }
        int8_t max_ = 0;
        if (block > 0)
        {
            max_ = res[0];
#ifdef __aarch64__
            for (int n = 1; n < 16; n++)
            {
                max_ = arm_max_int8(max_, res[n]);
            }
#else
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 0));
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 1));
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 2));
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 3));
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 4));
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 5));
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 6));
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 7));
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 8));
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 9));
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 10));
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 11));
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 12));
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 13));
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 14));
            max_ = arm_max_int8(max_, vgetq_lane_s8(res, 15));
#endif
        }
        else
        {
            max_ = line0[0];
        }
        index = block * 32;
        for (int j = index; j < in_hw; j++)
        {
            max_ = arm_max_int8(max_, line0[0]);
            line0++;
        }
        *out_ptr = max_;
    }
}

int pooling_kernel_int8_perf_prerun(struct tensor* input, struct tensor* out, struct pool_param* param)
{
    int pool_size = POOL_GENERIC;

    /* global pooling */
    if (param->global)
    {
        if (param->pool_method == POOL_AVG)
            param->funct = (pooling_kernel_int8_t)avg_global_int8;
        else if (param->pool_method == POOL_MAX)
            param->funct = (pooling_kernel_int8_t)max_global_int8;

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

    /* general max pooling, k2s2, k2k2p1, k3s1p1, k3s2, k3s2p1 */
    if (param->pool_method == POOL_MAX)
    {
        if ((param->pad_h0 == param->pad_w0) && (param->pad_h1 == param->pad_w1))
        {
            if (pool_size == POOL_K2S2)
                param->funct = (pooling_kernel_int8_t)max_2x2s2_int8;
            else if (pool_size == POOL_K3S2)
                param->funct = (pooling_kernel_int8_t)max_3x3s2_int8;
        }
    }
    /* general avg pooling, k2s2, k2s2p1, k3s2, k3s2p1 */
    else if (param->pool_method == POOL_AVG)
    {
        if ((param->pad_h0 == param->pad_w0) && (param->pad_h1 == param->pad_w1))
        {
            if (pool_size == POOL_K2S2)
                param->funct = (pooling_kernel_int8_t)avg_2x2s2_int8;
            else if (pool_size == POOL_K3S2)
                param->funct = (pooling_kernel_int8_t)avg_3x3s2_int8;
        }
    }

    if (param->funct == NULL)
    {
        TLOG_ERR("perf pooling func not be find\n");
        return -1;
    }

    return 0;
}

int pooling_kernel_int8_perf_run(struct tensor* input, struct tensor* output, struct pool_param* param, int num_thread)
{
    int is_caffe = param->caffe_flavor;
    pooling_kernel_int8_t kernel = (pooling_kernel_int8_t)(param->funct);

    int batch = input->dims[0];
    int c = input->dims[1];
    int in_h = input->dims[2];
    int in_w = input->dims[3];

    int out_h = output->dims[2];
    int out_w = output->dims[3];

    int pad_h0 = param->pad_h0;
    int pad_h1 = param->pad_h1;
    int pad_w0 = param->pad_w0;
    int pad_w1 = param->pad_w1;

    int in_h_origin = in_h;
    int in_w_origin = in_w;
    int in_h_pad = in_h + pad_h0;
    int in_w_pad = in_w + pad_w0;

    int img_size = c * in_h * in_w;
    int feature_size = c * out_h * out_w;
    float input_scale = input->scale;
    float output_scale = output->scale;

    if (param->input_pad != NULL)
    {
        param->pad_h0 = 0;
        param->pad_w0 = 0;
        in_h += 1;
        in_w += 1;
    }

    for (int n = 0; n < batch; n++)
    {
        void* input_frame = input->data + n * img_size * input->elem_size;
        void* output_frame = output->data + n * feature_size * output->elem_size;

        if (param->input_pad != NULL)
        {
            pad_0_align_3D_int8((int8_t*)param->input_pad + n * c * in_h_pad * in_w_pad, (int8_t*)input_frame,
                                in_h_origin, in_w_origin, in_h_pad, in_w_pad, c, pad_h0, pad_w0);
        }

#pragma omp parallel for num_threads(num_thread)
        for (int ch = 0; ch < c; ch++)
        {
            void* cur_input = NULL;
            if (param->input_pad != NULL)
            {
                cur_input = param->input_pad + ch * in_h_pad * in_w_pad * input->elem_size;
            }
            else
            {
                cur_input = input_frame + ch * in_h * in_w * input->elem_size;
            }
            void* cur_output = output_frame + ch * out_h * out_w * output->elem_size;
            kernel(cur_input, cur_output, 1, in_h, in_w, out_h, out_w, param->kernel_h, param->kernel_w,
                   param->stride_h, param->stride_w, param->pad_h0, param->pad_w0, param->pad_h1, param->pad_w1,
                   is_caffe, input_scale, output_scale);
        }
    }

    return 0;
}
