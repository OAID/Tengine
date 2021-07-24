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
 * Author: quanwang@openailab.com
 */

#include "conv_kernel_x86.h"

#include "wino_conv_kernel_x86.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if __AVX__
#include <immintrin.h>
#endif

#ifndef _MSC_VER
#include <sys/time.h>
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#endif

static int get_private_mem_size(struct tensor* filter)
{
    if (filter->data_type == TENGINE_DT_UINT8)    // simulator uint8 inference with fp32
        return filter->elem_num * filter->elem_size * 4;
    else
        return filter->elem_num * filter->elem_size;    // caution
}

static void interleave(struct tensor* filter, struct conv_priv_info* priv_info)
{
    /* simply copy the data */
    memcpy(priv_info->interleave_buffer, filter->data, filter->elem_num * filter->elem_size);
}

static void interleave_uint8(struct tensor* filter, struct conv_priv_info* priv_info)
{
    /* dequant uint8 weight to fp32 for simulator */
    float* weight_fp32 = (float* )priv_info->interleave_buffer;
    uint8_t* weight_uint8 = (uint8_t*)filter->data;
    float scale = filter->scale;
    int zero_point = filter->zero_point;

    for (int i = 0; i < filter->elem_num; i++)
    {
        weight_fp32[i] = ((float)weight_uint8[i] - (float)zero_point) * scale;
    }
}

void im2col_fp32(float* data_img, float* data_col, int inh, int inw, int inc, int outh, int outw, int ksize_h,
            int ksize_w, int sh, int sw, int ph, int pw, int dh, int dw)
{
    const int channels_col = ksize_h * ksize_w * inc;

    for (int c = 0; c < channels_col; ++c)
    {
        const int kw = c % ksize_w;
        int c_ = c / ksize_w;
        const int kh = c_ % ksize_h;
        c_ = c_ / ksize_h;
        const int im_col = kw * dw - pw;
        const int w_low = max(0, -im_col / sw + (-im_col % sw > 0));
        const int w_high = min(outw, (inw - im_col) / sw + ((inw - im_col) % sw > 0));

        for (int h = 0; h < outh; ++h)
        {
            const int im_row = kh * dh + h * sh - ph;
            float* out = data_col + (c * outh + h) * outw;
            const float* end = out + w_high;

            if (im_row >= 0 && im_row < inh)
            {
                float* in = data_img + inw * (im_row + inh * c_) + im_col + (w_low - 1) * sw;

                memset(out, 0, w_low * sizeof(float));
                out += w_low;
                while (out < end)
                {
                    in += sw;
                    *(out++) = *in;
                }
                memset(out, 0, (outw - w_high) * sizeof(float));
            }
            else
            {
                memset(out, 0, outw * sizeof(float));
            }
        }
    }
}

void im2col_uint8(uint8_t* data_img, float* data_col, struct tensor* input_tensor, struct tensor* output_tensor, struct conv_param* param)
{
    int ksize_h = param->kernel_h;
    int ksize_w = param->kernel_w;

    int inc = param->input_channel / param->group;
    int sh = param->stride_h;
    int sw = param->stride_w;
    int ph = param->pad_h0;
    int pw = param->pad_w0;
    int dh = param->dilation_h;
    int dw = param->dilation_w;

    int inh = input_tensor->dims[2];
    int inw = input_tensor->dims[3];
    int outh = output_tensor->dims[2];
    int outw = output_tensor->dims[3];

    float scale = input_tensor->scale;
    int zero_point = input_tensor->zero_point;

    const int channels_col = ksize_h * ksize_w * inc;

    for (int c = 0; c < channels_col; ++c)
    {
        const int kw = c % ksize_w;
        int c_ = c / ksize_w;
        const int kh = c_ % ksize_h;
        c_ = c_ / ksize_h;
        const int im_col = kw * dw - pw;
        const int w_low = max(0, -im_col / sw + (-im_col % sw > 0));
        const int w_high = min(outw, (inw - im_col) / sw + ((inw - im_col) % sw > 0));

        for (int h = 0; h < outh; ++h)
        {
            const int im_row = kh * dh + h * sh - ph;
            float* out = data_col + (c * outh + h) * outw;
            const float* end = out + w_high;

            if (im_row >= 0 && im_row < inh)
            {
                uint8_t * in = data_img + inw * (im_row + inh * c_) + im_col + (w_low - 1) * sw;

                memset(out, 0, w_low * sizeof(float));
                out += w_low;
                while (out < end)
                {
                    in += sw;

                    float in_fp32 = ((float)in[0] - (float)zero_point) * scale;
                    out[0] = in_fp32;
                    out++;
                }
                memset(out, 0, (outw - w_high) * sizeof(float));
            }
            else
            {
                memset(out, 0, outw * sizeof(float));
            }
        }
    }
}

void im2col_int8(int8_t* data_img, int8_t* data_col, struct tensor* input_tensor, struct tensor* output_tensor, struct conv_param* param)
{
    int ksize_h = param->kernel_h;
    int ksize_w = param->kernel_w;

    int inc = param->input_channel / param->group;
    int sh = param->stride_h;
    int sw = param->stride_w;
    int ph = param->pad_h0;
    int pw = param->pad_w0;
    int dh = param->dilation_h;
    int dw = param->dilation_w;

    int inh = input_tensor->dims[2];
    int inw = input_tensor->dims[3];
    int outh = output_tensor->dims[2];
    int outw = output_tensor->dims[3];

    const int channels_col = ksize_h * ksize_w * inc;

    for (int c = 0; c < channels_col; ++c)
    {
        const int kw = c % ksize_w;
        int c_ = c / ksize_w;
        const int kh = c_ % ksize_h;
        c_ = c_ / ksize_h;
        const int im_col = kw * dw - pw;
        const int w_low = max(0, -im_col / sw + (-im_col % sw > 0));
        const int w_high = min(outw, (inw - im_col) / sw + ((inw - im_col) % sw > 0));

        for (int h = 0; h < outh; ++h)
        {
            const int im_row = kh * dh + h * sh - ph;
            int8_t * out = data_col + (c * outh + h) * outw;
            const int8_t * end = out + w_high;

            if (im_row >= 0 && im_row < inh)
            {
                int8_t * in = data_img + inw * (im_row + inh * c_) + im_col + (w_low - 1) * sw;
                memset(out, 0, w_low * sizeof(int8_t));
                out += w_low;
                while (out < end)
                {
                    in += sw;
                    out[0] = in[0];
                    out++;
                }
                memset(out, 0, (outw - w_high) * sizeof(int8_t));
            }
            else
            {
                memset(out, 0, outw * sizeof(int8_t));
            }
        }
    }
}

static void im2col_ir(struct tensor* input, struct tensor* output, struct conv_priv_info* priv_info,
                      struct conv_param* param, int n, int group)
{
    int input_chan = param->input_channel / param->group;
    int image_size = input->dims[1] * input->dims[2] * input->dims[3];
    int group_size = input_chan * input->dims[2] * input->dims[3];

	void* input_base = (void*)((uint8_t*)input->data + (n * image_size + group * group_size) * input->elem_size);
	void* im2col_buf = (void*)priv_info->im2col_buffer;

    if (input->data_type == TENGINE_DT_FP32)
    {
        im2col_fp32(input_base, im2col_buf, input->dims[2], input->dims[3], input_chan, output->dims[2], output->dims[3],
                    param->kernel_h, param->kernel_w, param->stride_h, param->stride_w, param->pad_h0, param->pad_w0, param->dilation_h, param->dilation_w);
    }
    else if (input->data_type == TENGINE_DT_UINT8)
    {
        im2col_uint8(input_base, im2col_buf, input, output, param);
    }
    else if (input->data_type == TENGINE_DT_INT8)
    {
        im2col_int8(input_base, im2col_buf, input, output, param);
    }
    else
    {
        TLOG_ERR("Input data type %d not to be supported.\n", input->data_type);
    }
}

void input_pack4_fp32(int K, int N, float* pB, float* pB_t, int num_thread)
{
    int nn_size = N >> 3;
    int remian_size_start = nn_size << 3;

// [ch00, ch10, ch20, ch30, ch01, ch11, ch21, ch31, ch02, ch12, ch22, ch32, ch03, ch13, ch23, ch33 ....]
#pragma omp parallel for num_threads(num_thread)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = ii * 8;
        const float* img = pB + i;
        float* tmp = pB_t + (i / 8) * 8 * K;

        for (int j = 0; j < K; j++)
        {
#if __AVX__
            _mm256_storeu_ps(tmp, _mm256_loadu_ps(img));
#else
            tmp[0] = img[0];
            tmp[1] = img[1];
            tmp[2] = img[2];
            tmp[3] = img[3];
            tmp[4] = img[4];
            tmp[5] = img[5];
            tmp[6] = img[6];
            tmp[7] = img[7];
#endif    // __SSE__
            tmp += 8;
            img += N;
        }
    }

// [ch00, ch01, ch02, ch03 ....]
#pragma omp parallel for num_threads(num_thread)
    for (int i = remian_size_start; i < N; i++)
    {
        const float* img = pB + i;
        float* tmp = pB_t + (i / 8 + i % 8) * 8 * K;

        for (int j = 0; j < K; j++)
        {
            tmp[0] = img[0];

            tmp += 1;
            img += N;
        }
    }
}

static void sgemm_fp(int M, int N, int K, float* pA_t, float* pB_t, float* pC, int num_thread)
{
    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = M >> 3;
    remain_outch_start = nn_outch << 3;

#pragma omp parallel for num_threads(num_thread)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int i = pp * 8;

        float* output0 = pC + ( i )*N;
        float* output1 = pC + (i + 1) * N;
        float* output2 = pC + (i + 2) * N;
        float* output3 = pC + (i + 3) * N;
        float* output4 = pC + (i + 4) * N;
        float* output5 = pC + (i + 5) * N;
        float* output6 = pC + (i + 6) * N;
        float* output7 = pC + (i + 7) * N;

        int j = 0;
        for (; j + 7 < N; j += 8)
        {
            float* va = pA_t + (i / 8) * 8 * K;
            float* vb = pB_t + (j / 8) * 8 * K;
#if __AVX__
            __m256 _sum0 = _mm256_set1_ps(0.0);
            __m256 _sum1 = _mm256_set1_ps(0.0);
            __m256 _sum2 = _mm256_set1_ps(0.0);
            __m256 _sum3 = _mm256_set1_ps(0.0);
            __m256 _sum4 = _mm256_set1_ps(0.0);
            __m256 _sum5 = _mm256_set1_ps(0.0);
            __m256 _sum6 = _mm256_set1_ps(0.0);
            __m256 _sum7 = _mm256_set1_ps(0.0);

            int k = 0;
            for (; k + 3 < K; k = k + 4)
            {
                // k0
                __m256 _va0 = _mm256_broadcast_ss(va);
                __m256 _va1 = _mm256_broadcast_ss(va + 1);
                __m256 _va2 = _mm256_broadcast_ss(va + 2);
                __m256 _va3 = _mm256_broadcast_ss(va + 3);
                __m256 _vb0 = _mm256_loadu_ps(vb);
                __m256 _vb1 = _mm256_loadu_ps(vb + 8);
                __m256 _vb2 = _mm256_loadu_ps(vb + 16);
                __m256 _vb3 = _mm256_loadu_ps(vb + 24);
                _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0);    // sum0 = (a00-a07) * k00
                _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1);    // sum1 = (a00-a07) * k10
                _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2);    // sum2 = (a00-a07) * k20
                _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3);    // sum3 = (a00-a07) * k30
                _va0 = _mm256_broadcast_ss(va + 4);
                _va1 = _mm256_broadcast_ss(va + 5);
                _va2 = _mm256_broadcast_ss(va + 6);
                _va3 = _mm256_broadcast_ss(va + 7);
                _sum4 = _mm256_fmadd_ps(_vb0, _va0, _sum4);    // sum4 = (a00-a07) * k40
                _sum5 = _mm256_fmadd_ps(_vb0, _va1, _sum5);    // sum5 = (a00-a07) * k50
                _sum6 = _mm256_fmadd_ps(_vb0, _va2, _sum6);    // sum6 = (a00-a07) * k60
                _sum7 = _mm256_fmadd_ps(_vb0, _va3, _sum7);    // sum7 = (a00-a07) * k70

                va += 8;

                // k1
                _va0 = _mm256_broadcast_ss(va);
                _va1 = _mm256_broadcast_ss(va + 1);
                _va2 = _mm256_broadcast_ss(va + 2);
                _va3 = _mm256_broadcast_ss(va + 3);
                _sum0 = _mm256_fmadd_ps(_vb1, _va0, _sum0);    // sum0 += (a10-a17) * k01
                _sum1 = _mm256_fmadd_ps(_vb1, _va1, _sum1);    // sum1 += (a10-a17) * k11
                _sum2 = _mm256_fmadd_ps(_vb1, _va2, _sum2);    // sum2 += (a10-a17) * k21
                _sum3 = _mm256_fmadd_ps(_vb1, _va3, _sum3);    // sum3 += (a10-a17) * k31
                _va0 = _mm256_broadcast_ss(va + 4);
                _va1 = _mm256_broadcast_ss(va + 5);
                _va2 = _mm256_broadcast_ss(va + 6);
                _va3 = _mm256_broadcast_ss(va + 7);
                _sum4 = _mm256_fmadd_ps(_vb1, _va0, _sum4);    // sum4 += (a10-a17) * k41
                _sum5 = _mm256_fmadd_ps(_vb1, _va1, _sum5);    // sum5 += (a10-a17) * k51
                _sum6 = _mm256_fmadd_ps(_vb1, _va2, _sum6);    // sum6 += (a10-a17) * k61
                _sum7 = _mm256_fmadd_ps(_vb1, _va3, _sum7);    // sum7 += (a10-a17) * k71

                va += 8;

                // k2
                _va0 = _mm256_broadcast_ss(va);
                _va1 = _mm256_broadcast_ss(va + 1);
                _va2 = _mm256_broadcast_ss(va + 2);
                _va3 = _mm256_broadcast_ss(va + 3);
                _sum0 = _mm256_fmadd_ps(_vb2, _va0, _sum0);    // sum0 += (a20-a27) * k02
                _sum1 = _mm256_fmadd_ps(_vb2, _va1, _sum1);    // sum1 += (a20-a27) * k12
                _sum2 = _mm256_fmadd_ps(_vb2, _va2, _sum2);    // sum2 += (a20-a27) * k22
                _sum3 = _mm256_fmadd_ps(_vb2, _va3, _sum3);    // sum3 += (a20-a27) * k32
                _va0 = _mm256_broadcast_ss(va + 4);
                _va1 = _mm256_broadcast_ss(va + 5);
                _va2 = _mm256_broadcast_ss(va + 6);
                _va3 = _mm256_broadcast_ss(va + 7);
                _sum4 = _mm256_fmadd_ps(_vb2, _va0, _sum4);    // sum4 += (a20-a27) * k42
                _sum5 = _mm256_fmadd_ps(_vb2, _va1, _sum5);    // sum5 += (a20-a27) * k52
                _sum6 = _mm256_fmadd_ps(_vb2, _va2, _sum6);    // sum6 += (a20-a27) * k62
                _sum7 = _mm256_fmadd_ps(_vb2, _va3, _sum7);    // sum7 += (a20-a27) * k72

                va += 8;

                // k3
                _va0 = _mm256_broadcast_ss(va);
                _va1 = _mm256_broadcast_ss(va + 1);
                _va2 = _mm256_broadcast_ss(va + 2);
                _va3 = _mm256_broadcast_ss(va + 3);
                _sum0 = _mm256_fmadd_ps(_vb3, _va0, _sum0);    // sum0 += (a30-a37) * k03
                _sum1 = _mm256_fmadd_ps(_vb3, _va1, _sum1);    // sum1 += (a30-a37) * k13
                _sum2 = _mm256_fmadd_ps(_vb3, _va2, _sum2);    // sum2 += (a30-a37) * k23
                _sum3 = _mm256_fmadd_ps(_vb3, _va3, _sum3);    // sum3 += (a30-a37) * k33
                _va0 = _mm256_broadcast_ss(va + 4);
                _va1 = _mm256_broadcast_ss(va + 5);
                _va2 = _mm256_broadcast_ss(va + 6);
                _va3 = _mm256_broadcast_ss(va + 7);
                _sum4 = _mm256_fmadd_ps(_vb3, _va0, _sum4);    // sum4 += (a30-a37) * k43
                _sum5 = _mm256_fmadd_ps(_vb3, _va1, _sum5);    // sum5 += (a30-a37) * k53
                _sum6 = _mm256_fmadd_ps(_vb3, _va2, _sum6);    // sum6 += (a30-a37) * k63
                _sum7 = _mm256_fmadd_ps(_vb3, _va3, _sum7);    // sum7 += (a30-a37) * k73

                va += 8;
                vb += 32;
            }

            for (; k < K; k++)
            {
                // k0
                __m256 _va0 = _mm256_broadcast_ss(va);
                __m256 _va1 = _mm256_broadcast_ss(va + 1);
                __m256 _va2 = _mm256_broadcast_ss(va + 2);
                __m256 _va3 = _mm256_broadcast_ss(va + 3);
                __m256 _va4 = _mm256_broadcast_ss(va + 4);
                __m256 _va5 = _mm256_broadcast_ss(va + 5);
                __m256 _va6 = _mm256_broadcast_ss(va + 6);
                __m256 _va7 = _mm256_broadcast_ss(va + 7);
                __m256 _vb0 = _mm256_loadu_ps(vb);
                _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0);    // sum0 = (a00-a07) * k00
                _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1);    // sum1 = (a00-a07) * k10
                _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2);    // sum2 = (a00-a07) * k20
                _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3);    // sum3 = (a00-a07) * k30
                _sum4 = _mm256_fmadd_ps(_vb0, _va4, _sum4);    // sum4 = (a00-a07) * k40
                _sum5 = _mm256_fmadd_ps(_vb0, _va5, _sum5);    // sum5 = (a00-a07) * k50
                _sum6 = _mm256_fmadd_ps(_vb0, _va6, _sum6);    // sum6 = (a00-a07) * k60
                _sum7 = _mm256_fmadd_ps(_vb0, _va7, _sum7);    // sum7 = (a00-a07) * k70

                va += 8;
                vb += 8;
            }

            _mm256_storeu_ps(output0, _sum0);
            _mm256_storeu_ps(output1, _sum1);
            _mm256_storeu_ps(output2, _sum2);
            _mm256_storeu_ps(output3, _sum3);
            _mm256_storeu_ps(output4, _sum4);
            _mm256_storeu_ps(output5, _sum5);
            _mm256_storeu_ps(output6, _sum6);
            _mm256_storeu_ps(output7, _sum7);
#else
            float sum0[8] = {0};
            float sum1[8] = {0};
            float sum2[8] = {0};
            float sum3[8] = {0};
            float sum4[8] = {0};
            float sum5[8] = {0};
            float sum6[8] = {0};
            float sum7[8] = {0};

            for (int k = 0; k < K; k++)
            {
                for (int n = 0; n < 8; n++)
                {
                    sum0[n] += va[0] * vb[n];
                    sum1[n] += va[1] * vb[n];
                    sum2[n] += va[2] * vb[n];
                    sum3[n] += va[3] * vb[n];
                    sum4[n] += va[4] * vb[n];
                    sum5[n] += va[5] * vb[n];
                    sum6[n] += va[6] * vb[n];
                    sum7[n] += va[7] * vb[n];
                }

                va += 8;
                vb += 8;
            }

            for (int n = 0; n < 8; n++)
            {
                output0[n] = sum0[n];
                output1[n] = sum1[n];
                output2[n] = sum2[n];
                output3[n] = sum3[n];
                output4[n] = sum4[n];
                output5[n] = sum5[n];
                output6[n] = sum6[n];
                output7[n] = sum7[n];
            }
#endif    // __AVX__
            output0 += 8;
            output1 += 8;
            output2 += 8;
            output3 += 8;
            output4 += 8;
            output5 += 8;
            output6 += 8;
            output7 += 8;
        }

        for (; j < N; j++)
        {
            float* va = pA_t + (i / 8) * 8 * K;
            float* vb = pB_t + (j / 8 + j % 8) * 8 * K;

#if __AVX__
            __m256 _sum0_7 = _mm256_set1_ps(0.0);
            __m256 _sum0 = _mm256_set1_ps(0.0);
            __m256 _sum1 = _mm256_set1_ps(0.0);
            __m256 _sum2 = _mm256_set1_ps(0.0);
            __m256 _sum3 = _mm256_set1_ps(0.0);

            int k = 0;
            for (; k + 3 < K; k = k + 4)
            {
                __m256 _vb0 = _mm256_broadcast_ss(vb);
                __m256 _vb1 = _mm256_broadcast_ss(vb + 1);
                __m256 _vb2 = _mm256_broadcast_ss(vb + 2);
                __m256 _vb3 = _mm256_broadcast_ss(vb + 3);
                __m256 _va0 = _mm256_loadu_ps(va);
                __m256 _va1 = _mm256_loadu_ps(va + 8);
                __m256 _va2 = _mm256_loadu_ps(va + 16);
                __m256 _va3 = _mm256_loadu_ps(va + 24);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);    // sum0 += (k00-k70) * a00
                _sum1 = _mm256_fmadd_ps(_va1, _vb1, _sum1);    // sum1 += (k01-k71) * a10
                _sum2 = _mm256_fmadd_ps(_va2, _vb2, _sum2);    // sum2 += (k02-k72) * a20
                _sum3 = _mm256_fmadd_ps(_va3, _vb3, _sum3);    // sum3 += (k03-k73) * a30

                va += 32;
                vb += 4;
            }

            _sum0 = _mm256_add_ps(_sum0, _sum1);
            _sum2 = _mm256_add_ps(_sum2, _sum3);
            _sum0_7 = _mm256_add_ps(_sum0_7, _sum0);
            _sum0_7 = _mm256_add_ps(_sum0_7, _sum2);

            for (; k < K; k++)
            {
                __m256 _vb0 = _mm256_broadcast_ss(vb);
                __m256 _va = _mm256_loadu_ps(va);

                _sum0_7 = _mm256_fmadd_ps(_va, _vb0, _sum0_7);    // sum0 += (k00-k70) * a00

                va += 8;
                vb += 1;
            }

            float output_sum0_7[8] = {0.f};
            _mm256_storeu_ps(output_sum0_7, _sum0_7);

            output0[0] = output_sum0_7[0];
            output1[0] = output_sum0_7[1];
            output2[0] = output_sum0_7[2];
            output3[0] = output_sum0_7[3];
            output4[0] = output_sum0_7[4];
            output5[0] = output_sum0_7[5];
            output6[0] = output_sum0_7[6];
            output7[0] = output_sum0_7[7];
#else
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            float sum4 = 0;
            float sum5 = 0;
            float sum6 = 0;
            float sum7 = 0;

            for (int k = 0; k < K; k++)
            {
                sum0 += va[0] * vb[0];
                sum1 += va[1] * vb[0];
                sum2 += va[2] * vb[0];
                sum3 += va[3] * vb[0];
                sum4 += va[4] * vb[0];
                sum5 += va[5] * vb[0];
                sum6 += va[6] * vb[0];
                sum7 += va[7] * vb[0];

                va += 8;
                vb += 1;
            }
            output0[0] = sum0;
            output1[0] = sum1;
            output2[0] = sum2;
            output3[0] = sum3;
            output4[0] = sum4;
            output5[0] = sum5;
            output6[0] = sum6;
            output7[0] = sum7;
#endif    // __AVX__
            output0++;
            output1++;
            output2++;
            output3++;
            output4++;
            output5++;
            output6++;
            output7++;
        }
    }

    nn_outch = (M - remain_outch_start) >> 2;

    for (int pp = 0; pp < nn_outch; pp++)
    {
        int i = remain_outch_start + pp * 4;

        float* output0 = pC + ( i )*N;
        float* output1 = pC + (i + 1) * N;
        float* output2 = pC + (i + 2) * N;
        float* output3 = pC + (i + 3) * N;

        int j = 0;
        for (; j + 7 < N; j += 8)
        {
            float* va = pA_t + (i / 8 + (i % 8) / 4) * 8 * K;
            float* vb = pB_t + (j / 8) * 8 * K;
#if __AVX__
            __m256 _sum0 = _mm256_set1_ps(0.0);
            __m256 _sum1 = _mm256_set1_ps(0.0);
            __m256 _sum2 = _mm256_set1_ps(0.0);
            __m256 _sum3 = _mm256_set1_ps(0.0);

            int k = 0;
            for (; k + 3 < K; k = k + 4)
            {
                // k0
                __m256 _va0 = _mm256_broadcast_ss(va);
                __m256 _va1 = _mm256_broadcast_ss(va + 1);
                __m256 _va2 = _mm256_broadcast_ss(va + 2);
                __m256 _va3 = _mm256_broadcast_ss(va + 3);
                __m256 _vb0 = _mm256_loadu_ps(vb);
                __m256 _vb1 = _mm256_loadu_ps(vb + 8);
                __m256 _vb2 = _mm256_loadu_ps(vb + 16);
                __m256 _vb3 = _mm256_loadu_ps(vb + 24);
                _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0);    // sum0 = (a00-a07) * k00
                _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1);    // sum1 = (a00-a07) * k10
                _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2);    // sum2 = (a00-a07) * k20
                _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3);    // sum3 = (a00-a07) * k30

                va += 4;

                // k1
                _va0 = _mm256_broadcast_ss(va);
                _va1 = _mm256_broadcast_ss(va + 1);
                _va2 = _mm256_broadcast_ss(va + 2);
                _va3 = _mm256_broadcast_ss(va + 3);
                _sum0 = _mm256_fmadd_ps(_vb1, _va0, _sum0);    // sum0 += (a10-a17) * k01
                _sum1 = _mm256_fmadd_ps(_vb1, _va1, _sum1);    // sum1 += (a10-a17) * k11
                _sum2 = _mm256_fmadd_ps(_vb1, _va2, _sum2);    // sum2 += (a10-a17) * k21
                _sum3 = _mm256_fmadd_ps(_vb1, _va3, _sum3);    // sum3 += (a10-a17) * k31

                va += 4;

                // k2
                _va0 = _mm256_broadcast_ss(va);
                _va1 = _mm256_broadcast_ss(va + 1);
                _va2 = _mm256_broadcast_ss(va + 2);
                _va3 = _mm256_broadcast_ss(va + 3);
                _sum0 = _mm256_fmadd_ps(_vb2, _va0, _sum0);    // sum0 += (a20-a27) * k02
                _sum1 = _mm256_fmadd_ps(_vb2, _va1, _sum1);    // sum1 += (a20-a27) * k12
                _sum2 = _mm256_fmadd_ps(_vb2, _va2, _sum2);    // sum2 += (a20-a27) * k22
                _sum3 = _mm256_fmadd_ps(_vb2, _va3, _sum3);    // sum3 += (a20-a27) * k32

                va += 4;

                // k3
                _va0 = _mm256_broadcast_ss(va);
                _va1 = _mm256_broadcast_ss(va + 1);
                _va2 = _mm256_broadcast_ss(va + 2);
                _va3 = _mm256_broadcast_ss(va + 3);
                _sum0 = _mm256_fmadd_ps(_vb3, _va0, _sum0);    // sum0 += (a30-a37) * k03
                _sum1 = _mm256_fmadd_ps(_vb3, _va1, _sum1);    // sum1 += (a30-a37) * k13
                _sum2 = _mm256_fmadd_ps(_vb3, _va2, _sum2);    // sum2 += (a30-a37) * k23
                _sum3 = _mm256_fmadd_ps(_vb3, _va3, _sum3);    // sum3 += (a30-a37) * k33

                va += 4;
                vb += 32;
            }

            for (; k < K; k++)
            {
                // k0
                __m256 _va0 = _mm256_broadcast_ss(va);
                __m256 _va1 = _mm256_broadcast_ss(va + 1);
                __m256 _va2 = _mm256_broadcast_ss(va + 2);
                __m256 _va3 = _mm256_broadcast_ss(va + 3);
                __m256 _vb0 = _mm256_loadu_ps(vb);
                _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0);    // sum0 = (a00-a07) * k00
                _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1);    // sum1 = (a00-a07) * k10
                _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2);    // sum2 = (a00-a07) * k20
                _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3);    // sum3 = (a00-a07) * k30

                va += 4;
                vb += 8;
            }

            _mm256_storeu_ps(output0, _sum0);
            _mm256_storeu_ps(output1, _sum1);
            _mm256_storeu_ps(output2, _sum2);
            _mm256_storeu_ps(output3, _sum3);
#else
            float sum0[8] = {0};
            float sum1[8] = {0};
            float sum2[8] = {0};
            float sum3[8] = {0};

            for (int k = 0; k < K; k++)
            {
                for (int n = 0; n < 8; n++)
                {
                    sum0[n] += va[0] * vb[n];
                    sum1[n] += va[1] * vb[n];
                    sum2[n] += va[2] * vb[n];
                    sum3[n] += va[3] * vb[n];
                }

                va += 4;
                vb += 8;
            }

            for (int n = 0; n < 8; n++)
            {
                output0[n] = sum0[n];
                output1[n] = sum1[n];
                output2[n] = sum2[n];
                output3[n] = sum3[n];
            }
#endif    // __AVX__
            output0 += 8;
            output1 += 8;
            output2 += 8;
            output3 += 8;
        }

        for (; j < N; j++)
        {
            float* va = pA_t + (i / 8 + (i % 8) / 4) * 8 * K;
            float* vb = pB_t + (j / 8 + j % 8) * 8 * K;
#if __AVX__
            __m128 _sum0_3 = _mm_set1_ps(0.0);
            __m128 _sum0 = _mm_set1_ps(0.0);
            __m128 _sum1 = _mm_set1_ps(0.0);
            __m128 _sum2 = _mm_set1_ps(0.0);
            __m128 _sum3 = _mm_set1_ps(0.0);

            int k = 0;
            for (; k + 3 < K; k = k + 4)
            {
                __m128 _vb0 = _mm_set1_ps(vb[0]);
                __m128 _vb1 = _mm_set1_ps(vb[1]);
                __m128 _vb2 = _mm_set1_ps(vb[2]);
                __m128 _vb3 = _mm_set1_ps(vb[3]);
                __m128 _va0 = _mm_loadu_ps(va);
                __m128 _va1 = _mm_loadu_ps(va + 4);
                __m128 _va2 = _mm_loadu_ps(va + 8);
                __m128 _va3 = _mm_loadu_ps(va + 12);

                _sum0 = _mm_fmadd_ps(_va0, _vb0, _sum0);    // sum0 += (k00-k30) * a00
                _sum1 = _mm_fmadd_ps(_va1, _vb1, _sum1);    // sum1 += (k01-k31) * a10
                _sum2 = _mm_fmadd_ps(_va2, _vb2, _sum2);    // sum2 += (k02-k32) * a20
                _sum3 = _mm_fmadd_ps(_va3, _vb3, _sum3);    // sum3 += (k03-k33) * a30

                va += 16;
                vb += 4;
            }

            _sum0 = _mm_add_ps(_sum0, _sum1);
            _sum2 = _mm_add_ps(_sum2, _sum3);
            _sum0_3 = _mm_add_ps(_sum0_3, _sum0);
            _sum0_3 = _mm_add_ps(_sum0_3, _sum2);

            for (; k < K; k++)
            {
                __m128 _vb0 = _mm_set1_ps(vb[0]);
                __m128 _va = _mm_loadu_ps(va);

                _sum0_3 = _mm_fmadd_ps(_va, _vb0, _sum0_3);    // sum0 += (k00-k30) * a00

                va += 4;
                vb += 1;
            }

            float output_sum0_3[4] = {0.f};
            _mm_storeu_ps(output_sum0_3, _sum0_3);
            output0[0] = output_sum0_3[0];
            output1[0] = output_sum0_3[1];
            output2[0] = output_sum0_3[2];
            output3[0] = output_sum0_3[3];
#else
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;

            for (int k = 0; k < K; k++)
            {
                sum0 += va[0] * vb[0];
                sum1 += va[1] * vb[0];
                sum2 += va[2] * vb[0];
                sum3 += va[3] * vb[0];

                va += 4;
                vb += 1;
            }
            output0[0] = sum0;
            output1[0] = sum1;
            output2[0] = sum2;
            output3[0] = sum3;
#endif    // __AVX__
            output0++;
            output1++;
            output2++;
            output3++;
        }
    }

    remain_outch_start += nn_outch << 2;

    // output ch0
    for (int i = remain_outch_start; i < M; i++)
    {
        float* output = pC + i * N;

        int j = 0;
        for (; j + 7 < N; j += 8)
        {
            float* va = pA_t + (i / 8 + (i % 8) / 4 + i % 4) * 8 * K;
            float* vb = pB_t + (j / 8) * 8 * K;
#if __AVX__
            __m256 _sum0 = _mm256_set1_ps(0.0);

            int k = 0;
            for (; k + 3 < K; k = k + 4)
            {
                // k0
                __m256 _va0 = _mm256_broadcast_ss(va);
                __m256 _va1 = _mm256_broadcast_ss(va + 1);
                __m256 _va2 = _mm256_broadcast_ss(va + 2);
                __m256 _va3 = _mm256_broadcast_ss(va + 3);
                __m256 _vb0 = _mm256_loadu_ps(vb);
                __m256 _vb1 = _mm256_loadu_ps(vb + 8);
                __m256 _vb2 = _mm256_loadu_ps(vb + 16);
                __m256 _vb3 = _mm256_loadu_ps(vb + 24);

                _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0);    // sum0 = (a00-a07) * k00
                _sum0 = _mm256_fmadd_ps(_vb1, _va1, _sum0);    // sum0 += (a10-a17) * k01
                _sum0 = _mm256_fmadd_ps(_vb2, _va2, _sum0);    // sum0 += (a20-a27) * k02
                _sum0 = _mm256_fmadd_ps(_vb3, _va3, _sum0);    // sum0 += (a30-a37) * k03

                va += 4;
                vb += 32;
            }

            for (; k < K; k++)
            {
                // k0
                __m256 _va0 = _mm256_broadcast_ss(va);
                __m256 _vb0 = _mm256_loadu_ps(vb);

                _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0);    // sum0 = (a00-a07) * k00

                va += 1;
                vb += 8;
            }

            _mm256_storeu_ps(output, _sum0);
#else
            float sum[8] = {0};

            for (int k = 0; k < K; k++)
            {
                for (int n = 0; n < 8; n++)
                {
                    sum[n] += va[0] * vb[n];
                }

                va += 1;
                vb += 8;
            }

            for (int n = 0; n < 8; n++)
            {
                output[n] = sum[n];
            }
#endif    // __AVX__
            output += 8;
        }

        for (; j < N; j++)
        {
            float* va = pA_t + (i / 8 + (i % 8) / 4 + i % 4) * 8 * K;
            float* vb = pB_t + (j / 8 + j % 8) * 8 * K;

            int k = 0;
#if __AVX__
            __m128 _sum0 = _mm_set1_ps(0.f);

            for (; k + 3 < K; k += 4)
            {
                __m128 _p0 = _mm_loadu_ps(vb);
                __m128 _k0 = _mm_loadu_ps(va);
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_p0, _k0));

                va += 4;
                vb += 4;
            }
#ifdef _WIN32
            float sum0 = _sum0.m128_f32[0] + _sum0.m128_f32[1] + _sum0.m128_f32[2] + _sum0.m128_f32[3];
#else
            float sum0 = _sum0[0] + _sum0[1] + _sum0[2] + _sum0[3];
#endif
#else
            float sum0 = 0.f;
#endif    // __AVX__
            for (; k < K; k++)
            {
                sum0 += va[0] * vb[0];

                va += 1;
                vb += 1;
            }
            output[0] = sum0;

            output++;
        }
    }
}

void input_pack4_int8(int K, int N, int8_t* pB, int8_t* pB_t, int num_thread)
{
    int nn_size = N >> 3;
    int remian_size_start = nn_size << 3;

// [ch00, ch10, ch20, ch30, ch01, ch11, ch21, ch31, ch02, ch12, ch22, ch32, ch03, ch13, ch23, ch33 ....]
#pragma omp parallel for num_threads(num_thread)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = ii * 8;
        const int8_t* img = pB + i;
        int8_t* tmp = pB_t + (i / 8) * 8 * K;

        for (int j = 0; j < K; j++)
        {
            tmp[0] = img[0];
            tmp[1] = img[1];
            tmp[2] = img[2];
            tmp[3] = img[3];
            tmp[4] = img[4];
            tmp[5] = img[5];
            tmp[6] = img[6];
            tmp[7] = img[7];

            tmp += 8;
            img += N;
        }
    }

// [ch00, ch01, ch02, ch03 ....]
#pragma omp parallel for num_threads(num_thread)
    for (int i = remian_size_start; i < N; i++)
    {
        const int8_t* img = pB + i;
        int8_t* tmp = pB_t + (i / 8 + i % 8) * 8 * K;

        for (int j = 0; j < K; j++)
        {
            tmp[0] = img[0];

            tmp += 1;
            img += N;
        }
    }
}
static void sgemm_i8(int M, int N, int K, int8_t* pA_t, int8_t* pB_t, int32_t* pC, int num_thread)
{
    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = M >> 3;
    remain_outch_start = nn_outch << 3;

#pragma omp parallel for num_threads(num_thread)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int i = pp * 8;

        int32_t* output0 = pC + ( i )*N;
        int32_t* output1 = pC + (i + 1) * N;
        int32_t* output2 = pC + (i + 2) * N;
        int32_t* output3 = pC + (i + 3) * N;
        int32_t* output4 = pC + (i + 4) * N;
        int32_t* output5 = pC + (i + 5) * N;
        int32_t* output6 = pC + (i + 6) * N;
        int32_t* output7 = pC + (i + 7) * N;

        int j = 0;
        for (; j + 7 < N; j += 8)
        {
            int8_t* va = pA_t + (i / 8) * 8 * K;
            int8_t* vb = pB_t + (j / 8) * 8 * K;

#if 0 //__AVX__
            __m256i _sum0 = _mm256_set1_epi32(0);
            __m256i _sum1 = _mm256_set1_epi32(0);
            __m256i _sum2 = _mm256_set1_epi32(0);
            __m256i _sum3 = _mm256_set1_epi32(0);
            __m256i _sum4 = _mm256_set1_epi32(0);
            __m256i _sum5 = _mm256_set1_epi32(0);
            __m256i _sum6 = _mm256_set1_epi32(0);
            __m256i _sum7 = _mm256_set1_epi32(0);

            int k = 0;
            for (; k + 3 < K; k = k + 4) {
                // k0
                __m256i _va0 = _mm256_set1_epi32(*va);
                __m256i _va1 = _mm256_set1_epi32(*(va + 1));
                __m256i _va2 = _mm256_set1_epi32(*(va + 2));
                __m256i _va3 = _mm256_set1_epi32(*(va + 3));
                __m256i _vb0 = _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)vb));
                __m256i _vb1 =
                    _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(vb + 8)));
                __m256i _vb2 =
                    _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(vb + 16)));
                __m256i _vb3 =
                    _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(vb + 24)));
                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va0), _sum0);
                _sum1 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va1), _sum1);
                _sum2 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va2), _sum2);
                _sum3 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va3), _sum3);
                _va0 = _mm256_set1_epi32(*(va + 4));
                _va1 = _mm256_set1_epi32(*(va + 5));
                _va2 = _mm256_set1_epi32(*(va + 6));
                _va3 = _mm256_set1_epi32(*(va + 7));
                _sum4 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va0), _sum4);
                _sum5 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va1), _sum5);
                _sum6 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va2), _sum6);
                _sum7 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va3), _sum7);

                va += 8;

                // k1
                _va0 = _mm256_set1_epi32(*va);
                _va1 = _mm256_set1_epi32(*(va + 1));
                _va2 = _mm256_set1_epi32(*(va + 2));
                _va3 = _mm256_set1_epi32(*(va + 3));
                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_vb1, _va0), _sum0);
                _sum1 = _mm256_add_epi32(_mm256_mullo_epi32(_vb1, _va1), _sum1);
                _sum2 = _mm256_add_epi32(_mm256_mullo_epi32(_vb1, _va2), _sum2);
                _sum3 = _mm256_add_epi32(_mm256_mullo_epi32(_vb1, _va3), _sum3);
                _va0 = _mm256_set1_epi32(*(va + 4));
                _va1 = _mm256_set1_epi32(*(va + 5));
                _va2 = _mm256_set1_epi32(*(va + 6));
                _va3 = _mm256_set1_epi32(*(va + 7));
                _sum4 = _mm256_add_epi32(_mm256_mullo_epi32(_vb1, _va0), _sum4);
                _sum5 = _mm256_add_epi32(_mm256_mullo_epi32(_vb1, _va1), _sum5);
                _sum6 = _mm256_add_epi32(_mm256_mullo_epi32(_vb1, _va2), _sum6);
                _sum7 = _mm256_add_epi32(_mm256_mullo_epi32(_vb1, _va3), _sum7);

                va += 8;

                // k2
                _va0 = _mm256_set1_epi32(*va);
                _va1 = _mm256_set1_epi32(*(va + 1));
                _va2 = _mm256_set1_epi32(*(va + 2));
                _va3 = _mm256_set1_epi32(*(va + 3));
                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_vb2, _va0), _sum0);
                _sum1 = _mm256_add_epi32(_mm256_mullo_epi32(_vb2, _va1), _sum1);
                _sum2 = _mm256_add_epi32(_mm256_mullo_epi32(_vb2, _va2), _sum2);
                _sum3 = _mm256_add_epi32(_mm256_mullo_epi32(_vb2, _va3), _sum3);
                _va0 = _mm256_set1_epi32(*(va + 4));
                _va1 = _mm256_set1_epi32(*(va + 5));
                _va2 = _mm256_set1_epi32(*(va + 6));
                _va3 = _mm256_set1_epi32(*(va + 7));
                _sum4 = _mm256_add_epi32(_mm256_mullo_epi32(_vb2, _va0), _sum4);
                _sum5 = _mm256_add_epi32(_mm256_mullo_epi32(_vb2, _va1), _sum5);
                _sum6 = _mm256_add_epi32(_mm256_mullo_epi32(_vb2, _va2), _sum6);
                _sum7 = _mm256_add_epi32(_mm256_mullo_epi32(_vb2, _va3), _sum7);

                va += 8;

                // k3
                _va0 = _mm256_set1_epi32(*va);
                _va1 = _mm256_set1_epi32(*(va + 1));
                _va2 = _mm256_set1_epi32(*(va + 2));
                _va3 = _mm256_set1_epi32(*(va + 3));
                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_vb3, _va0), _sum0);
                _sum1 = _mm256_add_epi32(_mm256_mullo_epi32(_vb3, _va1), _sum1);
                _sum2 = _mm256_add_epi32(_mm256_mullo_epi32(_vb3, _va2), _sum2);
                _sum3 = _mm256_add_epi32(_mm256_mullo_epi32(_vb3, _va3), _sum3);
                _va0 = _mm256_set1_epi32(*(va + 4));
                _va1 = _mm256_set1_epi32(*(va + 5));
                _va2 = _mm256_set1_epi32(*(va + 6));
                _va3 = _mm256_set1_epi32(*(va + 7));
                _sum4 = _mm256_add_epi32(_mm256_mullo_epi32(_vb3, _va0), _sum4);
                _sum5 = _mm256_add_epi32(_mm256_mullo_epi32(_vb3, _va1), _sum5);
                _sum6 = _mm256_add_epi32(_mm256_mullo_epi32(_vb3, _va2), _sum6);
                _sum7 = _mm256_add_epi32(_mm256_mullo_epi32(_vb3, _va3), _sum7);

                va += 8;
                vb += 32;
            }
            for (; k < K; k++) {
                __m256i _va0 = _mm256_set1_epi32(*va);
                __m256i _va1 = _mm256_set1_epi32(*(va + 1));
                __m256i _va2 = _mm256_set1_epi32(*(va + 2));
                __m256i _va3 = _mm256_set1_epi32(*(va + 3));
                __m256i _va4 = _mm256_set1_epi32(*(va + 4));
                __m256i _va5 = _mm256_set1_epi32(*(va + 5));
                __m256i _va6 = _mm256_set1_epi32(*(va + 6));
                __m256i _va7 = _mm256_set1_epi32(*(va + 7));
                __m256i _vb0 = _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)vb));
                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va0), _sum0);
                _sum1 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va1), _sum1);
                _sum2 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va2), _sum2);
                _sum3 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va3), _sum3);
                _sum4 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va4), _sum4);
                _sum5 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va5), _sum5);
                _sum6 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va6), _sum6);
                _sum7 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va7), _sum7);

                va += 8;
                vb += 8;
            }

            _mm256_storeu_si256((__m256i* )output0, _sum0);
            _mm256_storeu_si256((__m256i* )output1, _sum1);
            _mm256_storeu_si256((__m256i* )output2, _sum2);
            _mm256_storeu_si256((__m256i* )output3, _sum3);
            _mm256_storeu_si256((__m256i* )output4, _sum4);
            _mm256_storeu_si256((__m256i* )output5, _sum5);
            _mm256_storeu_si256((__m256i* )output6, _sum6);
            _mm256_storeu_si256((__m256i* )output7, _sum7);

#else
            int32_t sum0[8] = {0};
            int32_t sum1[8] = {0};
            int32_t sum2[8] = {0};
            int32_t sum3[8] = {0};
            int32_t sum4[8] = {0};
            int32_t sum5[8] = {0};
            int32_t sum6[8] = {0};
            int32_t sum7[8] = {0};

            for (int k = 0; k < K; k++)
            {
                for (int n = 0; n < 8; n++)
                {
                    sum0[n] += va[0] * vb[n];
                    sum1[n] += va[1] * vb[n];
                    sum2[n] += va[2] * vb[n];
                    sum3[n] += va[3] * vb[n];
                    sum4[n] += va[4] * vb[n];
                    sum5[n] += va[5] * vb[n];
                    sum6[n] += va[6] * vb[n];
                    sum7[n] += va[7] * vb[n];
                }

                va += 8;
                vb += 8;
            }

            for (int n = 0; n < 8; n++)
            {
                output0[n] = sum0[n];
                output1[n] = sum1[n];
                output2[n] = sum2[n];
                output3[n] = sum3[n];
                output4[n] = sum4[n];
                output5[n] = sum5[n];
                output6[n] = sum6[n];
                output7[n] = sum7[n];
            }
#endif
            output0 += 8;
            output1 += 8;
            output2 += 8;
            output3 += 8;
            output4 += 8;
            output5 += 8;
            output6 += 8;
            output7 += 8;
        }

        for (; j < N; j++)
        {
            int8_t* va = pA_t + (i / 8) * 8 * K;
            int8_t* vb = pB_t + (j / 8 + j % 8) * 8 * K;

#if 0 //__AVX__
            __m256i _sum0_7 = _mm256_set1_epi32(0);
            __m256i _sum0 = _mm256_set1_epi32(0);
            __m256i _sum1 = _mm256_set1_epi32(0);
            __m256i _sum2 = _mm256_set1_epi32(0);
            __m256i _sum3 = _mm256_set1_epi32(0);

            int k = 0;
            for (; k + 3 < K; k = k + 4) {
                __m256i _vb0 = _mm256_set1_epi32(*vb);
                __m256i _vb1 = _mm256_set1_epi32(*(vb + 1));
                __m256i _vb2 = _mm256_set1_epi32(*(vb + 2));
                __m256i _vb3 = _mm256_set1_epi32(*(vb + 3));
                __m256i _va0 = _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)va));
                __m256i _va1 =
                    _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(va + 8)));
                __m256i _va2 =
                    _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(va + 16)));
                __m256i _va3 =
                    _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(va + 24)));

                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_va0, _vb0), _sum0);
                _sum1 = _mm256_add_epi32(_mm256_mullo_epi32(_va1, _vb1), _sum1);
                _sum2 = _mm256_add_epi32(_mm256_mullo_epi32(_va2, _vb2), _sum2);
                _sum3 = _mm256_add_epi32(_mm256_mullo_epi32(_va3, _vb3), _sum3);

                va += 32;
                vb += 4;
            }

            _sum0 = _mm256_add_epi32(_sum0, _sum1);
            _sum2 = _mm256_add_epi32(_sum2, _sum3);
            _sum0_7 = _mm256_add_epi32(_sum0_7, _sum0);
            _sum0_7 = _mm256_add_epi32(_sum0_7, _sum2);

            for (; k < K; k++) {
                __m256i _vb0 = _mm256_set1_epi32(*vb);
                __m256i _va = _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)va));

                _sum0_7 = _mm256_add_epi32(_mm256_mullo_epi32(_va, _vb0), _sum0_7);

                va += 8;
                vb += 1;
            }

            int32_t output_sum0_7[8] = {0};
            _mm256_storeu_si256((__m256i* )output_sum0_7, _sum0_7);

            output0[0] = output_sum0_7[0];
            output1[0] = output_sum0_7[1];
            output2[0] = output_sum0_7[2];
            output3[0] = output_sum0_7[3];
            output4[0] = output_sum0_7[4];
            output5[0] = output_sum0_7[5];
            output6[0] = output_sum0_7[6];
            output7[0] = output_sum0_7[7];
#else
            int32_t sum0 = 0;
            int32_t sum1 = 0;
            int32_t sum2 = 0;
            int32_t sum3 = 0;
            int32_t sum4 = 0;
            int32_t sum5 = 0;
            int32_t sum6 = 0;
            int32_t sum7 = 0;

            for (int k = 0; k < K; k++)
            {
                sum0 += va[0] * vb[0];
                sum1 += va[1] * vb[0];
                sum2 += va[2] * vb[0];
                sum3 += va[3] * vb[0];
                sum4 += va[4] * vb[0];
                sum5 += va[5] * vb[0];
                sum6 += va[6] * vb[0];
                sum7 += va[7] * vb[0];

                va += 8;
                vb += 1;
            }
            output0[0] = sum0;
            output1[0] = sum1;
            output2[0] = sum2;
            output3[0] = sum3;
            output4[0] = sum4;
            output5[0] = sum5;
            output6[0] = sum6;
            output7[0] = sum7;
#endif
            output0++;
            output1++;
            output2++;
            output3++;
            output4++;
            output5++;
            output6++;
            output7++;
        }
    }

    nn_outch = (M - remain_outch_start) >> 2;

    for (int pp = 0; pp < nn_outch; pp++)
    {
        int i = remain_outch_start + pp * 4;

        int32_t* output0 = pC + ( i )*N;
        int32_t* output1 = pC + (i + 1) * N;
        int32_t* output2 = pC + (i + 2) * N;
        int32_t* output3 = pC + (i + 3) * N;

        int j = 0;
        for (; j + 7 < N; j += 8)
        {
            int8_t* va = pA_t + (i / 8 + (i % 8) / 4) * 8 * K;
            int8_t* vb = pB_t + (j / 8) * 8 * K;
#if 0 //__AVX__
            __m256i _sum0 = _mm256_set1_epi32(0);
            __m256i _sum1 = _mm256_set1_epi32(0);
            __m256i _sum2 = _mm256_set1_epi32(0);
            __m256i _sum3 = _mm256_set1_epi32(0);

            int k = 0;
            for (; k + 3 < K; k = K + 4) {
                // k0
                __m256i _va0 = _mm256_set1_epi32(*va);
                __m256i _va1 = _mm256_set1_epi32(*(va + 1));
                __m256i _va2 = _mm256_set1_epi32(*(va + 2));
                __m256i _va3 = _mm256_set1_epi32(*(va + 3));
                __m256i _vb0 = _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)vb));
                __m256i _vb1 =
                    _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(vb + 8)));
                __m256i _vb2 =
                    _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(vb + 16)));
                __m256i _vb3 =
                    _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(vb + 24)));
                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va0), _sum0);
                _sum1 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va1), _sum1);
                _sum2 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va2), _sum2);
                _sum3 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va3), _sum3);

                va += 4;

                // k1
                _va0 = _mm256_set1_epi32(*va);
                _va1 = _mm256_set1_epi32(*(va + 1));
                _va2 = _mm256_set1_epi32(*(va + 2));
                _va3 = _mm256_set1_epi32(*(va + 3));
                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_vb1, _va0), _sum0);
                _sum1 = _mm256_add_epi32(_mm256_mullo_epi32(_vb1, _va1), _sum1);
                _sum2 = _mm256_add_epi32(_mm256_mullo_epi32(_vb1, _va2), _sum2);
                _sum3 = _mm256_add_epi32(_mm256_mullo_epi32(_vb1, _va3), _sum3);

                va += 4;

                // k2
                _va0 = _mm256_set1_epi32(*va);
                _va1 = _mm256_set1_epi32(*(va + 1));
                _va2 = _mm256_set1_epi32(*(va + 2));
                _va3 = _mm256_set1_epi32(*(va + 3));
                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_vb2, _va0), _sum0);
                _sum1 = _mm256_add_epi32(_mm256_mullo_epi32(_vb2, _va1), _sum1);
                _sum2 = _mm256_add_epi32(_mm256_mullo_epi32(_vb2, _va2), _sum2);
                _sum3 = _mm256_add_epi32(_mm256_mullo_epi32(_vb2, _va3), _sum3);

                va += 4;

                // k3
                _va0 = _mm256_set1_epi32(*va);
                _va1 = _mm256_set1_epi32(*(va + 1));
                _va2 = _mm256_set1_epi32(*(va + 2));
                _va3 = _mm256_set1_epi32(*(va + 3));
                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_vb3, _va0), _sum0);
                _sum1 = _mm256_add_epi32(_mm256_mullo_epi32(_vb3, _va1), _sum1);
                _sum2 = _mm256_add_epi32(_mm256_mullo_epi32(_vb3, _va2), _sum2);
                _sum3 = _mm256_add_epi32(_mm256_mullo_epi32(_vb3, _va3), _sum3);

                va += 4;
                vb += 32;
            }

            for (; k < K; k++) {
                __m256i _va0 = _mm256_set1_epi32(*va);
                __m256i _va1 = _mm256_set1_epi32(*(va + 1));
                __m256i _va2 = _mm256_set1_epi32(*(va + 2));
                __m256i _va3 = _mm256_set1_epi32(*(va + 3));
                __m256i _vb0 = _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)vb));
                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va0), _sum0);
                _sum1 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va1), _sum1);
                _sum2 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va2), _sum2);
                _sum3 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va3), _sum3);

                va += 4;
                vb += 8;
            }

            _mm256_storeu_si256((__m256i* )output0, _sum0);
            _mm256_storeu_si256((__m256i* )output1, _sum1);
            _mm256_storeu_si256((__m256i* )output2, _sum2);
            _mm256_storeu_si256((__m256i* )output3, _sum3);
#else
            int32_t sum0[8] = {0};
            int32_t sum1[8] = {0};
            int32_t sum2[8] = {0};
            int32_t sum3[8] = {0};

            for (int k = 0; k < K; k++)
            {
                for (int n = 0; n < 8; n++)
                {
                    sum0[n] += va[0] * vb[n];
                    sum1[n] += va[1] * vb[n];
                    sum2[n] += va[2] * vb[n];
                    sum3[n] += va[3] * vb[n];
                }

                va += 4;
                vb += 8;
            }

            for (int n = 0; n < 8; n++)
            {
                output0[n] = sum0[n];
                output1[n] = sum1[n];
                output2[n] = sum2[n];
                output3[n] = sum3[n];
            }
#endif
            output0 += 8;
            output1 += 8;
            output2 += 8;
            output3 += 8;
        }

        for (; j < N; j++)
        {
            int8_t* va = pA_t + (i / 8 + (i % 8) / 4) * 8 * K;
            int8_t* vb = pB_t + (j / 8 + j % 8) * 8 * K;
#if 0 //__AVX__
        __m256i _sum0_3 = _mm256_set1_epi32(0);
        __m256i _sum0 = _mm256_set1_epi32(0);
        __m256i _sum1 = _mm256_set1_epi32(0);
        __m256i _sum2 = _mm256_set1_epi32(0);
        __m256i _sum3 = _mm256_set1_epi32(0);

        int k=0;
        for (; k + 3 < K; k = k + 4)
        {
            __m256i _vb0 = _mm256_set1_epi32(*vb);
            __m256i _vb1 = _mm256_set1_epi32(*(vb + 1));
            __m256i _vb2 = _mm256_set1_epi32(*(vb + 2));
            __m256i _vb3 = _mm256_set1_epi32(*(vb + 3));
            __m256i _va0 = _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)va));
            __m256i _va1 = _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(va + 4)));
            __m256i _va2 = _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(va + 8)));
            __m256i _va3 = _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(va + 12)));

            _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_va0, _vb0), _sum0);
            _sum1 = _mm256_add_epi32(_mm256_mullo_epi32(_va1, _vb1), _sum1);
            _sum2 = _mm256_add_epi32(_mm256_mullo_epi32(_va2, _vb2), _sum2);
            _sum3 = _mm256_add_epi32(_mm256_mullo_epi32(_va3, _vb3), _sum3);

            va+=16;
            vb+=4;
        }

        _sum0 = _mm256_add_epi32(_sum0, _sum1);
        _sum2 = _mm256_add_epi32(_sum2, _sum3);
        _sum0_3 = _mm256_add_epi32(_sum0_3, _sum0);
        _sum0_3 = _mm256_add_epi32(_sum0_3, _sum2);

        for (; k < K; k++)
        {
            __m256i _vb0 = _mm256_set1_epi32(*vb);
            __m256i _va = _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)va));

            _sum0_3 = _mm256_add_epi32(_mm256_mullo_epi32(_va, _vb0), _sum0_3);

            va += 4;
            vb += 1;
        }
        //drop last 4 value
        int32_t output_sum0_3[4] = {0};
        _mm256_storeu_si256((__m256i* )output_sum0_3, _sum0_3);
        output0[0] = output_sum0_3[0];
        output1[0] = output_sum0_3[1];
        output2[0] = output_sum0_3[2];
        output3[0] = output_sum0_3[3];
#else
            int32_t sum0 = 0;
            int32_t sum1 = 0;
            int32_t sum2 = 0;
            int32_t sum3 = 0;

            for (int k = 0; k < K; k++)
            {
                sum0 += va[0] * vb[0];
                sum1 += va[1] * vb[0];
                sum2 += va[2] * vb[0];
                sum3 += va[3] * vb[0];

                va += 4;
                vb += 1;
            }
            output0[0] = sum0;
            output1[0] = sum1;
            output2[0] = sum2;
            output3[0] = sum3;
#endif
            output0++;
            output1++;
            output2++;
            output3++;
        }
    }

    remain_outch_start += nn_outch << 2;

    // output ch0
    for (int i = remain_outch_start; i < M; i++)
    {
        int32_t* output = pC + i * N;

        int j = 0;
        for (; j + 7 < N; j += 8)
        {
            int8_t* va = pA_t + (i / 8 + (i % 8) / 4 + i % 4) * 8 * K;
            int8_t* vb = pB_t + (j / 8) * 8 * K;
#if 0 //__AVX__
            __m256i _sum0 = _mm256_set1_epi32(0);

            int k = 0;
            for (; k + 3 < K; k = k + 4) {
                __m256i _va0 = _mm256_set1_epi32(*va);
                __m256i _va1 = _mm256_set1_epi32(*(va + 1));
                __m256i _va2 = _mm256_set1_epi32(*(va + 2));
                __m256i _va3 = _mm256_set1_epi32(*(va + 3));
                __m256i _vb0 = _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)vb));
                __m256i _vb1 =
                    _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(vb + 8)));
                __m256i _vb2 =
                    _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(vb + 16)));
                __m256i _vb3 =
                    _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)(vb + 24)));

                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va0), _sum0);
                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_vb1, _va1), _sum0);
                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_vb2, _va2), _sum0);
                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_vb3, _va3), _sum0);

                va += 4;
                vb += 32;
            }

            for (; k < K; k++) {
                __m256i _va0 = _mm256_set1_epi32(*va);
                __m256i _vb0 = _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)vb));

                _sum0 = _mm256_add_epi32(_mm256_mullo_epi32(_vb0, _va0), _sum0);

                va += 1;
                vb += 8;
            }

            _mm256_storeu_si256((__m256i* )output, _sum0);
#else
            int32_t sum[8] = {0};

            for (int k = 0; k < K; k++)
            {
                for (int n = 0; n < 8; n++)
                {
                    sum[n] += va[0] * vb[n];
                }

                va += 1;
                vb += 8;
            }

            for (int n = 0; n < 8; n++)
            {
                output[n] = sum[n];
            }
#endif
            output += 8;
        }

        for (; j < N; j++)
        {
            int8_t* va = pA_t + (i / 8 + (i % 8) / 4 + i % 4) * 8 * K;
            int8_t* vb = pB_t + (j / 8 + j % 8) * 8 * K;

            int k = 0;
            int32_t sum0 = 0.f;

            for (; k < K; k++)
            {
                sum0 += va[0] * vb[0];

                va += 1;
                vb += 1;
            }
            output[0] = sum0;

            output++;
        }
    }
}

static void sgemm_fp32(struct tensor* input, struct tensor* filter, struct tensor* bias,
                       struct tensor* output, struct conv_priv_info* priv_info, struct conv_param* param, int n,
                       int group, int num_thread)
{
    int kernel_size = param->kernel_h * param->kernel_w * param->input_channel / param->group;
    int outchan_g = param->output_channel / param->group;

    int out_h = output->dims[2];
    int out_w = output->dims[3];
    int out_image_size = output->dims[1] * output->dims[2] * output->dims[3];

    float* interleave_fp32 = ( float* )priv_info->interleave_buffer_pack4 + outchan_g * group * kernel_size;
    float* im2col_pack4_fp32 = priv_info->im2col_buffer_pack4;
    float* output_fp32 = ( float* )output->data + n * out_image_size + outchan_g * group * out_h * out_w;
    float* bias_fp32 = NULL;

    if (bias)
        bias_fp32 = ( float* )bias->data + outchan_g * group;

    float* filter_sgemm = interleave_fp32;
    float* input_sgemm_pack4 = im2col_pack4_fp32;
    float* output_sgemm = output_fp32;

    sgemm_fp(outchan_g, out_h * out_w, kernel_size, filter_sgemm, input_sgemm_pack4, output_sgemm, num_thread);

    // process bias
    if (bias)
    {
        for (int i = 0; i < outchan_g; i++)
        {
            for (int j = 0; j < out_h * out_w; j++)
            {
                int output_off = i * (out_h * out_w) + j;
                output_fp32[output_off] += bias_fp32[i];
            }
        }
    }

    // process activation relu
    if (param->activation == 0)
    {
        for (int i = 0; i < outchan_g; i++)
        {
            for (int j = 0; j < out_h * out_w; j++)
            {
                int output_off = i * (out_h * out_w) + j;

                if (output_fp32[output_off] < 0)
                    output_fp32[output_off] = 0;
            }
        }
    }

    // process activation relu6
    if (param->activation > 0)
    {
        for (int i = 0; i < outchan_g; i++)
        {
            for (int j = 0; j < out_h * out_w; j++)
            {
                int output_off = i * (out_h * out_w) + j;

                if (output_fp32[output_off] < 0)
                    output_fp32[output_off] = 0;
                if (output_fp32[output_off] > 6)
                    output_fp32[output_off] = 6;
            }
        }
    }
}

static void sgemm_uint8(struct tensor* input, struct tensor* filter, struct tensor* bias,
                        struct tensor* output, struct conv_priv_info* priv_info, struct conv_param* param, int n,
                        int group, int num_thread)
{
    int kernel_size = param->kernel_h * param->kernel_w * param->input_channel / param->group;
    int outchan_g = param->output_channel / param->group;

    int out_h = output->dims[2];
    int out_w = output->dims[3];
    int out_image_size = output->dims[1] * output->dims[2] * output->dims[3];

    float* interleave_fp32 = ( float* )priv_info->interleave_buffer_pack4 + outchan_g * group * kernel_size;
    float* im2col_pack4_fp32 = priv_info->im2col_buffer_pack4;
    uint8_t * output_uint8 = ( uint8_t* )output->data + n * out_image_size + outchan_g * group * out_h * out_w;
    int* bias_int32 = NULL;
    float bias_scale = 0.f;

    if (bias)
    {
        bias_int32 = ( int* )bias->data + outchan_g * group;
        bias_scale = input->scale * filter->scale;
    }

    float* filter_sgemm = interleave_fp32;
    float* input_sgemm_pack4 = im2col_pack4_fp32;
    float* output_sgemm = (float*)sys_malloc((unsigned long)outchan_g * out_h * out_w * sizeof(float));

    sgemm_fp(outchan_g, out_h * out_w, kernel_size, filter_sgemm, input_sgemm_pack4, output_sgemm, num_thread);

    /* process bias */
    if (bias)
    {
        for (int i = 0; i < outchan_g; i++)
        {
            for (int j = 0; j < out_h * out_w; j++)
            {
                int output_off = i * (out_h * out_w) + j;
                output_sgemm[output_off] += (float )bias_int32[i] * bias_scale;
            }
        }
    }

    /* process activation relu */
    if (param->activation == 0)
    {
        for (int i = 0; i < outchan_g; i++)
        {
            for (int j = 0; j < out_h * out_w; j++)
            {
                int output_off = i * (out_h * out_w) + j;

                if (output_sgemm[output_off] < 0)
                    output_sgemm[output_off] = 0;
            }
        }
    }

    /* process activation relu6 */
    if (param->activation > 0)
    {
        for (int i = 0; i < outchan_g; i++)
        {
            for (int j = 0; j < out_h * out_w; j++)
            {
                int output_off = i * (out_h * out_w) + j;

                if (output_sgemm[output_off] < 0)
                    output_sgemm[output_off] = 0;
                if (output_sgemm[output_off] > 6)
                    output_sgemm[output_off] = 6;
            }
        }
    }

    /* quant from fp32 to uint8 */
    for (int i = 0; i < outchan_g; i++)
    {
        for (int j = 0; j < out_h * out_w; j++)
        {
            int output_off = i * (out_h * out_w) + j;

            int udata = ( int )(round(output_sgemm[output_off] / output->scale) + output->zero_point);
            if (udata > 255)
                udata = 255;
            else if (udata < 0)
                udata = 0;
            output_uint8[output_off] = udata;
        }
    }

    sys_free(output_sgemm);
}

static void sgemm_int8(struct tensor* input, struct tensor* filter, struct tensor* bias,
                        struct tensor* output, struct conv_priv_info* priv_info, struct conv_param* param, int n,
                        int group, int num_thread)
{
    int kernel_size = param->kernel_h * param->kernel_w * param->input_channel / param->group;
    int outchan_g = param->output_channel / param->group;

    int out_h = output->dims[2];
    int out_w = output->dims[3];
    int out_image_size = output->dims[1] * output->dims[2] * output->dims[3];

    int8_t* interleave_int8 = ( int8_t* )priv_info->interleave_buffer_pack4 + outchan_g * group * kernel_size;
    int8_t* im2col_pack4_int8 = priv_info->im2col_buffer_pack4;
    int8_t * output_int8 = ( int8_t* )output->data + n * out_image_size + outchan_g * group * out_h * out_w;
    int32_t * bias_int32 = NULL;

    if (bias)
        bias_int32 = ( int* )bias->data + outchan_g * group;

    float input_scale = input->scale;
    float* kernel_scales = filter->scale_list;
    float output_scale = output->scale;

    int8_t* filter_sgemm = interleave_int8;
    int8_t* input_sgemm_pack4 = im2col_pack4_int8;
    int32_t* output_sgemm_int32 = (int32_t*)sys_malloc((unsigned long)outchan_g * out_h * out_w * sizeof(int32_t));
    float* output_sgemm_fp32 = (float*)sys_malloc((unsigned long)outchan_g * out_h * out_w * sizeof(float));

    sgemm_i8(outchan_g, out_h * out_w, kernel_size, filter_sgemm, input_sgemm_pack4, output_sgemm_int32, num_thread);

    /* process bias and dequant output from int32 to fp32 */
#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < outchan_g; i++)
    {
        for (int j = 0; j < out_h * out_w; j++)
        {
            int output_off = i * (out_h * out_w) + j;
            if (bias)
                output_sgemm_fp32[output_off] = (float )(output_sgemm_int32[output_off] + bias_int32[i]) * input_scale * kernel_scales[i];
            else
                output_sgemm_fp32[output_off] = (float )output_sgemm_int32[output_off] * input_scale * kernel_scales[i];
        }
    }

    /* process activation relu */
    if (param->activation == 0)
    {
#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < outchan_g; i++)
        {
            for (int j = 0; j < out_h * out_w; j++)
            {
                int output_off = i * (out_h * out_w) + j;

                if (output_sgemm_fp32[output_off] < 0)
                    output_sgemm_fp32[output_off] = 0;
            }
        }
    }

    /* process activation relu6 */
    if (param->activation > 0)
    {
#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < outchan_g; i++)
        {
            for (int j = 0; j < out_h * out_w; j++)
            {
                int output_off = i * (out_h * out_w) + j;

                if (output_sgemm_fp32[output_off] < 0)
                    output_sgemm_fp32[output_off] = 0;
                if (output_sgemm_fp32[output_off] > 6)
                    output_sgemm_fp32[output_off] = 6;
            }
        }
    }

    /* quant from fp32 to int8 */
    for (int i = 0; i < outchan_g; i++)
    {
#pragma omp parallel for num_threads(num_thread)
        for (int j = 0; j < out_h * out_w; j++)
        {
            int output_off = i * (out_h * out_w) + j;

            int32_t data_i32 = ( int32_t )(round(output_sgemm_fp32[output_off] / output_scale));
            if (data_i32 > 127)
                data_i32 = 127;
            else if (data_i32 < -127)
                data_i32 = -127;
            output_int8[output_off] = (int8_t)data_i32;
        }
    }

    sys_free(output_sgemm_int32);
    sys_free(output_sgemm_fp32);
}

/* check the conv wheather need to be using winograd */
static int winograd_support(struct conv_param* param, int in_h, int in_w)
{
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    int input_chan = param->input_channel;
    int output_chan = param->output_channel;
    int group = param->group;

    if (in_h <= 10 && in_w <= 10)
        return 0;

    if (group != 1 || kernel_h != 3 || kernel_w != 3 || stride_h != 1 || stride_w != 1 || dilation_h != 1 ||
        dilation_w != 1 || input_chan < 16 || output_chan < 16 || output_chan % 16)
        return 0;

    return 1;
}

int conv_hcl_get_shared_mem_size(struct tensor* input, struct tensor* output, struct conv_param* param)
{
    int group = param->group;
    int input_chan = param->input_channel / group;
    int kernel_size = input_chan * param->kernel_h * param->kernel_w;
    int output_xy = output->dims[2] * output->dims[3];
    int elem_size = input->elem_size;

    // simulator uint8 inference with fp32
    if (input->data_type == TENGINE_DT_UINT8)
        elem_size = 4;

    return elem_size * output_xy * kernel_size;
}

int conv_hcl_get_shared_pack4_mem_size(struct tensor* filter, struct tensor* output, struct conv_param* param)
{
    int K = filter->elem_num / filter->dims[0];
    int N = output->dims[2] * output->dims[3];
    int elem_size = filter->elem_size;

    // simulator uint8 inference with fp32
    if (filter->data_type == TENGINE_DT_UINT8)
        elem_size = 4;

    return (8 * K * (N / 8 + N % 8)) * elem_size;
}

int conv_hcl_get_interleave_pack4_size(int M, int K, struct tensor* filter)
{
    int elem_size = filter->elem_size;

    // simulator uint8 inference with fp32
    if (filter->data_type == TENGINE_DT_UINT8)
        elem_size = 4;

    int size = 8 * K * (M / 8 + (M % 8) / 4 + M % 4) * elem_size;
    return size;
}

void conv_hcl_interleave_pack4_fp32(int M, int K, struct conv_priv_info* priv_info)
{
    float* pA = ( float* )priv_info->interleave_buffer;
    float* pA_t = ( float* )priv_info->interleave_buffer_pack4;

    int nn_outch = M >> 3;
    int remain_outch_start = nn_outch << 3;

    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        const float* k0 = pA + (p + 0) * K;
        const float* k1 = pA + (p + 1) * K;
        const float* k2 = pA + (p + 2) * K;
        const float* k3 = pA + (p + 3) * K;
        const float* k4 = pA + (p + 4) * K;
        const float* k5 = pA + (p + 5) * K;
        const float* k6 = pA + (p + 6) * K;
        const float* k7 = pA + (p + 7) * K;

        float* ktmp = pA_t + (p / 8) * 8 * K;

        for (int q = 0; q < K; q++)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k1[0];
            ktmp[2] = k2[0];
            ktmp[3] = k3[0];
            ktmp[4] = k4[0];
            ktmp[5] = k5[0];
            ktmp[6] = k6[0];
            ktmp[7] = k7[0];
            ktmp += 8;

            k0 += 1;
            k1 += 1;
            k2 += 1;
            k3 += 1;
            k4 += 1;
            k5 += 1;
            k6 += 1;
            k7 += 1;
        }
    }

    nn_outch = (M - remain_outch_start) >> 2;
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        const float* k0 = pA + (p + 0) * K;
        const float* k1 = pA + (p + 1) * K;
        const float* k2 = pA + (p + 2) * K;
        const float* k3 = pA + (p + 3) * K;

        float* ktmp = pA_t + (p / 8 + (p % 8) / 4) * 8 * K;

        for (int q = 0; q < K; q++)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k1[0];
            ktmp[2] = k2[0];
            ktmp[3] = k3[0];
            ktmp += 4;

            k0 += 1;
            k1 += 1;
            k2 += 1;
            k3 += 1;
        }
    }

    remain_outch_start += nn_outch << 2;

    for (int p = remain_outch_start; p < M; p++)
    {
        const float* k0 = pA + (p + 0) * K;

        float* ktmp = pA_t + (p / 8 + (p % 8) / 4 + p % 4) * 8 * K;

        for (int q = 0; q < K; q++)
        {
            ktmp[0] = k0[0];
            ktmp++;
            k0++;
        }
    }
}

void conv_hcl_interleave_pack4_int8(int M, int K, struct conv_priv_info* priv_info)
{
    int8_t* pA = ( int8_t * )priv_info->interleave_buffer;
    int8_t* pA_t = ( int8_t* )priv_info->interleave_buffer_pack4;

    int nn_outch = M >> 3;
    int remain_outch_start = nn_outch << 3;

    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        const int8_t* k0 = pA + (p + 0) * K;
        const int8_t* k1 = pA + (p + 1) * K;
        const int8_t* k2 = pA + (p + 2) * K;
        const int8_t* k3 = pA + (p + 3) * K;
        const int8_t* k4 = pA + (p + 4) * K;
        const int8_t* k5 = pA + (p + 5) * K;
        const int8_t* k6 = pA + (p + 6) * K;
        const int8_t* k7 = pA + (p + 7) * K;

        int8_t* ktmp = pA_t + (p / 8) * 8 * K;

        for (int q = 0; q < K; q++)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k1[0];
            ktmp[2] = k2[0];
            ktmp[3] = k3[0];
            ktmp[4] = k4[0];
            ktmp[5] = k5[0];
            ktmp[6] = k6[0];
            ktmp[7] = k7[0];
            ktmp += 8;

            k0 += 1;
            k1 += 1;
            k2 += 1;
            k3 += 1;
            k4 += 1;
            k5 += 1;
            k6 += 1;
            k7 += 1;
        }
    }

    nn_outch = (M - remain_outch_start) >> 2;
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        const int8_t* k0 = pA + (p + 0) * K;
        const int8_t* k1 = pA + (p + 1) * K;
        const int8_t* k2 = pA + (p + 2) * K;
        const int8_t* k3 = pA + (p + 3) * K;

        int8_t* ktmp = pA_t + (p / 8 + (p % 8) / 4) * 8 * K;

        for (int q = 0; q < K; q++)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k1[0];
            ktmp[2] = k2[0];
            ktmp[3] = k3[0];
            ktmp += 4;

            k0 += 1;
            k1 += 1;
            k2 += 1;
            k3 += 1;
        }
    }

    remain_outch_start += nn_outch << 2;

    for (int p = remain_outch_start; p < M; p++)
    {
        const int8_t* k0 = pA + (p + 0) * K;

        int8_t* ktmp = pA_t + (p / 8 + (p % 8) / 4 + p % 4) * 8 * K;

        for (int q = 0; q < K; q++)
        {
            ktmp[0] = k0[0];
            ktmp++;
            k0++;
        }
    }
}

int conv_hcl_prerun(struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* output_tensor,
                    struct conv_priv_info* priv_info, struct conv_param* param)
{
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];

    /* check winograd implement, only for conv3x3s1 */
    if (input_tensor->data_type == TENGINE_DT_FP32)
    {
        priv_info->winograd = winograd_support(param, in_h, in_w);
        if (priv_info->winograd)
        {
            return wino_conv_hcl_prerun(input_tensor, filter_tensor, output_tensor, priv_info, param);
        }
    }

    if (!priv_info->external_im2col_mem)
    {
        int mem_size = conv_hcl_get_shared_mem_size(input_tensor, output_tensor, param);
        void* mem = sys_malloc(mem_size);
        priv_info->im2col_buffer = mem;
        priv_info->im2col_buffer_size = mem_size;
    }
    if (!priv_info->external_im2col_pack4_mem)
    {
        int mem_size = conv_hcl_get_shared_pack4_mem_size(filter_tensor, output_tensor, param);
        void* mem = sys_malloc(mem_size);
        priv_info->im2col_buffer_pack4 = mem;
        priv_info->im2col_buffer_pack4_size = mem_size;
    }

    if (!priv_info->external_interleave_mem)
    {
        int mem_size = get_private_mem_size(filter_tensor);
        void* mem = sys_malloc(mem_size);
        priv_info->interleave_buffer = mem;
        priv_info->interleave_buffer_size = mem_size;
    }

    if (input_tensor->data_type == TENGINE_DT_UINT8)
        interleave_uint8(filter_tensor, priv_info);
    else
        interleave(filter_tensor, priv_info);

    if (priv_info->external_interleave_pack4_mem)
    {
        int M = filter_tensor->dims[0];
        int K = filter_tensor->elem_num / filter_tensor->dims[0];

        int mem_size = conv_hcl_get_interleave_pack4_size(M, K, filter_tensor);
        void* mem = sys_malloc(mem_size);
        priv_info->interleave_buffer_pack4 = mem;
        priv_info->interleave_buffer_pack4_size = mem_size;

        if (input_tensor->data_type == TENGINE_DT_FP32 || input_tensor->data_type == TENGINE_DT_UINT8)
            conv_hcl_interleave_pack4_fp32(M, K, priv_info);
        else
            conv_hcl_interleave_pack4_int8(M, K, priv_info);

        if (!priv_info->external_interleave_mem && priv_info->interleave_buffer)
        {
            sys_free(priv_info->interleave_buffer);
            priv_info->interleave_buffer = NULL;
        }
    }
    else
    {
        priv_info->interleave_buffer_pack4 = priv_info->interleave_buffer;
        priv_info->interleave_buffer_pack4_size = priv_info->interleave_buffer_size;
    }

    return 0;
}

int conv_hcl_postrun(struct conv_priv_info* priv_info)
{
    if (priv_info->winograd)
    {
        return wino_conv_hcl_postrun(priv_info);
    }

    if (priv_info->external_interleave_pack4_mem && !priv_info->external_interleave_mem &&
        priv_info->interleave_buffer != NULL)
    {
        sys_free(priv_info->interleave_buffer_pack4);
        priv_info->interleave_buffer_pack4 = NULL;
    }

    if (!priv_info->external_im2col_mem && priv_info->im2col_buffer != NULL)
    {
        sys_free(priv_info->im2col_buffer);
        priv_info->im2col_buffer = NULL;
    }
    if (!priv_info->external_im2col_pack4_mem && priv_info->im2col_buffer_pack4 != NULL)
    {
        sys_free(priv_info->im2col_buffer_pack4);
        priv_info->im2col_buffer_pack4 = NULL;
    }
    if (priv_info->external_interleave_pack4_mem && priv_info->interleave_buffer_pack4 != NULL)
    {
        sys_free(priv_info->interleave_buffer_pack4);
        priv_info->interleave_buffer_pack4 = NULL;
    }

    return 0;
}

int conv_hcl_run(struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* bias_tensor,
                 struct tensor* output_tensor, struct conv_priv_info* priv_info, struct conv_param* param,
                 int num_thread, int cpu_affinity)
{
    int group = param->group;
    int type = input_tensor->data_type;

    if (priv_info->winograd)
    {
        return wino_conv_hcl_run(input_tensor, filter_tensor, bias_tensor, output_tensor, priv_info, param, num_thread,
                                 cpu_affinity);
    }

    for (int i = 0; i < input_tensor->dims[0]; i++)    // batch size
    {
        for (int j = 0; j < group; j++)
        {
            im2col_ir(input_tensor, output_tensor, priv_info, param, i, j);

            int K = filter_tensor->elem_num / filter_tensor->dims[0];
            int N = output_tensor->dims[2] * output_tensor->dims[3];

            void* im2col_buffer = priv_info->im2col_buffer;

            if (priv_info->external_interleave_pack4_mem)
            {
                if (type == TENGINE_DT_FP32 || type == TENGINE_DT_UINT8)
                    input_pack4_fp32(K, N, im2col_buffer, priv_info->im2col_buffer_pack4, num_thread);
                else
                    input_pack4_int8(K, N, im2col_buffer, priv_info->im2col_buffer_pack4, num_thread);
            }
            else
            {
                priv_info->im2col_buffer_pack4 = im2col_buffer;
            }

            if (type == TENGINE_DT_FP32)
                sgemm_fp32(input_tensor, filter_tensor, bias_tensor, output_tensor, priv_info, param, i, j, num_thread);
            else if (type == TENGINE_DT_UINT8)
                sgemm_uint8(input_tensor, filter_tensor, bias_tensor, output_tensor, priv_info, param, i, j, num_thread);
            else if (type == TENGINE_DT_INT8)
                sgemm_int8(input_tensor, filter_tensor, bias_tensor, output_tensor, priv_info, param, i, j, num_thread);
            else
            {
                TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);
                return -1;
            }
        }
    }

    return 0;
}

int conv_hcl_set_shared_mem(struct conv_priv_info* priv_info, void* mem, int mem_size)
{
    priv_info->external_im2col_mem = 1;
    priv_info->im2col_buffer = mem;
    priv_info->im2col_buffer_size = mem_size;
    return 0;
}

int conv_hcl_set_shared_pack4_mem(struct conv_priv_info* priv_info, void* mem, int mem_size)
{
    priv_info->external_im2col_pack4_mem = 1;
    priv_info->im2col_buffer_pack4 = mem;
    priv_info->im2col_buffer_pack4_size = mem_size;

    return 0;
}
