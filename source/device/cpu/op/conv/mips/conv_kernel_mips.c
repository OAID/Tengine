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

#include "conv_kernel_mips.h"

#include "wino_conv_kernel_mips.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#if __mips_msa
#include <msa.h>
#endif
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static int get_private_mem_size(struct tensor* filter)
{
    return filter->elem_num * filter->elem_size; // caution
}

static void interleave(struct tensor* filter, struct conv_priv_info* priv_info)
{
    /* simply copy the data */
    memcpy(priv_info->interleave_buffer, filter->data, filter->elem_num * filter->elem_size);
}

void im2col(float* data_img, float* data_col, int inh, int inw, int inc, int outh, int outw, int outc, int ksize_h,
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

static void im2col_ir(struct tensor* input, struct tensor* output, struct conv_priv_info* priv_info,
                      struct conv_param* param, int n, int group)
{
    int input_chan = param->input_channel / param->group;
    int image_size = input->dims[1] * input->dims[2] * input->dims[3];
    int group_size = input_chan * input->dims[2] * input->dims[3];

    void* input_base = input->data + (n * image_size + group * group_size) * input->elem_size;
    void* im2col_buf = priv_info->im2col_buffer;

    int input_zero = 0;

    if (input->data_type == TENGINE_DT_UINT8)
        input_zero = input->zero_point;

    im2col(input_base, im2col_buf, input->dims[2], input->dims[3], input_chan, output->dims[2], output->dims[3],
           output->dims[1] / param->group, param->kernel_h, param->kernel_w, param->stride_h, param->stride_w,
           param->pad_h0, param->pad_w0, param->dilation_h, param->dilation_w);
}

void input_pack4(int K, int N, float* pB, float* pB_t, int num_thread)
{
    int nn_size = N >> 2;
    int remian_size_start = nn_size << 2;

// [ch00, ch10, ch20, ch30, ch01, ch11, ch21, ch31, ch02, ch12, ch22, ch32, ch03, ch13, ch23, ch33 ....]
#pragma omp parallel for num_threads(num_thread)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = ii * 4;
        const float* img = pB + i;
        float* tmp = pB_t + (i / 4) * 4 * K;

        for (int j = 0; j < K; j++)
        {
#if __mips_msa
            __msa_st_w(__msa_ld_w(img, 0), tmp, 0);
#else
            tmp[0] = img[0];
            tmp[1] = img[1];
            tmp[2] = img[2];
            tmp[3] = img[3];
#endif // __mips_msa
            tmp += 4;
            img += N;
        }
    }

// [ch00, ch01, ch02, ch03 ....]
#pragma omp parallel for num_threads(num_thread)
    for (int i = remian_size_start; i < N; i++)
    {
        const float* img = pB + i;
        float* tmp = pB_t + (i / 4 + i % 4) * 4 * K;

        for (int j = 0; j < K; j++)
        {
            tmp[0] = img[0];

            tmp += 1;
            img += N;
        }
    }
}

// unloop output M, unloop N, packet 4x4, using intrinsic
static void sgemm(int M, int N, int K, float* pA_t, float* pB_t, float* pC, int num_thread)
{
    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = M >> 2;
    remain_outch_start = nn_outch << 2;

// output ch0 - ch3
#pragma omp parallel for num_threads(num_thread)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int i = pp * 4;

        float* output0 = pC + (i)*N;
        float* output1 = pC + (i + 1) * N;
        float* output2 = pC + (i + 2) * N;
        float* output3 = pC + (i + 3) * N;

        int j = 0;
        for (; j + 3 < N; j += 4)
        {
            float* va = pA_t + (i / 4) * 4 * K;
            float* vb = pB_t + (j / 4) * 4 * K;

#if __mips_msa
            v4f32 _sum0 = {0.f};
            v4f32 _sum1 = {0.f};
            v4f32 _sum2 = {0.f};
            v4f32 _sum3 = {0.f};

            for (int k = 0; k < K; k++)
            {
                // k0
                __builtin_prefetch(vb + 32);
                __builtin_prefetch(va + 32);
                v4f32 _vb = (v4f32)__msa_ld_w(vb, 0);
                v4i32 _va0123 = __msa_ld_w(va, 0);

                _sum0 = __msa_fmadd_w(_sum0, _vb, (v4f32)__msa_splati_w(_va0123, 0)); // sum0 = (a00-a03) * k00
                _sum1 = __msa_fmadd_w(_sum1, _vb, (v4f32)__msa_splati_w(_va0123, 1)); // sum1 = (a00-a03) * k10
                _sum2 = __msa_fmadd_w(_sum2, _vb, (v4f32)__msa_splati_w(_va0123, 2)); // sum2 = (a00-a03) * k20
                _sum3 = __msa_fmadd_w(_sum3, _vb, (v4f32)__msa_splati_w(_va0123, 3)); // sum3 = (a00-a03) * k30

                va += 4;
                vb += 4;
            }
            __msa_st_w((v4i32)_sum0, output0, 0);
            __msa_st_w((v4i32)_sum1, output1, 0);
            __msa_st_w((v4i32)_sum2, output2, 0);
            __msa_st_w((v4i32)_sum3, output3, 0);
#else
            float sum0[4] = {0};
            float sum1[4] = {0};
            float sum2[4] = {0};
            float sum3[4] = {0};

            for (int k = 0; k < K; k++)
            {
                for (int n = 0; n < 4; n++)
                {
                    sum0[n] += va[0] * vb[n];
                    sum1[n] += va[1] * vb[n];
                    sum2[n] += va[2] * vb[n];
                    sum3[n] += va[3] * vb[n];
                }

                va += 4;
                vb += 4;
            }

            for (int n = 0; n < 4; n++)
            {
                output0[n] = sum0[n];
                output1[n] = sum1[n];
                output2[n] = sum2[n];
                output3[n] = sum3[n];
            }
#endif // __mips_msa
            output0 += 4;
            output1 += 4;
            output2 += 4;
            output3 += 4;
        }

        for (; j < N; j++)
        {
            float* va = pA_t + (i / 4) * 4 * K;
            float* vb = pB_t + (j / 4 + j % 4) * 4 * K;

#if __mips_msa
            v4f32 _sum0_3 = {0.f};
            v4f32 _sum0 = {0.f};
            v4f32 _sum1 = {0.f};
            v4f32 _sum2 = {0.f};
            v4f32 _sum3 = {0.f};

            int k = 0;
            for (; k + 3 < K; k = k + 4)
            {
                __builtin_prefetch(vb + 32);
                __builtin_prefetch(va + 128);
                v4i32 _vb0123 = __msa_ld_w(vb, 0);
                v4f32 _va0 = (v4f32)__msa_ld_w(va, 0);
                v4f32 _va1 = (v4f32)__msa_ld_w(va + 4, 0);
                v4f32 _va2 = (v4f32)__msa_ld_w(va + 8, 0);
                v4f32 _va3 = (v4f32)__msa_ld_w(va + 12, 0);

                _sum0 = __msa_fmadd_w(_sum0, _va0, (v4f32)__msa_splati_w(_vb0123, 0)); // sum0 += (k00-k30) * a00
                _sum1 = __msa_fmadd_w(_sum1, _va1, (v4f32)__msa_splati_w(_vb0123, 1)); // sum1 += (k01-k31) * a10
                _sum2 = __msa_fmadd_w(_sum2, _va2, (v4f32)__msa_splati_w(_vb0123, 2)); // sum2 += (k02-k32) * a20
                _sum3 = __msa_fmadd_w(_sum3, _va3, (v4f32)__msa_splati_w(_vb0123, 3)); // sum3 += (k03-k33) * a30

                va += 16;
                vb += 4;
            }

            _sum0 = __msa_fadd_w(_sum0, _sum1);
            _sum2 = __msa_fadd_w(_sum2, _sum3);
            _sum0_3 = __msa_fadd_w(_sum2, _sum0);
            // _sum0_3 = __msa_fadd_w(_sum0_3, _sum2);

            for (; k < K; k++)
            {
                v4f32 _vb0 = {vb[0], vb[0], vb[0], vb[0]};
                v4f32 _va = (v4f32)__msa_ld_w(va, 0);

                _sum0_3 = __msa_fmadd_w(_sum0_3, _va, _vb0); // sum0 += (k00-k30) * a00

                va += 4;
                vb += 1;
            }
            output0[0] = _sum0_3[0];
            output1[0] = _sum0_3[1];
            output2[0] = _sum0_3[2];
            output3[0] = _sum0_3[3];
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
#endif // __mips_msa
            output0++;
            output1++;
            output2++;
            output3++;
        }
    }

// output ch0
#pragma omp parallel for num_threads(num_thread)
    for (int i = remain_outch_start; i < M; i++)
    {
        float* output = pC + i * N;

        int j = 0;
        for (; j + 3 < N; j += 4)
        {
            float* va = pA_t + (i / 4 + i % 4) * 4 * K;
            float* vb = pB_t + (j / 4) * 4 * K;
#if __mips_msa
            v4f32 _sum0 = {0.f};

            int k = 0;
            for (; k + 3 < K; k = k + 4)
            {
                // k0
                __builtin_prefetch(va + 32);
                __builtin_prefetch(vb + 128);
                v4i32 _va0123 = __msa_ld_w(va, 0);
                v4f32 _vb0 = (v4f32)__msa_ld_w(vb, 0);
                v4f32 _vb1 = (v4f32)__msa_ld_w(vb + 4, 0);
                v4f32 _vb2 = (v4f32)__msa_ld_w(vb + 8, 0);
                v4f32 _vb3 = (v4f32)__msa_ld_w(vb + 12, 0);

                _sum0 = __msa_fmadd_w(_sum0, _vb0, (v4f32)__msa_splati_w(_va0123, 0)); // sum0 = (a00-a03) * k00
                _sum0 = __msa_fmadd_w(_sum0, _vb1, (v4f32)__msa_splati_w(_va0123, 1)); // sum0 += (a10-a13) * k01
                _sum0 = __msa_fmadd_w(_sum0, _vb2, (v4f32)__msa_splati_w(_va0123, 2)); // sum0 += (a20-a23) * k02
                _sum0 = __msa_fmadd_w(_sum0, _vb3, (v4f32)__msa_splati_w(_va0123, 3)); // sum0 += (a30-a33) * k03

                va += 4;
                vb += 16;
            }

            for (; k < K; k++)
            {
                // k0
                v4f32 _va0 = {va[0]};
                v4f32 _vb0 = (v4f32)__msa_ld_w(vb, 0);

                _sum0 = __msa_fmadd_w(_sum0, _vb0, _va0); // sum0 = (a00-a03) * k00

                va += 1;
                vb += 4;
            }
            __msa_st_w((v4i32)_sum0, output, 0);
#else
            float sum[4] = {0};

            for (int k = 0; k < K; k++)
            {
                for (int n = 0; n < 4; n++)
                {
                    sum[n] += va[0] * vb[n];
                }

                va += 1;
                vb += 4;
            }

            for (int n = 0; n < 4; n++)
            {
                output[n] = sum[n];
            }
#endif // __mips_msa
            output += 4;
        }

        for (; j < N; j++)
        {
            float* va = pA_t + (i / 4 + i % 4) * 4 * K;
            float* vb = pB_t + (j / 4 + j % 4) * 4 * K;

            int k = 0;
#if __mips_msa
            v4f32 _sum0 = {0.f};

            for (; k + 3 < K; k += 4)
            {
                __builtin_prefetch(vb + 32);
                __builtin_prefetch(va + 32);
                v4f32 _p0 = (v4f32)__msa_ld_w(vb, 0);
                v4f32 _k0 = (v4f32)__msa_ld_w(va, 0);
                _sum0 = __msa_fmadd_w(_sum0, _p0, _k0);

                va += 4;
                vb += 4;
            }
            float sum0 = _sum0[0] + _sum0[1] + _sum0[2] + _sum0[3];
#else
            float sum0 = 0.f;
#endif // __mips_msa
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

    float* interleave_fp32 = (float*)priv_info->interleave_buffer_pack4 + outchan_g * group * kernel_size;
    float* im2col_pack4_fp32 = priv_info->im2col_buffer_pack4;
    float* output_fp32 = (float*)output->data + n * out_image_size + outchan_g * group * out_h * out_w;
    float* bias_fp32 = NULL;

    if (bias)
        bias_fp32 = (float*)bias->data + outchan_g * group;

    float* filter_sgemm = interleave_fp32;
    float* input_sgemm_pack4 = im2col_pack4_fp32;
    float* output_sgemm = output_fp32;

    sgemm(outchan_g, out_h * out_w, kernel_size, filter_sgemm, input_sgemm_pack4, output_sgemm, num_thread);

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

    if (group != 1 || kernel_h != 3 || kernel_w != 3 || stride_h != 1 || stride_w != 1 || dilation_h != 1 || dilation_w != 1 || input_chan < 16 || output_chan < 16)
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

    return elem_size * output_xy * kernel_size;
}

int conv_hcl_get_shared_pack4_mem_size(struct tensor* filter, struct tensor* output, struct conv_param* param)
{
    int K = filter->elem_num / filter->dims[0];
    int N = output->dims[2] * output->dims[3];
    int elem_size = filter->elem_size;

    return (4 * K * (N / 4 + N % 4)) * elem_size;
}

int conv_hcl_get_interleave_pack4_size(int M, int K, struct tensor* filter)
{
    int size = 4 * K * (M / 4 + M % 4) * filter->elem_size;
    return size;
}

void conv_hcl_interleave_pack4(int M, int K, struct conv_priv_info* priv_info)
{
    float* pA = (float*)priv_info->interleave_buffer;
    float* pA_t = (float*)priv_info->interleave_buffer_pack4;

    int nn_outch = M >> 2;
    int remain_outch_start = nn_outch << 2;

    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        const float* k0 = pA + (p + 0) * K;
        const float* k1 = pA + (p + 1) * K;
        const float* k2 = pA + (p + 2) * K;
        const float* k3 = pA + (p + 3) * K;

        float* ktmp = pA_t + (p / 4) * 4 * K;

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

    for (int p = remain_outch_start; p < M; p++)
    {
        const float* k0 = pA + (p + 0) * K;

        float* ktmp = pA_t + (p / 4 + p % 4) * 4 * K;

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
    priv_info->winograd = winograd_support(param, in_h, in_w);
    if (priv_info->winograd)
    {
        return wino_conv_hcl_prerun(input_tensor, filter_tensor, output_tensor, priv_info, param);
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

    interleave(filter_tensor, priv_info);

    if (priv_info->external_interleave_pack4_mem)
    {
        int M = filter_tensor->dims[0];
        int K = filter_tensor->elem_num / filter_tensor->dims[0];

        int mem_size = conv_hcl_get_interleave_pack4_size(M, K, filter_tensor);
        void* mem = sys_malloc(mem_size);
        priv_info->interleave_buffer_pack4 = mem;
        priv_info->interleave_buffer_pack4_size = mem_size;

        conv_hcl_interleave_pack4(M, K, priv_info);

        if (!priv_info->external_interleave_mem && priv_info->interleave_buffer)
        {
            sys_free(priv_info->interleave_buffer);
            priv_info->interleave_buffer = NULL;
        }
    }

    return 0;
}

int conv_hcl_postrun(struct conv_priv_info* priv_info)
{
    if (priv_info->winograd)
    {
        return wino_conv_hcl_postrun(priv_info);
    }

    if (priv_info->external_interleave_pack4_mem && !priv_info->external_interleave_mem && priv_info->interleave_buffer != NULL)
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

    for (int i = 0; i < input_tensor->dims[0]; i++) // batch size
    {
        for (int j = 0; j < group; j++)
        {
            im2col_ir(input_tensor, output_tensor, priv_info, param, i, j);

            int K = filter_tensor->elem_num / filter_tensor->dims[0];
            int N = output_tensor->dims[2] * output_tensor->dims[3];

            float* im2col_fp32 = priv_info->im2col_buffer;
            float* im2col_pack4_fp32 = priv_info->im2col_buffer_pack4;
            input_pack4(K, N, im2col_fp32, im2col_pack4_fp32, num_thread);

            if (type == TENGINE_DT_FP32)
                sgemm_fp32(input_tensor, filter_tensor, bias_tensor, output_tensor, priv_info, param, i, j, num_thread);
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
