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
 * Author: xlchen@openailab.com
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <arm_neon.h>
#include <sys/time.h>

#include "conv_kernel_arm.h"
#include "compiler_fp16.h"

#define PER_OUT_CHAN 16

void hgemm_4x16_a76(__fp16* biases, __fp16* input, __fp16* kernel, long kernel_size, __fp16* output,
                    long output_xy, long fused_relu);
void hgemm_4x4_a76(__fp16* biases, __fp16* input, __fp16* kernel, long kernel_size, __fp16* output,
                   long output_xy, long fused_relu);

void im2col_fp16_1x1(__fp16* input, long input_xy, __fp16* col, long col_cnt, long input_chan);
void im2col_fp16_3x3(__fp16* input, long input_x, long input_y, long input_chan, __fp16* col, long stride);

void im2col(__fp16* im, __fp16* col, int input_chan, int input_x, int input_y, int kernel_x, int kernel_y, int stride_x,
            int stride_y, int dilation_x, int dilation_y, int pad_w0, int pad_w1, int pad_h0, int pad_h1, int output_x,
            int output_y, int col_start, int col_end)
{
    int kernel_size = kernel_x * kernel_y * input_chan;
    int input_xy = input_x * input_y;
    int pad_x = pad_w0;
    int pad_y = pad_h0;
    __fp16* cur_col = col + col_start * kernel_size;
    int col_i, col_j, kch, ky, kx, i;

    if ((kernel_x == 1) && (kernel_y == 1) && (stride_x == 1) && (stride_y == 1))
    {
        {
            int col_cnt = (col_end & -4) - (col_start & -4);
            im2col_fp16_1x1(im + col_start, input_xy, cur_col, col_cnt, input_chan);
            cur_col += col_cnt * kernel_size;
            col_i = col_end & -4;
        }
        // final 4 input
        if (col_end & 0x3)
        {
            for (col_j = 0; col_j < kernel_size; col_j++)
            {
                for (i = 0; i < 4; i++)
                {
                    if ((col_i + i) < col_end)
                        *cur_col++ = *(im + input_xy * col_j + col_i + i);
                    else
                        *cur_col++ = 0.0;
                }
            }
        }
    }
    else if ((kernel_x == 3) && (kernel_y == 3) && (dilation_x == 1) && (dilation_y == 1))
    {
        int is_pad0 = (pad_w0 == 0) && (pad_h0 == 0) && (pad_w1 == 0) && (pad_h1 == 0);
        for (col_i = (col_start & -4); col_i < (col_end & -4); col_i += 4)
        {
            cur_col = col + col_i * kernel_size;
            int imy0 = col_i / output_x;
            int imy3 = (col_i + 3) / output_x;
            int imx0 = col_i - imy0 * output_x;
            int imx3 = (col_i + 3) - imy3 * output_x;
            if ((imy0 == imy3) && (is_pad0 || (imy0 != 0 && imx0 != 0 && imy0 != (output_y - 1) && imx3 != (output_x - 1))))
            {
                __fp16* l0 = im + (imy0 * stride_y - pad_y) * input_x + (imx0 * stride_x - pad_x);

                {
                    im2col_fp16_3x3(l0, input_x, input_y, input_chan, cur_col, stride_x);
                    cur_col += 4 * kernel_size;
                }
            }
            else
            {
                int cnt_y[4] = {imy0, (col_i + 1) / output_x, (col_i + 2) / output_x, imy3};
                int cnt_x[4] = {imx0, col_i - cnt_y[1] * output_x + 1, col_i - cnt_y[2] * output_x + 2, imx3};
                int imx_start[4] = {cnt_x[0] * stride_x - pad_x, cnt_x[1] * stride_x - pad_x,
                                    cnt_x[2] * stride_x - pad_x, cnt_x[3] * stride_x - pad_x};
                int imy_start[4] = {cnt_y[0] * stride_y - pad_y, cnt_y[1] * stride_y - pad_y,
                                    cnt_y[2] * stride_y - pad_y, cnt_y[3] * stride_y - pad_y};
                for (kch = 0; kch < input_chan; kch++)
                    for (ky = 0; ky < 3; ky++)
                        for (kx = 0; kx < 3; kx++)
                        {
                            int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                            int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                            for (i = 0; i < 4; i++)
                            {
                                if (imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                else
                                    *cur_col++ = 0.0;
                            }
                        }
            }
        }
        // final 4 input
        if (col_end & 0x3)
        {
            int cnt_y[4] = {col_i / output_x, (col_i + 1) / output_x, (col_i + 2) / output_x, (col_i + 3) / output_x};
            int cnt_x[4] = {col_i - cnt_y[0] * output_x, col_i - cnt_y[1] * output_x + 1,
                            col_i - cnt_y[2] * output_x + 2, col_i - cnt_y[3] * output_x + 3};
            int imx_start[4] = {cnt_x[0] * stride_x - pad_x, cnt_x[1] * stride_x - pad_x, cnt_x[2] * stride_x - pad_x,
                                cnt_x[3] * stride_x - pad_x};
            int imy_start[4] = {cnt_y[0] * stride_y - pad_y, cnt_y[1] * stride_y - pad_y, cnt_y[2] * stride_y - pad_y,
                                cnt_y[3] * stride_y - pad_y};
            for (kch = 0; kch < input_chan; kch++)
                for (ky = 0; ky < 3; ky++)
                    for (kx = 0; kx < 3; kx++)
                    {
                        int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                        int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                        for (i = 0; i < 4; i++)
                        {
                            if ((col_i + i) < col_end && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0.0;
                        }
                    }
        }
    }
    else
    { // for general cases
        for (col_i = (col_start & -4); col_i < (col_end & -4); col_i += 4)
        {
            int cnt_y[4] = {col_i / output_x, (col_i + 1) / output_x, (col_i + 2) / output_x, (col_i + 3) / output_x};
            int cnt_x[4] = {col_i - cnt_y[0] * output_x, col_i - cnt_y[1] * output_x + 1,
                            col_i - cnt_y[2] * output_x + 2, col_i - cnt_y[3] * output_x + 3};
            int imx_start[4] = {cnt_x[0] * stride_x - pad_x, cnt_x[1] * stride_x - pad_x, cnt_x[2] * stride_x - pad_x,
                                cnt_x[3] * stride_x - pad_x};
            int imy_start[4] = {cnt_y[0] * stride_y - pad_y, cnt_y[1] * stride_y - pad_y, cnt_y[2] * stride_y - pad_y,
                                cnt_y[3] * stride_y - pad_y};
            for (kch = 0; kch < input_chan; kch++)
                for (ky = 0; ky < (kernel_y * dilation_y); ky += dilation_y)
                    for (kx = 0; kx < (kernel_x * dilation_x); kx += dilation_x)
                    {
                        int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                        int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                        for (i = 0; i < 4; i++)
                        {
                            if (imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0.0;
                        }
                    }
        }
        // final 4 input
        if (col_end & 0x3)
        {
            int cnt_y[4] = {col_i / output_x, (col_i + 1) / output_x, (col_i + 2) / output_x, (col_i + 3) / output_x};
            int cnt_x[4] = {col_i - cnt_y[0] * output_x, col_i - cnt_y[1] * output_x + 1,
                            col_i - cnt_y[2] * output_x + 2, col_i - cnt_y[3] * output_x + 3};
            int imx_start[4] = {cnt_x[0] * stride_x - pad_x, cnt_x[1] * stride_x - pad_x, cnt_x[2] * stride_x - pad_x,
                                cnt_x[3] * stride_x - pad_x};
            int imy_start[4] = {cnt_y[0] * stride_y - pad_y, cnt_y[1] * stride_y - pad_y, cnt_y[2] * stride_y - pad_y,
                                cnt_y[3] * stride_y - pad_y};
            for (kch = 0; kch < input_chan; kch++)
                for (ky = 0; ky < (kernel_y * dilation_y); ky += dilation_y)
                    for (kx = 0; kx < (kernel_x * dilation_x); kx += dilation_x)
                    {
                        int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                        int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                        for (i = 0; i < 4; i++)
                        {
                            if ((col_i + i) < col_end && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0.0;
                        }
                    }
        }
    }
}

// interleave 0 ~ (output_chan & -16) kernels with 16 in form of k[0-15][0],k[0-15][1],k[0-15][2]..
// interleave (output_chan & -16) ~ ((output_chan + 3) & -4) tail kernls with 4 in form of
// k[0-3][0],k[0-3][1],k[0-3][2]..
void interleave_kernel(__fp16* kernel, __fp16* kernel_interleaved, int kernel_chan, int kernel_size)
{
    int i, j;
    __fp16 *cur_kernel0, *cur_kernel1, *cur_kernel2, *cur_kernel3, *cur_kernel4, *cur_kernel5, *cur_kernel6,
        *cur_kernel7;
    __fp16 *cur_kernel8, *cur_kernel9, *cur_kernel10, *cur_kernel11, *cur_kernel12, *cur_kernel13, *cur_kernel14,
        *cur_kernel15;
    __fp16* cur_kernel_interleaved = kernel_interleaved;

    // interleave 16 kernels
    for (i = 0; i < (kernel_chan & -16); i += 16)
    {
        cur_kernel0 = kernel + kernel_size * i;
        cur_kernel1 = kernel + kernel_size * (i + 1);
        cur_kernel2 = kernel + kernel_size * (i + 2);
        cur_kernel3 = kernel + kernel_size * (i + 3);
        cur_kernel4 = kernel + kernel_size * (i + 4);
        cur_kernel5 = kernel + kernel_size * (i + 5);
        cur_kernel6 = kernel + kernel_size * (i + 6);
        cur_kernel7 = kernel + kernel_size * (i + 7);
        cur_kernel8 = kernel + kernel_size * (i + 8);
        cur_kernel9 = kernel + kernel_size * (i + 9);
        cur_kernel10 = kernel + kernel_size * (i + 10);
        cur_kernel11 = kernel + kernel_size * (i + 11);
        cur_kernel12 = kernel + kernel_size * (i + 12);
        cur_kernel13 = kernel + kernel_size * (i + 13);
        cur_kernel14 = kernel + kernel_size * (i + 14);
        cur_kernel15 = kernel + kernel_size * (i + 15);
        for (j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel0[j];
            *(cur_kernel_interleaved++) = cur_kernel1[j];
            *(cur_kernel_interleaved++) = cur_kernel2[j];
            *(cur_kernel_interleaved++) = cur_kernel3[j];
            *(cur_kernel_interleaved++) = cur_kernel4[j];
            *(cur_kernel_interleaved++) = cur_kernel5[j];
            *(cur_kernel_interleaved++) = cur_kernel6[j];
            *(cur_kernel_interleaved++) = cur_kernel7[j];
            *(cur_kernel_interleaved++) = cur_kernel8[j];
            *(cur_kernel_interleaved++) = cur_kernel9[j];
            *(cur_kernel_interleaved++) = cur_kernel10[j];
            *(cur_kernel_interleaved++) = cur_kernel11[j];
            *(cur_kernel_interleaved++) = cur_kernel12[j];
            *(cur_kernel_interleaved++) = cur_kernel13[j];
            *(cur_kernel_interleaved++) = cur_kernel14[j];
            *(cur_kernel_interleaved++) = cur_kernel15[j];
        }
    }

    for (i = (kernel_chan & -16); i < (kernel_chan & -4); i += 4)
    {
        cur_kernel0 = kernel + kernel_size * i;
        cur_kernel1 = kernel + kernel_size * (i + 1);
        cur_kernel2 = kernel + kernel_size * (i + 2);
        cur_kernel3 = kernel + kernel_size * (i + 3);
        for (j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel0[j];
            *(cur_kernel_interleaved++) = cur_kernel1[j];
            *(cur_kernel_interleaved++) = cur_kernel2[j];
            *(cur_kernel_interleaved++) = cur_kernel3[j];
        }
    }
    // last 4 kernel
    cur_kernel0 = kernel + kernel_size * i;
    cur_kernel1 = kernel + kernel_size * (i + 1);
    cur_kernel2 = kernel + kernel_size * (i + 2);
    if ((kernel_chan & 0x3) == 3)
    {
        for (j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel0[j];
            *(cur_kernel_interleaved++) = cur_kernel1[j];
            *(cur_kernel_interleaved++) = cur_kernel2[j];
            *(cur_kernel_interleaved++) = 0.0;
        }
    }
    else if ((kernel_chan & 0x3) == 2)
    {
        for (j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel0[j];
            *(cur_kernel_interleaved++) = cur_kernel1[j];
            *(cur_kernel_interleaved++) = 0.0;
            *(cur_kernel_interleaved++) = 0.0;
        }
    }
    else if ((kernel_chan & 0x3) == 1)
    {
        for (j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel0[j];
            *(cur_kernel_interleaved++) = 0.0;
            *(cur_kernel_interleaved++) = 0.0;
            *(cur_kernel_interleaved++) = 0.0;
        }
    }
}

static void interleave(struct tensor* filter, struct conv_priv_info* priv_info, struct conv_param* param)
{
    int group = param->group;
    int out_chan = filter->dims[0] / group;
    int kernel_size = filter->dims[1] * filter->dims[2] * filter->dims[3];

    int kernel_size_g = kernel_size * out_chan;
    int kernel_interleaved_size_g = kernel_size * ((out_chan + 3) & -4);

    __fp16* kernel = (__fp16*)filter->data;

    __fp16* interleave_buf = (__fp16*)priv_info->interleave_buffer;
    for (int g = 0; g < group; g++)
    {
        __fp16* cur_kernel = kernel + g * kernel_size_g;
        __fp16* cur_interleave = interleave_buf + g * kernel_interleaved_size_g;
        interleave_kernel(cur_kernel, cur_interleave, out_chan, kernel_size);
    }
}

static void hgemm_set(__fp16* col, __fp16* kernel, __fp16* biases, __fp16* output, int kernel_size,
                      int ch_start, int ch_end, int output_xy, int relu_fused, int num_thread, int cpu_affinity)
{
    int nn_outch = ch_end / PER_OUT_CHAN;
    int col_end3 = output_xy & 0x3;

    if (col_end3)
    {
#pragma omp parallel for num_threads(num_thread)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * PER_OUT_CHAN;

            __fp16* biasptr = biases ? (__fp16*)(biases + p) : NULL;
            __fp16* kernel_tmp = (__fp16*)(kernel + p * kernel_size);
            __fp16* output_tmp = (__fp16*)(output + p * output_xy);

            int col_line = 0;
            for (col_line = 0; col_line + 3 < output_xy; col_line += 4)
            {
                __fp16* col_tmp = (__fp16*)(col + col_line * kernel_size);
                hgemm_4x16_a76(biasptr, col_tmp, kernel_tmp, kernel_size, output_tmp + col_line, output_xy, relu_fused);
            }
            {
                __fp16 result[64];
                __fp16* col_tmp = (__fp16*)(col + col_line * kernel_size);
                hgemm_4x16_a76(biasptr, col_tmp, kernel_tmp, kernel_size, result, 4, relu_fused);

                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < (col_end3); j++)
                        *(output + (p + i) * output_xy + col_line + j) = result[(i << 2) + j];
                }
            }
        }
    }
    else
    {
#pragma omp parallel for num_threads(num_thread)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * PER_OUT_CHAN;

            __fp16* biasptr = biases ? (__fp16*)(biases + p) : NULL;
            __fp16* kernel_tmp = (__fp16*)(kernel + p * kernel_size);
            __fp16* output_tmp = (__fp16*)(output + p * output_xy);

            for (int col_line = 0; col_line + 3 < output_xy; col_line += 4)
            {
                __fp16* col_tmp = (__fp16*)(col + col_line * kernel_size);
                hgemm_4x16_a76(biasptr, col_tmp, kernel_tmp, kernel_size, output_tmp + col_line, output_xy, relu_fused);
            }
        }
    }
}

static void hgemm4x4(__fp16* col, __fp16* kernel, __fp16* biases, __fp16* output, int kernel_size,
                     int ch_start, int ch_end, int output_xy, int relu_fused, int num_thread, int cpu_affinity)
{
    __fp16 result[16];
    __fp16* cur_biases = NULL;
    int col_line, kernel_num;
    __fp16 *cur_col, *cur_kernel, *cur_output;
    int i, j;
    int col_end3 = output_xy & 0x3;
    int kernel_end3 = ch_end & 0x3;

    for (kernel_num = ch_start; kernel_num < (ch_end & -4); kernel_num += 4)
    {
        if (biases)
            cur_biases = biases + kernel_num;
        cur_kernel = kernel + kernel_num * kernel_size;
        cur_output = output + kernel_num * output_xy;
        for (col_line = 0; col_line < (output_xy & -4); col_line += 4)
        {
            cur_col = col + col_line * kernel_size;
            hgemm_4x4_a76(cur_biases, cur_col, cur_kernel, kernel_size, cur_output + col_line, output_xy, relu_fused);
        }
        if (col_end3)
        {
            cur_col = col + col_line * kernel_size;
            hgemm_4x4_a76(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, relu_fused);

            for (i = 0; i < 4; i++)
            {
                for (j = 0; j < (col_end3); j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
            }
        }
    }
    if (kernel_end3)
    {
        if (biases)
            cur_biases = biases + kernel_num;
        cur_kernel = kernel + kernel_num * kernel_size;
        for (col_line = 0; col_line < (output_xy & -4); col_line += 4)
        {
            cur_col = col + col_line * kernel_size;
            hgemm_4x4_a76(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, relu_fused);

            for (i = 0; i < kernel_end3; i++)
                for (j = 0; j < 4; j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
        }
        if (col_end3)
        {
            cur_col = col + col_line * kernel_size;
            hgemm_4x4_a76(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, relu_fused);

            for (i = 0; i < (kernel_end3); i++)
            {
                for (j = 0; j < (col_end3); j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
            }
        }
    }
}

int fp16_conv_hcl_get_shared_mem_size(struct tensor* input,
                                      struct tensor* output,
                                      struct conv_param* param)
{
    int group = param->group;
    int input_chan = param->input_channel / group;
    int kernel_size = input_chan * param->kernel_h * param->kernel_w;

    int output_xy = output->dims[2] * output->dims[3];
    int mem_size = sizeof(__fp16) * kernel_size * ((output_xy + 3) & -4) + 128;

    return mem_size;
}

static int get_private_mem_size(struct tensor* filter, struct conv_param* param)
{
    int group = param->group;
    int out_chan = filter->dims[0] / group;
    int kernel_size = filter->dims[1] * filter->dims[2] * filter->dims[3];

    int mem_size = sizeof(__fp16) * kernel_size * ((out_chan + 3) & -4) * group + 128;

    return mem_size;
}

int fp16_conv_hcl_prerun(struct tensor* input_tensor,
                         struct tensor* filter_tensor,
                         struct tensor* output_tensor,
                         struct conv_priv_info* priv_info,
                         struct conv_param* param)
{
    if (!priv_info->external_im2col_mem)
    {
        int mem_size = fp16_conv_hcl_get_shared_mem_size(input_tensor, output_tensor, param);
        void* mem = sys_malloc(mem_size);
        priv_info->im2col_buffer = mem;
        priv_info->im2col_buffer_size = mem_size;
    }

    if (!priv_info->external_interleave_mem)
    {
        int mem_size = get_private_mem_size(filter_tensor, param);
        void* mem = sys_malloc(mem_size);
        priv_info->interleave_buffer = mem;
        priv_info->interleave_buffer_size = mem_size;
    }

    interleave(filter_tensor, priv_info, param);

    return 0;
}

int fp16_conv_hcl_postrun(struct conv_priv_info* priv_info)
{
    if (!priv_info->external_interleave_mem && priv_info->interleave_buffer != NULL)
    {
        sys_free(priv_info->interleave_buffer);
        priv_info->interleave_buffer = NULL;
    }

    if (!priv_info->external_im2col_mem && priv_info->im2col_buffer != NULL)
    {
        sys_free(priv_info->im2col_buffer);
        priv_info->im2col_buffer = NULL;
    }

    return 0;
}

int fp16_conv_hcl_run(struct tensor* input_tensor,
                      struct tensor* filter_tensor,
                      struct tensor* bias_tensor,
                      struct tensor* output_tensor,
                      struct conv_priv_info* priv_info,
                      struct conv_param* param,
                      int num_thread, int cpu_affinity)
{
    /* param */
    // TLOG_ERR("run into fp16_conv_hcl_run!\n");
    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    int pad_h0 = param->pad_h0;
    int pad_h1 = param->pad_h1;
    int pad_w0 = param->pad_w0;
    int pad_w1 = param->pad_w1;
    long fused_relu = param->activation;

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
    __fp16* input_buf = (__fp16*)input_tensor->data;
    __fp16* output_buf = (__fp16*)output_tensor->data;
    __fp16* col_buf = (__fp16*)priv_info->im2col_buffer;
    __fp16* interleave_buf = (__fp16*)priv_info->interleave_buffer;
    __fp16* biases_buf = NULL;
    if (bias_tensor)
        biases_buf = (__fp16*)bias_tensor->data;

    int sgemm_set_chan = out_c / PER_OUT_CHAN * PER_OUT_CHAN;
    int sgemm_set_remain = out_c % PER_OUT_CHAN;
    for (int n = 0; n < batch; n++) // batch size
    {
        for (int g = 0; g < group; g++)
        {
            /* im2col */
            __fp16* cur_input = input_buf + (n * group + g) * input_size;

            im2col(cur_input, col_buf, in_c, in_w, in_h, kernel_w, kernel_h,
                   stride_w, stride_h, dilation_w, dilation_h, pad_w0, pad_w1, pad_h0, pad_h1,
                   out_w, out_h, 0, out_hw);

            /* gemm */
            __fp16* cur_kernel = interleave_buf + g * (kernel_size * ((out_c + 3) & -4));
            __fp16* cur_output = output_buf + (n * group + g) * output_size;
            __fp16* cur_bias = biases_buf ? (biases_buf + g * out_c) : NULL;
            hgemm_set(col_buf, cur_kernel, cur_bias, cur_output, kernel_size, 0, sgemm_set_chan, out_hw, fused_relu, num_thread, cpu_affinity);
            if (sgemm_set_remain)
            {
                hgemm4x4(col_buf, cur_kernel, cur_bias, cur_output, kernel_size, sgemm_set_chan, out_c, out_hw, fused_relu, num_thread, cpu_affinity);
            }
        }
    }

    return 0;
}
