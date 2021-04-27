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
 * Author: haoluo@openailab.com
 */

#include "conv_kernel_arm.h"

#include "api/c_api.h"
#include "utility/sys_port.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "wino_conv_kernel_arm.h"
#ifdef __aarch64__
#include "wino_conv_kernel_1_arm.h"
#endif

#ifdef __aarch64__
#define PER_OUT_CHAN 16
void sgemm_4x16_a72(float* biases, float* input, float* kernel, long kernel_size, float* output, long output_xy,
                    int activation, int layout);
void sgemm_4x4_a72(float* biases, float* input, float* kernel, long kernel_size, float* output, long output_xy,
                   int activation, int layout);
#else
#define PER_OUT_CHAN 12
void sgemm_4x12_a17(float* biases, float* input, float* kernel, int kernel_size, float* output, int output_xy,
                    int activation, int layout);
void sgemm_4x4_a17(float* biases, float* input, float* kernel, int kernel_size, float* output, int output_xy,
                   int activation, int layout);
#endif

void im2col_fp32_1x1(float* input, int input_xy, float* col, int col_cnt, int input_chan);
void im2col_fp32_3x3(float* input, int w, int h, int channel, float* cur_col, int stride);

static void interleave_kernel(float* kernel, float* kernel_interleaved, int kernel_chan, int kernel_size)
{
    int i, j, k;
    float* cur_kernel[PER_OUT_CHAN];
    float* cur_kernel_interleaved = kernel_interleaved;

    // interleave PER_OUT_CHAN kernels
    for (i = 0; i + PER_OUT_CHAN - 1 < kernel_chan; i += PER_OUT_CHAN)
    {
        for (k = 0; k < PER_OUT_CHAN; k++)
            cur_kernel[k] = kernel + kernel_size * (i + k);
        for (j = 0; j < kernel_size; j++)
        {
            for (k = 0; k < PER_OUT_CHAN; k++)
                *(cur_kernel_interleaved++) = cur_kernel[k][j];
        }
    }
    for (; i < (kernel_chan & -4); i += 4)
    {
        for (k = 0; k < 4; k++)
            cur_kernel[k] = kernel + kernel_size * (i + k);
        for (j = 0; j < kernel_size; j++)
        {
            for (k = 0; k < 4; k++)
                *(cur_kernel_interleaved++) = cur_kernel[k][j];
        }
    }
    // last 4 kernel
    for (k = 0; k < 3; k++)
        cur_kernel[k] = kernel + kernel_size * (i + k);
    if ((kernel_chan & 0x3) == 3)
    {
        for (j = 0; j < kernel_size; j++)
        {
            for (k = 0; k < 3; k++)
                *(cur_kernel_interleaved++) = cur_kernel[k][j];
            *(cur_kernel_interleaved++) = 0.f;
        }
    }
    else if ((kernel_chan & 0x3) == 2)
    {
        for (j = 0; j < kernel_size; j++)
        {
            for (k = 0; k < 2; k++)
                *(cur_kernel_interleaved++) = cur_kernel[k][j];
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
        }
    }
    else if ((kernel_chan & 0x3) == 1)
    {
        for (j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel[0][j];
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
        }
    }
}

/* kernel interleave */
static void interleave(struct tensor* filter, struct conv_priv_info* priv_info, struct conv_param* param)
{
    int group       = param->group;
    int kernel_size = filter->dims[1] * filter->dims[2] * filter->dims[3];
    int out_chan    = filter->dims[0] / group;
    int out_chan_align4 = (out_chan + 3) / 4 * 4;

    int kernel_size_algin = kernel_size * out_chan_align4;
    int kernel_size_group = kernel_size * out_chan;

    float* kernel = filter->data;
    float* interleave_buf = priv_info->interleave_buffer;
    for (int g = 0; g < group; g++)
    {
        float* cur_kernel     = kernel + g * kernel_size_group;
        float* cur_interleave = interleave_buf + g * kernel_size_algin;
        interleave_kernel(cur_kernel, cur_interleave, out_chan, kernel_size);
    }
}

static void im2col(float* input, float* col, int in_c, int in_w, int in_h, int k_w, int k_h, int s_w, int s_h, int d_w,
                   int d_h, int pad_w0, int pad_w1, int pad_h0, int pad_h1, int out_w, int out_h, int num_thread)
{
    if (k_w == 1 && k_h == 1 && s_w == 1 && s_h == 1)
    {
        int kernel_size = k_w * k_h * in_c;
        int in_xy = in_w * in_h;
        int out_xy = out_w * out_h;
        int col_end3 = out_xy & 3;
        #pragma omp parallel for num_threads(num_thread)
        for (int col_i = 0; col_i < out_xy - 3; col_i += 4)
        {
            float* cur_col = col + col_i * kernel_size;

            float* cur_input = input + col_i;
            im2col_fp32_1x1(cur_input, in_xy, cur_col, 4, in_c);
        }
        int col_i = out_xy & -4;
        float* cur_col;
        // final 4 input
        if (col_end3)
        {
            cur_col = col + col_i * kernel_size;
            for (int col_j = 0; col_j < kernel_size; col_j++)
            {
                for (int i = 0; i < 4; i++)
                {
                    if (i < col_end3)
                        *cur_col++ = *(input + col_j * in_xy + col_i + i);
                    else
                        *cur_col++ = 0;
                }
            }
        }
    }
#ifdef __aarch64__
    else if (d_w == 1 && d_h == 1 && k_w == 3 && k_h == 3 && s_w == s_h)
    {
        int kernel_size = k_w * k_h * in_c;
        int in_xy = in_w * in_h;
        int out_xy = out_w * out_h;
        int col_end3 = out_xy & 3;
        int is_pad0 = (pad_w0 == 0) && (pad_h0 == 0) && (pad_w1 == 0) && (pad_h1 == 0);
        #pragma omp parallel for num_threads(num_thread)
        for (int col_i = 0; col_i < (out_xy & -4); col_i += 4)
        {
            float* cur_col = col + col_i * kernel_size;
            int imy0 = col_i / out_w;
            int imy3 = (col_i + 3) / out_w;
            int imx0 = col_i - imy0 * out_w;
            int imx3 = (col_i + 3) - imy3 * out_w;
            if ((imy0 == imy3) && (is_pad0 || (imy0 != 0 && imx0 != 0 && imy0 != (out_h - 1) && imx3 != (out_w - 1))))
            {
                float* l0 = input + (imy0 * s_h - pad_h0) * in_w + (imx0 * s_w - pad_w0);
                {
                    im2col_fp32_3x3(l0, in_w, in_h, in_c, cur_col, s_w);
                    cur_col += 4 * kernel_size;
                }
            }
            else
            {
                int cnt_y[4] = {imy0, (col_i + 1) / out_w, (col_i + 2) / out_w, imy3};
                int cnt_x[4] = {imx0, col_i - cnt_y[1] * out_w + 1, col_i - cnt_y[2] * out_w + 2, imx3};
                int imx_start[4] = {cnt_x[0] * s_w - pad_w0, cnt_x[1] * s_w - pad_w0, cnt_x[2] * s_w - pad_w0,
                                    cnt_x[3] * s_w - pad_w0};
                int imy_start[4] = {cnt_y[0] * s_h - pad_h0, cnt_y[1] * s_h - pad_h0, cnt_y[2] * s_h - pad_h0,
                                    cnt_y[3] * s_h - pad_h0};
                for (int kch = 0; kch < in_c; kch++)
                    for (int ky = 0; ky < 3; ky++)
                        for (int kx = 0; kx < 3; kx++)
                        {
                            int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                            int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                            for (int i = 0; i < 4; i++)
                            {
                                if (imx[i] >= 0 && imx[i] < in_w && imy[i] >= 0 && imy[i] < in_h)
                                    *cur_col++ = *(input + in_xy * kch + in_w * imy[i] + imx[i]);
                                else
                                    *cur_col++ = 0.f;
                            }
                        }
            }
        }
        // final 4 input
        int col_i = out_xy & -4;
        if (col_end3)
        {
            float* cur_col = col + col_i * kernel_size;
            int cnt_y[4] = {col_i / out_w, (col_i + 1) / out_w, (col_i + 2) / out_w, (col_i + 3) / out_w};
            int cnt_x[4] = {col_i - cnt_y[0] * out_w, col_i - cnt_y[1] * out_w + 1, col_i - cnt_y[2] * out_w + 2,
                            col_i - cnt_y[3] * out_w + 3};
            int imx_start[4] = {cnt_x[0] * s_w - pad_w0, cnt_x[1] * s_w - pad_w0, cnt_x[2] * s_w - pad_w0,
                                cnt_x[3] * s_w - pad_w0};
            int imy_start[4] = {cnt_y[0] * s_h - pad_h0, cnt_y[1] * s_h - pad_h0, cnt_y[2] * s_h - pad_h0,
                                cnt_y[3] * s_h - pad_h0};
            for (int kch = 0; kch < in_c; kch++)
            {
                for (int ky = 0; ky < 3; ky++)
                {
                    for (int kx = 0; kx < 3; kx++)
                    {
                        int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                        int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                        for (int i = 0; i < 4; i++)
                        {
                            if (i < col_end3 && imx[i] >= 0 && imx[i] < in_w && imy[i] >= 0 && imy[i] < in_h)
                                *cur_col++ = *(input + in_xy * kch + in_w * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0.f;
                        }
                    }
                }
            }
        }
    }
#endif
    else
    {
        int out_xy = out_w * out_h;
        #pragma omp parallel for num_threads(num_thread)
        for (int col_i = 0; col_i < out_xy - 3; col_i += 4)
        {
            int kernel_size = k_w * k_h * in_c;
            int in_xy = in_w * in_h;
            int col_end3 = out_xy & 3;
            float* cur_col = col + col_i * kernel_size;
            int cnt_y[4] = {col_i / out_w, (col_i + 1) / out_w, (col_i + 2) / out_w, (col_i + 3) / out_w};
            int cnt_x[4] = {col_i - cnt_y[0] * out_w, col_i - cnt_y[1] * out_w + 1, col_i - cnt_y[2] * out_w + 2,
                            col_i - cnt_y[3] * out_w + 3};
            int imx_start[4] = {cnt_x[0] * s_w - pad_w0, cnt_x[1] * s_w - pad_w0, cnt_x[2] * s_w - pad_w0,
                                cnt_x[3] * s_w - pad_w0};
            int imy_start[4] = {cnt_y[0] * s_h - pad_h0, cnt_y[1] * s_h - pad_h0, cnt_y[2] * s_h - pad_h0,
                                cnt_y[3] * s_h - pad_h0};
            for (int kch = 0; kch < in_c; kch++)
                for (int ky = 0; ky < (k_h * d_h); ky += d_h)
                    for (int kx = 0; kx < (k_w * d_w); kx += d_w)
                    {
                        int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                        int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                        for (int i = 0; i < 4; i++)
                        {
                            if (imx[i] >= 0 && imx[i] < in_w && imy[i] >= 0 && imy[i] < in_h)
                                *cur_col++ = *(input + in_xy * kch + in_w * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0.f;
                        }
                    }
        }
        int col_i = out_xy & -4;
        float* cur_col;
        int kernel_size = k_w * k_h * in_c;
        int in_xy = in_w * in_h;
        int col_end3 = out_xy & 3;
        if (col_end3)
        {
            cur_col = col + col_i * kernel_size;
            int cnt_y[4] = {col_i / out_w, (col_i + 1) / out_w, (col_i + 2) / out_w, (col_i + 3) / out_w};
            int cnt_x[4] = {col_i - cnt_y[0] * out_w, col_i - cnt_y[1] * out_w + 1, col_i - cnt_y[2] * out_w + 2,
                            col_i - cnt_y[3] * out_w + 3};
            int imx_start[4] = {cnt_x[0] * s_w - pad_w0, cnt_x[1] * s_w - pad_w0, cnt_x[2] * s_w - pad_w0,
                                cnt_x[3] * s_w - pad_w0};
            int imy_start[4] = {cnt_y[0] * s_h - pad_h0, cnt_y[1] * s_h - pad_h0, cnt_y[2] * s_h - pad_h0,
                                cnt_y[3] * s_h - pad_h0};
            for (int kch = 0; kch < in_c; kch++)
                for (int ky = 0; ky < (k_h * d_h); ky += d_h)
                    for (int kx = 0; kx < (k_w * d_w); kx += d_w)
                    {
                        int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                        int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                        for (int i = 0; i < 4; i++)
                        {
                            if (i < col_end3 && imx[i] >= 0 && imx[i] < in_w && imy[i] >= 0 && imy[i] < in_h)
                                *cur_col++ = *(input + in_xy * kch + in_w * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0.f;
                        }
                    }
        }
    }
}

static void sgemm_set(float* col, float* kernel, float* biases, float* output, int kernel_size, int col_start,
                        int col_end, int kernel_start, int kernel_end, int output_xy, int activation, int num_thread, int cpu_affinity)
{
    int col_end3 = col_end & 0x3;
    int nn_outch = kernel_end / PER_OUT_CHAN;

#pragma omp parallel for num_threads(num_thread)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * PER_OUT_CHAN;
        float* biasptr = biases ? ( float* )(biases + p) : NULL;
        float* kernel_tmp = ( float* )(kernel + p * kernel_size);
        float* output_tmp = ( float* )(output + p * output_xy);

        for (int col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
#ifdef __aarch64__
        {
            float* col_tmp = ( float* )(col + col_line * kernel_size);
            sgemm_4x16_a72(biasptr, col_tmp, kernel_tmp, kernel_size, output_tmp + col_line, output_xy, activation, 0);
        }
        if (col_end3)
        {
            int col_line = col_end & -4;
            float result[4 * PER_OUT_CHAN];
            float* col_tmp = ( float* )(col + col_line * kernel_size);

            sgemm_4x16_a72(biasptr, col_tmp, kernel_tmp, kernel_size, result, 4, activation, 0);

            for (int i = 0; i < 16; i++)
            {
                for (int j = 0; j < (col_end3); j++)
                {
                    *(output + (p + i) * output_xy + col_line + j) = result[(i << 2) + j];
                }
            }
        }
#else
        {
            float* col_tmp = ( float* )(col + col_line * kernel_size);
            sgemm_4x12_a17(biasptr, col_tmp, kernel_tmp, kernel_size, output_tmp + col_line, output_xy, activation, 0);
        }
        if (col_end3)
        {
            int col_line = col_end & -4;
            float result[4 * PER_OUT_CHAN];
            float* col_tmp = ( float* )(col + col_line * kernel_size);

            sgemm_4x12_a17(biasptr, col_tmp, kernel_tmp, kernel_size, result, 4, activation, 0);

            for (int i = 0; i < PER_OUT_CHAN; i++)
            {
                for (int j = 0; j < (col_end3); j++)
                {
                    *(output + (p + i) * output_xy + col_line + j) = result[(i << 2) + j];
                }
            }
        }
#endif
    }
}

static void sgemm4x4(float* col, float* kernel, float* biases, float* output, int kernel_size, int col_start, int col_end,
                     int kernel_start, int kernel_end, int output_xy, int activation, int num_thread, int cpu_affinity)
{
    int col_end3 = col_end & 0x3;
    int kernel_end3 = kernel_end & 0x3;

#pragma omp parallel for num_threads(num_thread)
    for (int kernel_num = (kernel_start & -4); kernel_num  < (kernel_end & -4); kernel_num += 4)
    {
        float *cur_col, *cur_kernel, *cur_output;
        float* cur_biases = biases ? ( float* )(biases + kernel_num) : NULL;

        cur_kernel = ( float* )(kernel + kernel_num * kernel_size);
        cur_output = ( float* )(output + kernel_num * output_xy);
        for (int col_line = 0; col_line < (col_end & -4); col_line += 4)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
#ifdef __aarch64__
            sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, cur_output + col_line, output_xy, activation, 0);
#else
            sgemm_4x4_a17(cur_biases, cur_col, cur_kernel, kernel_size, cur_output + col_line, output_xy, activation, 0);
#endif
        }
        if (col_end3)
        {
            float result[16];
            int col_line = col_end & -4;
            cur_col = ( float* )(col + col_line * kernel_size);
#ifdef __aarch64__
            sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
#else
            sgemm_4x4_a17(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
#endif
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < (col_end3); j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
            }
        }
    }
    if (kernel_end3)
    {
        int kernel_num = (kernel_end & -4);
        float* cur_biases = biases ? ( float* )(biases + kernel_num) : NULL;
        float* cur_kernel = ( float* )(kernel + kernel_num * kernel_size);

#pragma omp parallel for num_threads(num_thread)
        for (int col_line = 0; col_line < (col_end & -4); col_line += 4)
        {
            float result[16];
            float* cur_col = ( float* )(col + col_line * kernel_size);
#ifdef __aarch64__
            sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
#else
            sgemm_4x4_a17(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
#endif
            for (int i = 0; i < kernel_end3; i++)
                for (int j = 0; j < 4; j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
        }

        if (col_end3)
        {
            float result[16];
            int col_line = col_end & -4;
            float* cur_col = ( float* )(col + col_line * kernel_size);
#ifdef __aarch64__
            sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
#else
            sgemm_4x4_a17(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
#endif
            for (int i = 0; i < (kernel_end3); i++)
            {
                for (int j = 0; j < (col_end3); j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
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
    int output_chan = param->output_channel;
    int group = param->group;

    if (in_h < 7 && in_w < 7)
        return 0;
    if (in_h < 10 && in_w < 10 && output_chan < 16)
        return 0;
    if (group != 1 || kernel_h != 3 || kernel_w != 3)
        return 0;
    if (dilation_h != 1 || dilation_w != 1 || stride_h != 1 || stride_w != 1)
        return 0;

    return 1;
}

/*
 * get the memory size for im2col of input tensor
 */
int conv_hcl_get_shared_mem_size(struct tensor* input, struct tensor* output, struct conv_param* param)
{
    int in_h  = input->dims[2];
    int in_w  = input->dims[3];
    int out_h = output->dims[2];
    int out_w = output->dims[3];
    int group = param->group;
    int input_chan  = param->input_channel / group;
    int kernel_size = input_chan * param->kernel_h * param->kernel_w;
    int out_cstep   = out_h * out_w;      // channel cstep, output_h * output_w
    int elem_size   = input->elem_size;   // uint8/int8 is 1 byte, fp32 is 4 bytes

    out_cstep = (out_cstep + 3) / 4 * 4;
    int mem_size = elem_size * kernel_size * out_cstep + 128;

    return mem_size;
}

/*
 * get the memory size for im2col + sgemm of kernel tensor interleave
 */
static int get_private_mem_size(struct tensor* filter, struct conv_param* param)
{
    int group = param->group;
    int out_chan = filter->dims[0] / group;
    int out_chan_align4 = (out_chan + 3) / 4 * 4;
    int kernel_size = filter->dims[1] * filter->dims[2] * filter->dims[3];
    int mem_size = kernel_size * filter->elem_size * out_chan_align4 * group + 128;    // caution

    return mem_size;
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
    priv_info->external_im2col_pack4_mem = 0;
    priv_info->im2col_buffer_pack4 = NULL;
    priv_info->im2col_buffer_pack4_size = 0;

    return 0;
}

int conv_hcl_get_shared_pack4_mem_size(struct tensor* filter, struct tensor* output, struct conv_param* param)
{
    return 0;
}

int conv_hcl_prerun(struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* output_tensor,
                    struct conv_priv_info* priv_info, struct conv_param* param)
{
    int in_c = input_tensor->dims[1];
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];

    /* check winograd implement, only for conv3x3s1 */
    priv_info->winograd = winograd_support(param, in_h, in_w);
    if (priv_info->winograd)
    {
#ifdef __aarch64__
        if(in_c >= 256)
            return wino_conv_hcl_prerun_1(input_tensor, filter_tensor, output_tensor, priv_info, param);
        else
#endif
            return wino_conv_hcl_prerun(input_tensor, filter_tensor, output_tensor, priv_info, param);
    }

    /* alloc mem of im2col  */
    if (!priv_info->external_im2col_mem)
    {
        int mem_size = conv_hcl_get_shared_mem_size(input_tensor, output_tensor, param);
        void* mem = sys_malloc(mem_size);
        priv_info->im2col_buffer      = mem;
        priv_info->im2col_buffer_size = mem_size;
    }

    /* alloc mem of kernel interleave */
    if (!priv_info->external_interleave_mem)
    {
        int mem_size = get_private_mem_size(filter_tensor, param);
        void* mem = sys_malloc(mem_size);
        priv_info->interleave_buffer      = mem;
        priv_info->interleave_buffer_size = mem_size;
    }

    /* kernel interleave */
    interleave(filter_tensor, priv_info, param);

    return 0;
}

int conv_hcl_postrun(struct conv_priv_info* priv_info)
{
    if (priv_info->winograd)
    {
        wino_conv_hcl_postrun(priv_info);
    }

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

int conv_hcl_run(struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* bias_tensor,
                 struct tensor* output_tensor, struct conv_priv_info* priv_info, struct conv_param* param,
                 int num_thread, int cpu_affinity)
{
    /* param */
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
    int act_type = param->activation;

    int batch = input_tensor->dims[0];
    int in_c = input_tensor->dims[1] / group;
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];
    int input_size = in_c * in_h * in_w;
    int kernel_size = in_c * kernel_h * kernel_w;
    int input_image_size = input_tensor->dims[1] * input_tensor->dims[2] * input_tensor->dims[3];

    if (priv_info->winograd)
    {
#ifdef __aarch64__
        if(in_c >= 256)
            return wino_conv_hcl_run_1(input_tensor, filter_tensor, bias_tensor, output_tensor, priv_info, param, num_thread, cpu_affinity);
        else
#endif
            return wino_conv_hcl_run(input_tensor, filter_tensor, bias_tensor, output_tensor, priv_info, param, num_thread, cpu_affinity);
    }

    int out_c = output_tensor->dims[1] / group;
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int out_hw = out_h * out_w;
    int output_size = out_c * out_h * out_w;
    int out_c_align = ((out_c + 3) & -4);
    int output_image_size = output_tensor->dims[1] * output_tensor->dims[2] * output_tensor->dims[3];

    /* buffer addr */
    float* input_buf = ( float* )input_tensor->data;
    float* output_buf = ( float* )output_tensor->data;
    float* biases_buf = NULL;
    if (bias_tensor != NULL)
        biases_buf = ( float* )bias_tensor->data;
    float* col_buf = ( float* )priv_info->im2col_buffer;
    float* interleave_buf = ( float* )priv_info->interleave_buffer;

    /* block size split parameter */
    int L2_CACHE_SIZE = ((cpu_affinity == TENGINE_CLUSTER_LITTLE) ? 512 : 1024) * 1024;
    int kernel_size_l1 = kernel_size;
    int col_cnt_l2 = L2_CACHE_SIZE / 4 / kernel_size_l1 * 7 / 8;
    col_cnt_l2 = col_cnt_l2 > 4 ? (col_cnt_l2 & -4) : 4;
    int sgemm_set_chan = out_c / PER_OUT_CHAN * PER_OUT_CHAN;
    int sgemm_set_remain = out_c % PER_OUT_CHAN;

    for (int n = 0; n < batch; n++)    // batch size
    {
        for (int g = 0; g < group; g++)
        {
            float* cur_input = input_buf + n * input_image_size + g * input_size;
            float* cur_kernel = interleave_buf + g * kernel_size * out_c_align;
            float* cur_output = output_buf + n * output_image_size + g * output_size;
            float* cur_bias = biases_buf ? (biases_buf + g * out_c) : NULL;

            /* im2col */
            im2col(cur_input, col_buf, in_c, in_w, in_h, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h,
                pad_w0, pad_w1, pad_h0, pad_h1, out_w, out_h, num_thread);

            for(int col_i = 0; col_i < out_hw; col_i += col_cnt_l2)
            {
                int col_start = col_i;
                int col_end = col_i + col_cnt_l2;
                col_end = col_end > out_hw ? out_hw : col_end;
                /* gemm */
                sgemm_set(col_buf, cur_kernel, cur_bias, cur_output, kernel_size, col_start, col_end, 0, sgemm_set_chan, out_hw, act_type,
                        num_thread, cpu_affinity);
                if (sgemm_set_remain)
                    sgemm4x4(col_buf, cur_kernel, cur_bias, cur_output, kernel_size, col_start, col_end, sgemm_set_chan, out_c, out_hw,
                            act_type, num_thread, cpu_affinity);
            }
        }
    }

    return 0;
}
