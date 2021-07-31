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

#include "deconv_kernel_arm.h"

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#ifdef __aarch64__
#define PER_OUT_CHAN 16
void sgemm_4x16_deconv_a72(float* input, float* kernel, long kernel_size, float* output, long weight_size);
void sgemm_4x4_deconv_a72(float* input, float* kernel, long kernel_size, float* output, long weight_size);
void sgemm_4x16_deconv_a53(float* input, float* kernel, long kernel_size, float* output, long weight_size);
void sgemm_4x4_deconv_a53(float* input, float* kernel, long kernel_size, float* output, long weight_size);
#else
#define PER_OUT_CHAN 12
void sgemm_4x12_deconv_a17(float* input, float* kernel, int kernel_size, float* output, int weight_size);
void sgemm_4x4_deconv_a17(float* input, float* kernel, int kernel_size, float* output, int weight_size);
void sgemm_4x12_deconv_a7(float* input, float* kernel, int kernel_size, float* output, int weight_size);
void sgemm_4x4_deconv_a7(float* input, float* kernel, int kernel_size, float* output, int weight_size);
#endif

static double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void interleave_kernel(float* kernel, float* kernel_interleaved, int kernel_chan, int kernel_size)
{
    int i, j, k;
    float* cur_kernel_interleaved = kernel_interleaved;

    // interleave PER_OUT_CHAN kernels
    for (i = 0; i + PER_OUT_CHAN - 1 < kernel_size; i += PER_OUT_CHAN)
    {
        for (j = 0; j < kernel_chan; j++)
        {
            for (k = 0; k < PER_OUT_CHAN; k++)
                *(cur_kernel_interleaved++) = kernel[j * kernel_size + i + k];
        }
    }
    for (; i < (kernel_size & -4); i += 4)
    {
        for (j = 0; j < kernel_chan; j++)
        {
            for (k = 0; k < 4; k++)
                *(cur_kernel_interleaved++) = kernel[j * kernel_size + i + k];
        }
    }
    // last 4 kernel
    int kernel_size3 = kernel_chan & 0x3;
    if (kernel_size3)
    {
        for (j = 0; j < kernel_chan; j++)
        {
            for (k = 0; k < kernel_size3; k++)
                *(cur_kernel_interleaved++) = kernel[j * kernel_size + i + k];
            for (; k < 4; k++)
                *(cur_kernel_interleaved++) = 0.0;
        }
    }
}

static void interleave(struct tensor* filter, struct deconv_priv_info* priv_info, struct deconv_param* param)
{
    int group = param->group;
    int out_chan = filter->dims[0] / group;
    int kernel_size = out_chan * filter->dims[2] * filter->dims[3];
    int in_chan = filter->dims[1];

    int kernel_size_algin = in_chan * ((kernel_size + 3) & -4);

    float* kernel = filter->data;
    float* interleave_buf = priv_info->interleave_buffer;
    for (int g = 0; g < group; g++)
    {
        float* cur_kernel = kernel + g * kernel_size * in_chan;
        float* cur_interleave = interleave_buf + g * kernel_size_algin;
        interleave_kernel(cur_kernel, cur_interleave, in_chan, kernel_size);
    }
}

static void transpose_input(float* input, float* inputT, int input_w, int input_h)
{
    int i, j, k;
    int input_w3 = input_w & 0x3;

    float* cur_input = inputT;

    for (i = 0; i < (input_w & -4); i += 4)
        for (j = 0; j < input_h; j++)
            for (k = 0; k < 4; k++)
                *cur_input++ = *(input + j * input_w + i + k);

    if (input_w3)
    {
        for (j = 0; j < input_h; j++)
        {
            for (k = 0; k < input_w3; k++)
                *cur_input++ = *(input + j * input_w + i + k);
            for (; k < 4; k++)
                *cur_input++ = 0;
        }
    }
}

static void col2im(float* col, float* im, float* bias, int output_ch, int output_x, int output_y,
                   int kernel_x, int kernel_y, int stride_x, int stride_y, int dilation_x, int dilation_y, int pad_x,
                   int pad_y, int input_x, int input_y)
{
    float* cur_col;
    int imx_start, imy_start, ix, iy, kch, kx, ky, imx, imy;
    int output_xy = output_x * output_y;
    int kernel_xy = kernel_x * kernel_y;
    int weight_size = output_ch * kernel_x * kernel_y;
    int is_nodilation = (dilation_x == 1 && dilation_y == 1);
    int is_4x4 = (kernel_x == 4 && kernel_y == 4 && is_nodilation);
    int is_8x8 = (kernel_x == 8 && kernel_y == 8 && is_nodilation);
    /* init bias */
    if (bias == NULL)
    {
        for (int i = 0; i < (output_xy * output_ch); i++)
            im[i] = 0;
    }
    else
    {
        float* cur_im = im;
        for (int i = 0; i < output_ch; i++)
            for (int j = 0; j < output_xy; j++)
                *cur_im++ = bias[i];
    }

    if (is_4x4)
    {
        for (iy = 0; iy < input_y; iy++)
        {
            imy_start = iy * stride_y - pad_y;
            for (ix = 0; ix < input_x; ix++)
            {
                imx_start = ix * stride_x - pad_x;
                cur_col = col + (iy * input_x + ix) * weight_size;
                if (iy != 0 && iy != (input_y - 1) && ix != 0 && ix != (input_x - 1))
                {
                    for (kch = 0; kch < output_ch; kch++)
                        for (ky = 0; ky < 4; ky++)
                        {
                            imy = imy_start + ky;
                            for (kx = 0; kx < 4; kx++)
                                *(im + output_xy * kch + output_x * imy + imx_start + kx) += *cur_col++;
                        }
                }
                else
                {
                    for (kch = 0; kch < output_ch; kch++)
                    {
                        for (ky = 0; ky < 4; ky++)
                        {
                            imy = imy_start + ky;
                            for (kx = 0; kx < 4; kx++)
                            {
                                imx = imx_start + kx;
                                if (imx >= 0 && imx < output_x && imy >= 0 && imy < output_y)
                                    *(im + output_xy * kch + output_x * imy + imx) += *cur_col;
                                cur_col++;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (is_8x8)
    {
        for (iy = 0; iy < input_y; iy++)
        {
            imy_start = iy * stride_y - pad_y;
            for (ix = 0; ix < input_x; ix++)
            {
                imx_start = ix * stride_x - pad_x;
                cur_col = col + (iy * input_x + ix) * weight_size;
                if (iy != 0 && iy != (input_y - 1) && ix != 0 && ix != (input_x - 1))
                {
                    for (kch = 0; kch < output_ch; kch++)
                        for (ky = 0; ky < 8; ky++)
                        {
                            imy = imy_start + ky;
                            for (kx = 0; kx < 8; kx++)
                                *(im + output_xy * kch + output_x * imy + imx_start + kx) += *cur_col++;
                        }
                }
                else
                {
                    for (kch = 0; kch < output_ch; kch++)
                        for (ky = 0; ky < 8; ky++)
                        {
                            imy = imy_start + ky;
                            for (kx = 0; kx < 8; kx++)
                            {
                                imx = imx_start + kx;
                                if (imx >= 0 && imx < output_x && imy >= 0 && imy < output_y)
                                    *(im + output_xy * kch + output_x * imy + imx) += *cur_col;
                                cur_col++;
                            }
                        }
                }
            }
        }
    }
    // general case
    else
    {
        for (iy = 0; iy < input_y; iy++)
        {
            imy_start = iy * stride_y - pad_y;
            for (ix = 0; ix < input_x; ix++)
            {
                imx_start = ix * stride_x - pad_x;
                cur_col = col + (iy * input_x + ix) * weight_size;
                if (iy != 0 && iy != (input_y - 1) && ix != 0 && ix != (input_x - 1))
                {
                    for (kch = 0; kch < output_ch; kch++)
                        for (ky = 0; ky < kernel_y; ky++)
                        {
                            imy = imy_start + ky * dilation_y;
                            for (kx = 0; kx < kernel_x; kx++)
                            {
                                imx = imx_start + kx * dilation_x;
                                *(im + output_xy * kch + output_x * imy + imx) += *cur_col++;
                            }
                        }
                }
                else
                {
                    for (kch = 0; kch < output_ch; kch++)
                    {
                        for (ky = 0; ky < kernel_y; ky++)
                        {
                            imy = imy_start + ky * dilation_y;
                            for (kx = 0; kx < kernel_x; kx++)
                            {
                                imx = imx_start + kx * dilation_x;
                                float out = bias[kch];
                                if (imx >= 0 && imx < output_x && imy >= 0 && imy < output_y)
                                    *(im + output_xy * kch + output_x * imy + imx) += *cur_col;
                                cur_col++;
                            }
                        }
                    }
                }
            }
        }
    }
}

static void sgemm_set(float* input, float* kernel, float* col, int in_ch, int in_hw, int kernel_size,
                      int kernel_start, int kernel_end, int num_thread, int cpu_affinity)
{
    int nn_kernel = (kernel_end - kernel_start) / PER_OUT_CHAN;
    int input_end3 = in_hw & 0x3;

    if (input_end3)
    {
#pragma omp parallel for num_threads(num_thread)
        for (int pp = 0; pp < nn_kernel; pp++)
        {
            int p = kernel_start + pp * PER_OUT_CHAN;

            float* cur_kernel = (float*)(kernel + p * in_ch);

            int i = 0;
            for (i = 0; i + 3 < in_hw; i += 4)
#ifdef __aarch64__
            {
                float* cur_input = (float*)(input + i * in_ch);
                float* cur_col = (float*)(col + i * kernel_size + p);
                if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                    sgemm_4x16_deconv_a53(cur_input, cur_kernel, in_ch, cur_col, kernel_size);
                else
                    sgemm_4x16_deconv_a72(cur_input, cur_kernel, in_ch, cur_col, kernel_size);
            }
            {
                float result[64];
                float* cur_input = (float*)(input + i * in_ch);
                if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                    sgemm_4x16_deconv_a53(cur_input, cur_kernel, in_ch, result, 16);
                else
                    sgemm_4x16_deconv_a72(cur_input, cur_kernel, in_ch, result, 16);
                for (int j = 0; j < (input_end3); j++)
                {
                    for (int k = 0; k < 16; k++)
                        *(col + (i + j) * kernel_size + p + k) = result[(j << 4) + k];
                }
            }
#else
            {
                float* cur_input = (float*)(input + i * in_ch);
                float* cur_col = (float*)(col + i * kernel_size + p);
                if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                    sgemm_4x12_deconv_a7(cur_input, cur_kernel, in_ch, cur_col, kernel_size);
                else
                    sgemm_4x12_deconv_a17(cur_input, cur_kernel, in_ch, cur_col, kernel_size);
            }
            {
                float result[48];
                float* cur_input = (float*)(input + i * in_ch);
                if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                    sgemm_4x12_deconv_a7(cur_input, cur_kernel, in_ch, result, 12);
                else
                    sgemm_4x12_deconv_a17(cur_input, cur_kernel, in_ch, result, 12);
                for (int j = 0; j < (input_end3); j++)
                {
                    for (int k = 0; k < 12; k++)
                        *(col + (i + j) * kernel_size + p + k) = result[j * 12 + k];
                }
            }
#endif
        }
    }
    else
    {
#pragma omp parallel for num_threads(num_thread)
        for (int pp = 0; pp < nn_kernel; pp++)
        {
            int p = kernel_start + pp * PER_OUT_CHAN;

            float* cur_kernel = (float*)(kernel + p * in_ch);

            int i = 0;
            for (; i + 3 < in_hw; i += 4)
            {
                float* cur_input = (float*)(input + i * in_ch);
                float* cur_col = (float*)(col + i * kernel_size + p);
#ifdef __aarch64__
                if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                    sgemm_4x16_deconv_a53(cur_input, cur_kernel, in_ch, cur_col, kernel_size);
                else
                    sgemm_4x16_deconv_a72(cur_input, cur_kernel, in_ch, cur_col, kernel_size);
#else
                if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                    sgemm_4x12_deconv_a7(cur_input, cur_kernel, in_ch, cur_col, kernel_size);
                else
                    sgemm_4x12_deconv_a17(cur_input, cur_kernel, in_ch, cur_col, kernel_size);
#endif
            }
        }
    }
}

static void sgemm4x4(float* input, float* kernel, float* col, int in_ch, int in_hw, int kernel_size,
                     int kernel_start, int kernel_end, int num_thread, int cpu_affinity)
{
    float result[16];
    int input_line, kernel_num;
    float *cur_col, *cur_kernel, *cur_input;
    int i, j;
    int input_end3 = in_hw & 0x3;
    int kernel_end3 = kernel_end & 0x3;

    for (kernel_num = kernel_start; kernel_num + 3 < (kernel_end & -4); kernel_num += 4)
    {
        cur_kernel = (float*)(kernel + kernel_num * in_ch);
        for (input_line = 0; input_line < (in_hw & -4); input_line += 4)
        {
            cur_input = (float*)(input + input_line * in_ch);
            cur_col = (float*)(col + input_line * kernel_size + kernel_num);
#ifdef __aarch64__
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                sgemm_4x4_deconv_a53(cur_input, cur_kernel, in_ch, cur_col, kernel_size);
            else
                sgemm_4x4_deconv_a72(cur_input, cur_kernel, in_ch, cur_col, kernel_size);
#else
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                sgemm_4x4_deconv_a7(cur_input, cur_kernel, in_ch, cur_col, kernel_size);
            else
                sgemm_4x4_deconv_a17(cur_input, cur_kernel, in_ch, cur_col, kernel_size);
#endif
        }
        if (input_end3)
        {
            cur_input = (float*)(input + input_line * in_ch);
#ifdef __aarch64__
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                sgemm_4x4_deconv_a53(cur_input, cur_kernel, in_ch, result, 4);
            else
                sgemm_4x4_deconv_a72(cur_input, cur_kernel, in_ch, result, 4);
#else
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                sgemm_4x4_deconv_a7(cur_input, cur_kernel, in_ch, result, 4);
            else
                sgemm_4x4_deconv_a17(cur_input, cur_kernel, in_ch, result, 4);
#endif
            for (j = 0; j < (input_end3); j++)
                for (i = 0; i < 4; i++)
                    *(col + (input_line + j) * kernel_size + kernel_num + i) = result[(j << 2) + i];
        }
    }
    if (kernel_end3)
    {
        cur_kernel = (float*)(kernel + kernel_num * in_ch);
        for (input_line = 0; input_line < (in_hw & -4); input_line += 4)
        {
            cur_input = (float*)(input + input_line * in_ch);
#ifdef __aarch64__
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                sgemm_4x4_deconv_a53(cur_input, cur_kernel, in_ch, result, 4);
            else
                sgemm_4x4_deconv_a72(cur_input, cur_kernel, in_ch, result, 4);
#else
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                sgemm_4x4_deconv_a7(cur_input, cur_kernel, in_ch, result, 4);
            else
                sgemm_4x4_deconv_a17(cur_input, cur_kernel, in_ch, result, 4);
#endif
            for (j = 0; j < 4; j++)
                for (i = 0; i < kernel_end3; i++)
                    *(col + (input_line + j) * kernel_size + kernel_num + i) = result[(j << 2) + i];
        }
        if (input_end3)
        {
            cur_input = (float*)(input + input_line * in_ch);
#ifdef __aarch64__
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                sgemm_4x4_deconv_a53(cur_input, cur_kernel, in_ch, result, 4);
            else
                sgemm_4x4_deconv_a72(cur_input, cur_kernel, in_ch, result, 4);
#else
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                sgemm_4x4_deconv_a7(cur_input, cur_kernel, in_ch, result, 4);
            else
                sgemm_4x4_deconv_a17(cur_input, cur_kernel, in_ch, result, 4);
#endif
            for (j = 0; j < input_end3; j++)
                for (i = 0; i < kernel_end3; i++)
                    *(col + (input_line + j) * kernel_size + kernel_num + i) = result[(j << 2) + i];
        }
    }
}

int deconv_hcl_prerun(struct tensor* input_tensor,
                      struct tensor* filter_tensor,
                      struct tensor* output_tensor,
                      struct deconv_priv_info* priv_info,
                      struct deconv_param* param)
{
    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int out_ch = output_tensor->dims[1] / group;
    int in_ch = input_tensor->dims[1] / group;
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];

    int input_size = in_ch * ((in_h * in_w + 3) & -4);
    int kernel_size = kernel_h * kernel_w * out_ch;
    int kernel_size_g = ((kernel_size + 3) & -4) * in_ch;
    {
        int trans_input_size = sizeof(float) * input_size + 128;
        priv_info->trans_input_buffer = (float*)sys_malloc(trans_input_size);
        priv_info->trans_input_size = trans_input_size;

        int interleave_size = sizeof(float) * kernel_size_g * group + 128;
        priv_info->interleave_buffer = (float*)sys_malloc(interleave_size);
        priv_info->interleave_buffer_size = interleave_size;

        int col_size = sizeof(float) * in_h * in_w * kernel_size + 128;
        priv_info->col_buffer = (float*)sys_malloc(col_size);
        priv_info->col_buffer_size = col_size;
    }

    interleave(filter_tensor, priv_info, param);

    return 0;
}

int deconv_hcl_postrun(struct deconv_priv_info* priv_info)
{
    if (priv_info->interleave_buffer != NULL)
    {
        sys_free(priv_info->interleave_buffer);
        priv_info->interleave_buffer = NULL;
    }

    if (priv_info->trans_input_buffer != NULL)
    {
        sys_free(priv_info->trans_input_buffer);
        priv_info->trans_input_buffer = NULL;
    }

    if (priv_info->col_buffer != NULL)
    {
        sys_free(priv_info->col_buffer);
        priv_info->col_buffer = NULL;
    }

    return 0;
}

int deconv_hcl_run(struct tensor* input_tensor,
                   struct tensor* filter_tensor,
                   struct tensor* bias_tensor,
                   struct tensor* output_tensor,
                   struct deconv_priv_info* priv_info,
                   struct deconv_param* param,
                   int num_thread,
                   int cpu_affinity)
{
    /* param */
    int group = param->group;
    int ksize = param->kernel_h;
    int stride = param->stride_h;
    int dilation = param->dilation_h;
    int pad = param->pad_w0;
    int act_type = param->activation;

    int batch = input_tensor->dims[0];
    int in_c = input_tensor->dims[1] / group;
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];
    int in_hw = in_h * in_w;
    int input_size = in_c * in_h * in_w;

    int out_c = output_tensor->dims[1] / group;
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int out_hw = out_h * out_w;
    int output_size = out_c * out_h * out_w;

    int kernel_size = out_c * ksize * ksize;
    int kernel_size_g = ((kernel_size + 3) & -4) * in_c;

    /* buffer addr */
    float* input_buf = (float*)input_tensor->data;
    float* output_buf = (float*)output_tensor->data;
    float* biases_buf = (float*)bias_tensor->data;
    float* trans_input_buf = (float*)priv_info->trans_input_buffer;
    float* col_buf = (float*)priv_info->col_buffer;
    float* interleave_buf = (float*)priv_info->interleave_buffer;

    int sgemm_set_num = kernel_size / PER_OUT_CHAN * PER_OUT_CHAN;
    int sgemm_set_remain = kernel_size % PER_OUT_CHAN;
    for (int n = 0; n < batch; n++) // batch size
    {
        for (int g = 0; g < group; g++)
        {
            /* im2col */
            float* cur_input = input_buf + (n * group + g) * input_size;
            float* cur_output = output_buf + (n * group + g) * output_size;
            float* cur_kernel = interleave_buf + g * kernel_size_g;
            transpose_input(cur_input, trans_input_buf, in_hw, in_c);

            /* gemm */
            sgemm_set(trans_input_buf, cur_kernel, col_buf, in_c, in_hw, kernel_size, 0, sgemm_set_num, num_thread, cpu_affinity);
            if (sgemm_set_remain)
                sgemm4x4(trans_input_buf, cur_kernel, col_buf, in_c, in_hw, kernel_size, sgemm_set_num, kernel_size, num_thread, cpu_affinity);
            float* cur_bias = biases_buf ? (biases_buf + g * out_c) : NULL;
            col2im(col_buf, cur_output, cur_bias, out_c, out_w, out_h, ksize, ksize, stride,
                   stride, dilation, dilation, pad, pad, in_w, in_h);
        }
    }

    return 0;
}
