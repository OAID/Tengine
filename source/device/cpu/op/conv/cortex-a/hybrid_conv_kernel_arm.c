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

#include <arm_neon.h>


#ifdef __aarch64__

#define PER_OUT_CHAN 16
#define GET_MAX(x, y, z) get_max_arm64(x, y, z)
#define ROUND(x) round(x)

void i8gemm_4x16_a72_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale,
                          long output_xy, long activation, long layout);
void i8gemm_4x4_a72_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale,
                         long output_xy, long activation, long layout);
void i8gemm_4x16_a53_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale,
                          long output_xy, long activation, long layout);
void i8gemm_4x4_a53_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale,
                         long output_xy, long activation, long layout);
void get_max_arm64(float* data, int size, float* max);
void im2col_hybrid_3x3(float* input, int w, int h, int chan, int8_t* col, int stride, float* scale);

#else

#define PER_OUT_CHAN 8
#define GET_MAX(x, y, z) get_max_arm32(x, y, z)
#define ROUND(x) round_na(x)

static inline int round_na(float input)
{
    int output;

#ifdef ANDROID
    asm("vcvtr.s32.f32 %[fp32], %[fp32] \n\t"
        "vmov   %[int32], %[fp32]\n\t"
        : [int32] "=r"(output), [fp32] "+X"(input));
#else
    asm("vcvtr.s32.f32 %[fp32], %[fp32] \n\t"
        "vmov   %[int32], %[fp32]\n\t"
        : [int32] "=r"(output), [fp32] "+w"(input));

#endif
    return output;
};

void i8gemm_4x8_a17_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale,
                         long output_xy, long activation, long layout);
void i8gemm_4x4_a17_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale,
                         long output_xy, long activation, long layout);
void i8gemm_4x8_a7_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale,
                        long output_xy, long activation, long layout);
void i8gemm_4x4_a7_chan(float* biases, int8_t* input, int8_t* kernel, long kernel_size, float* output, float* scale,
                        long output_xy, long activation, long layout);
void get_max_arm32(float* data, int size, float* max);

#endif

void im2col_hybrid_1x1(float* input, int input_xy, int8_t* col, int col_cnt, int input_chan, float* scale);

static void get_max_kernel(const float* data, int size, float* max, int c)
{
    for (int i = 0; i < c; i++)
    {
        const float* cur_data = data + i * size;
        float cur_max = 0;
        GET_MAX(( float* )cur_data, size, &cur_max);

        if (cur_max < 0.0001)
            cur_max = 0.0001;
        max[i] = cur_max;
    }
}

static void get_max(const float* data, int size, float* max)
{
    GET_MAX(( float* )data, size, ( float* )max);

    if ((*max) < 0.0001)
    {
        *max = 0.0001;
    }
}

static void interleave_kernel(float* kernel, int8_t* kernel_interleaved, int kernel_chan, int kernel_size,
                              float* kernel_max)
{
    int i, j, k;
    float* cur_kernel[PER_OUT_CHAN];
    int8_t* cur_kernel_interleaved = kernel_interleaved;

    float scale[kernel_chan];
    for (int i = 0; i < kernel_chan; i++)
        scale[i] = 127.0 / kernel_max[i];

    // interleave PER_OUT_CHAN kernels
    int kernel_chan_align = kernel_chan / PER_OUT_CHAN * PER_OUT_CHAN;
    for (i = 0; i < kernel_chan_align; i += PER_OUT_CHAN)
    {
        for (k = 0; k < PER_OUT_CHAN; k++)
            cur_kernel[k] = kernel + kernel_size * (i + k);
        for (j = 0; j < (kernel_size & -2); j += 2)
        {
            for (k = 0; k < PER_OUT_CHAN; k++)
            {
                *(cur_kernel_interleaved++) = ROUND(cur_kernel[k][j] * scale[i + k]);
                *(cur_kernel_interleaved++) = ROUND(cur_kernel[k][j + 1] * scale[i + k]);
            }
        }
        if (kernel_size & 0x1)
        {
            for (k = 0; k < PER_OUT_CHAN; k++)
            {
                *(cur_kernel_interleaved++) = ROUND(cur_kernel[k][j] * scale[i + k]);
                *(cur_kernel_interleaved++) = 0;
            }
        }
    }
    for (; i < (kernel_chan & -4); i += 4)
    {
        for (k = 0; k < 4; k++)
            cur_kernel[k] = kernel + kernel_size * (i + k);
        for (j = 0; j < (kernel_size & -2); j += 2)
        {
            for (k = 0; k < 4; k++)
            {
                *(cur_kernel_interleaved++) = ROUND(cur_kernel[k][j] * scale[i + k]);
                *(cur_kernel_interleaved++) = ROUND(cur_kernel[k][j + 1] * scale[i + k]);
            }
        }
        if (kernel_size & 0x1)
        {
            for (k = 0; k < 4; k++)
            {
                *(cur_kernel_interleaved++) = ROUND(cur_kernel[k][j] * scale[i + k]);
                *(cur_kernel_interleaved++) = 0;
            }
        }
    }
    // last 4 kernel
    for (k = 0; k < 3; k++)
        cur_kernel[k] = kernel + kernel_size * (i + k);
    int kernel_chan3 = kernel_chan & 0x3;
    if (kernel_chan3)
    {
        for (j = 0; j < (kernel_size & -2); j += 2)
        {
            for (k = 0; k < kernel_chan3; k++)
            {
                *(cur_kernel_interleaved++) = ROUND(cur_kernel[k][j] * scale[i + k]);
                *(cur_kernel_interleaved++) = ROUND(cur_kernel[k][j + 1] * scale[i + k]);
            }
            for (; k < 4; k++)
            {
                *(cur_kernel_interleaved++) = 0.0;
                *(cur_kernel_interleaved++) = 0.0;
            }
        }
        if (kernel_size & 0x1)
        {
            for (k = 0; k < kernel_chan3; k++)
            {
                *(cur_kernel_interleaved++) = ROUND(cur_kernel[k][j] * scale[i + k]);
                *(cur_kernel_interleaved++) = 0;
            }
            for (; k < 4; k++)
            {
                *(cur_kernel_interleaved++) = 0.0;
                *(cur_kernel_interleaved++) = 0.0;
            }
        }
    }
}

static void interleave(struct tensor* filter, struct conv_priv_info* priv_info, struct conv_param* param)
{
    int group = param->group;
    int out_chan = filter->dims[0] / group;
    int kernel_size = filter->dims[1] * filter->dims[2] * filter->dims[3];
    int kernel_size_align = ((kernel_size + 1) & -2);
    int kernel_size_g = kernel_size_align * ((out_chan + 3) & -4);

    float* kernel = filter->data;
    float* kernel_max = ( float* )priv_info->p_kernel_max;
    int8_t* interleave_buf = ( int8_t* )priv_info->interleave_buffer;
    for (int g = 0; g < group; g++)
    {
        float* cur_kernel = kernel + g * out_chan * kernel_size;
        int8_t* cur_interleave = interleave_buf + g * kernel_size_g;
        float* cur_kernel_max = kernel_max + g * out_chan;
        get_max_kernel(cur_kernel, kernel_size, cur_kernel_max, out_chan);
        interleave_kernel(cur_kernel, cur_interleave, out_chan, kernel_size, cur_kernel_max);
    }
}

static void im2col_int8(const float* im, int8_t* col, float input_max, int input_chan, int input_x, int input_y,
                        int kernel_x, int kernel_y, int stride_x, int stride_y, int dilation_x, int dilation_y,
                        int pad_w0, int pad_w1, int pad_h0, int pad_h1, int output_x, int output_y)
{
    int col_end = output_x * output_y;
    float scale = 127.f / input_max;
    int kernel_xy = kernel_x * kernel_y;
    int kernel_size = kernel_xy * input_chan;
    int kernel_size_aligned2 = (kernel_size + 1) & -2;
    int input_xy = input_x * input_y;
    int8_t* cur_col = col;
    int col_i, cnt_y[4], cnt_x[4], imx_start[4], imy_start[4];
    int i, k;
    int col_end3 = col_end & 0x3;
    int kernel_size1 = kernel_size & 0x1;
    int is_1x1 = (kernel_x == 1) && (kernel_y == 1) && (stride_x == 1) && (stride_y == 1);
    int is_3x3 = (kernel_x == 3) && (kernel_y == 3) && (dilation_x == 1) && (dilation_y == 1);
    int is_3x3_dilation = (dilation_x != 1) && (dilation_x == dilation_y) && (stride_x == 1) && (stride_y == 1) &&
                          (dilation_x == pad_w0) && (dilation_x == pad_h0) && (kernel_x == 3) && (kernel_y == 3);

    // is 1x1
    if (is_1x1)
    {
        int col_cnt = col_end & -4;
        im2col_hybrid_1x1(( float* )im, input_xy, cur_col, col_cnt, kernel_size, &scale);
        cur_col += col_cnt * kernel_size_aligned2;
        col_i = col_end & -4;
        // final 4 input
        if (col_end3)
        {
            int kch;
            for (kch = 0; kch < (kernel_size & -2); kch += 2)
                for (i = 0; i < 4; i++)
                    if ((col_i + i) < col_end)
                    {
                        *cur_col++ = ROUND(*(im + input_xy * (kch + 0) + col_i + i) * scale);
                        *cur_col++ = ROUND(*(im + input_xy * (kch + 1) + col_i + i) * scale);
                    }
                    else
                    {
                        *cur_col++ = 0;
                        *cur_col++ = 0;
                    }
            if (kernel_size1)
            {
                for (i = 0; i < 4; i++)
                    if ((col_i + i) < col_end)
                    {
                        *cur_col++ = ROUND(*(im + input_xy * (kch + 0) + col_i + i) * scale);
                        *cur_col++ = 0;
                    }
                    else
                    {
                        *cur_col++ = 0;
                        *cur_col++ = 0;
                    }
            }
        }
    }
    // 3x3 non dilation
    else if (is_3x3)
    {
        int kch, kchp, ky, kyp, imx[4], imy[4];
        int odd_line;
        int is_pad0 = (pad_w0 == 0) && (pad_h0 == 0) && (pad_w1 == 0) && (pad_h1 == 0);
        for (col_i = 0; col_i < (col_end & -4); col_i += 4)
        {
            cur_col = col + col_i * kernel_size_aligned2;
            for (i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_w0;
                imy_start[i] = cnt_y[i] * stride_y - pad_h0;
            }
            if ((cnt_y[0] == cnt_y[3]) &&
                (is_pad0 || (cnt_y[0] > 0 && cnt_x[0] > 0 && cnt_y[0] < (output_y - 1) && cnt_x[3] < (output_x - 1))))
            {
#ifdef __aarch64__
                float* input_start = ( float* )(im + imy_start[0] * input_x + imx_start[0]);
                im2col_hybrid_3x3(input_start, input_x, input_y, input_chan, cur_col, stride_x, &scale);
                cur_col += 4 * kernel_size_aligned2;
#else
                int stride_x2 = stride_x * 2;
                int stride_x3 = stride_x * 3;
                float* l00 = ( float* )(im + imy_start[0] * input_x + imx_start[0]);
                float* l01 = l00 + input_x;
                float* l02 = l00 + input_x * 2;
                float* l10 = l00 + input_xy;
                float* l11 = l10 + input_x;
                float* l12 = l10 + input_x * 2;
                for (kch = 0; kch < (input_chan & -2); kch += 2)
                {
                    cur_col[0] = ROUND(scale * l00[0]);
                    cur_col[1] = ROUND(scale * l00[1]);
                    cur_col[2] = ROUND(scale * l00[0 + stride_x]);
                    cur_col[3] = ROUND(scale * l00[1 + stride_x]);
                    cur_col[4] = ROUND(scale * l00[0 + stride_x2]);
                    cur_col[5] = ROUND(scale * l00[1 + stride_x2]);
                    cur_col[6] = ROUND(scale * l00[0 + stride_x3]);
                    cur_col[7] = ROUND(scale * l00[1 + stride_x3]);
                    cur_col[8] = ROUND(scale * l00[2]);
                    cur_col[9] = ROUND(scale * l01[0]);
                    cur_col[10] = ROUND(scale * l00[2 + stride_x]);
                    cur_col[11] = ROUND(scale * l01[0 + stride_x]);
                    cur_col[12] = ROUND(scale * l00[2 + stride_x2]);
                    cur_col[13] = ROUND(scale * l01[0 + stride_x2]);
                    cur_col[14] = ROUND(scale * l00[2 + stride_x3]);
                    cur_col[15] = ROUND(scale * l01[0 + stride_x3]);
                    cur_col[16] = ROUND(scale * l01[1]);
                    cur_col[17] = ROUND(scale * l01[2]);
                    cur_col[18] = ROUND(scale * l01[1 + stride_x]);
                    cur_col[19] = ROUND(scale * l01[2 + stride_x]);
                    cur_col[20] = ROUND(scale * l01[1 + stride_x2]);
                    cur_col[21] = ROUND(scale * l01[2 + stride_x2]);
                    cur_col[22] = ROUND(scale * l01[1 + stride_x3]);
                    cur_col[23] = ROUND(scale * l01[2 + stride_x3]);
                    cur_col[24] = ROUND(scale * l02[0]);
                    cur_col[25] = ROUND(scale * l02[1]);
                    cur_col[26] = ROUND(scale * l02[0 + stride_x]);
                    cur_col[27] = ROUND(scale * l02[1 + stride_x]);
                    cur_col[28] = ROUND(scale * l02[0 + stride_x2]);
                    cur_col[29] = ROUND(scale * l02[1 + stride_x2]);
                    cur_col[30] = ROUND(scale * l02[0 + stride_x3]);
                    cur_col[31] = ROUND(scale * l02[1 + stride_x3]);
                    cur_col[32] = ROUND(scale * l02[2]);
                    cur_col[33] = ROUND(scale * l10[0]);
                    cur_col[34] = ROUND(scale * l02[2 + stride_x]);
                    cur_col[35] = ROUND(scale * l10[0 + stride_x]);
                    cur_col[36] = ROUND(scale * l02[2 + stride_x2]);
                    cur_col[37] = ROUND(scale * l10[0 + stride_x2]);
                    cur_col[38] = ROUND(scale * l02[2 + stride_x3]);
                    cur_col[39] = ROUND(scale * l10[0 + stride_x3]);
                    cur_col[40] = ROUND(scale * l10[1]);
                    cur_col[41] = ROUND(scale * l10[2]);
                    cur_col[42] = ROUND(scale * l10[1 + stride_x]);
                    cur_col[43] = ROUND(scale * l10[2 + stride_x]);
                    cur_col[44] = ROUND(scale * l10[1 + stride_x2]);
                    cur_col[45] = ROUND(scale * l10[2 + stride_x2]);
                    cur_col[46] = ROUND(scale * l10[1 + stride_x3]);
                    cur_col[47] = ROUND(scale * l10[2 + stride_x3]);
                    cur_col[48] = ROUND(scale * l11[0]);
                    cur_col[49] = ROUND(scale * l11[1]);
                    cur_col[50] = ROUND(scale * l11[0 + stride_x]);
                    cur_col[51] = ROUND(scale * l11[1 + stride_x]);
                    cur_col[52] = ROUND(scale * l11[0 + stride_x2]);
                    cur_col[53] = ROUND(scale * l11[1 + stride_x2]);
                    cur_col[54] = ROUND(scale * l11[0 + stride_x3]);
                    cur_col[55] = ROUND(scale * l11[1 + stride_x3]);
                    cur_col[56] = ROUND(scale * l11[2]);
                    cur_col[57] = ROUND(scale * l12[0]);
                    cur_col[58] = ROUND(scale * l11[2 + stride_x]);
                    cur_col[59] = ROUND(scale * l12[0 + stride_x]);
                    cur_col[60] = ROUND(scale * l11[2 + stride_x2]);
                    cur_col[61] = ROUND(scale * l12[0 + stride_x2]);
                    cur_col[62] = ROUND(scale * l11[2 + stride_x3]);
                    cur_col[63] = ROUND(scale * l12[0 + stride_x3]);
                    cur_col[64] = ROUND(scale * l12[1]);
                    cur_col[65] = ROUND(scale * l12[2]);
                    cur_col[66] = ROUND(scale * l12[1 + stride_x]);
                    cur_col[67] = ROUND(scale * l12[2 + stride_x]);
                    cur_col[68] = ROUND(scale * l12[1 + stride_x2]);
                    cur_col[69] = ROUND(scale * l12[2 + stride_x2]);
                    cur_col[70] = ROUND(scale * l12[1 + stride_x3]);
                    cur_col[71] = ROUND(scale * l12[2 + stride_x3]);
                    cur_col += 72;
                    l00 += input_xy * 2;
                    l01 += input_xy * 2;
                    l02 += input_xy * 2;
                    l10 += input_xy * 2;
                    l11 += input_xy * 2;
                    l12 += input_xy * 2;
                }
                if (input_chan & 0x1)
                {
                    cur_col[0] = ROUND(scale * l00[0]);
                    cur_col[1] = ROUND(scale * l00[1]);
                    cur_col[2] = ROUND(scale * l00[0 + stride_x]);
                    cur_col[3] = ROUND(scale * l00[1 + stride_x]);
                    cur_col[4] = ROUND(scale * l00[0 + stride_x2]);
                    cur_col[5] = ROUND(scale * l00[1 + stride_x2]);
                    cur_col[6] = ROUND(scale * l00[0 + stride_x3]);
                    cur_col[7] = ROUND(scale * l00[1 + stride_x3]);
                    cur_col[8] = ROUND(scale * l00[2]);
                    cur_col[9] = ROUND(scale * l01[0]);
                    cur_col[10] = ROUND(scale * l00[2 + stride_x]);
                    cur_col[11] = ROUND(scale * l01[0 + stride_x]);
                    cur_col[12] = ROUND(scale * l00[2 + stride_x2]);
                    cur_col[13] = ROUND(scale * l01[0 + stride_x2]);
                    cur_col[14] = ROUND(scale * l00[2 + stride_x3]);
                    cur_col[15] = ROUND(scale * l01[0 + stride_x3]);
                    cur_col[16] = ROUND(scale * l01[1]);
                    cur_col[17] = ROUND(scale * l01[2]);
                    cur_col[18] = ROUND(scale * l01[1 + stride_x]);
                    cur_col[19] = ROUND(scale * l01[2 + stride_x]);
                    cur_col[20] = ROUND(scale * l01[1 + stride_x2]);
                    cur_col[21] = ROUND(scale * l01[2 + stride_x2]);
                    cur_col[22] = ROUND(scale * l01[1 + stride_x3]);
                    cur_col[23] = ROUND(scale * l01[2 + stride_x3]);
                    cur_col[24] = ROUND(scale * l02[0]);
                    cur_col[25] = ROUND(scale * l02[1]);
                    cur_col[26] = ROUND(scale * l02[0 + stride_x]);
                    cur_col[27] = ROUND(scale * l02[1 + stride_x]);
                    cur_col[28] = ROUND(scale * l02[0 + stride_x2]);
                    cur_col[29] = ROUND(scale * l02[1 + stride_x2]);
                    cur_col[30] = ROUND(scale * l02[0 + stride_x3]);
                    cur_col[31] = ROUND(scale * l02[1 + stride_x3]);
                    cur_col[32] = ROUND(scale * l02[2]);
                    cur_col[33] = 0;
                    cur_col[34] = ROUND(scale * l02[2 + stride_x]);
                    cur_col[35] = 0;
                    cur_col[36] = ROUND(scale * l02[2 + stride_x2]);
                    cur_col[37] = 0;
                    cur_col[38] = ROUND(scale * l02[2 + stride_x3]);
                    cur_col[39] = 0;
                }
#endif
            }
            else
            {
                odd_line = 0;
                kchp = 0;
                kyp = 0;
                for (kch = 0; kch < input_chan; kch++)
                    for (ky = 0; ky < 3; ky++)
                        if (odd_line)
                        {
                            for (i = 0; i < 4; i++)
                            {
                                imy[i] = imy_start[i] + kyp;
                                imx[i] = imx_start[i] + 2;
                                if (imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = ROUND(*(im + input_xy * kchp + input_x * imy[i] + imx[i]) * scale);
                                else
                                    *cur_col++ = 0;
                                imy[i] = imy_start[i] + ky;
                                if (imx_start[i] >= 0 && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ =
                                        ROUND(*(im + input_xy * kch + input_x * imy[i] + imx_start[i]) * scale);
                                else
                                    *cur_col++ = 0;
                            }
                            for (i = 0; i < 4; i++)
                                for (k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + 1 + k;
                                    if (imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                        *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx[i]) * scale);
                                    else
                                        *cur_col++ = 0;
                                }
                            odd_line = 0;
                        }
                        // even line  2n
                        else
                        {
                            for (i = 0; i < 4; i++)
                                imy[i] = imy_start[i] + ky;
                            for (i = 0; i < 4; i++)
                                for (k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + k;
                                    if (imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                        *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx[i]) * scale);
                                    else
                                        *cur_col++ = 0;
                                }
                            kchp = kch;
                            kyp = ky;
                            odd_line = 1;
                        }
                if (kernel_size1)
                    for (i = 0; i < 4; i++)
                    {
                        imy[i] = imy_start[i] + kyp;
                        imx[i] = imx_start[i] + 2;
                        if (imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                            *cur_col++ = ROUND(*(im + input_xy * kchp + input_x * imy[i] + imx[i]) * scale);
                        else
                            *cur_col++ = 0;
                        *cur_col++ = 0;
                    }
            }
        }
        if (col_end3)
        {
            cur_col = col + col_i * kernel_size_aligned2;
            for (i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_w0;
                imy_start[i] = cnt_y[i] * stride_y - pad_h0;
            }
            odd_line = 0;
            kchp = 0;
            kyp = 0;
            for (kch = 0; kch < input_chan; kch++)
                for (ky = 0; ky < 3; ky++)
                    // odd line 1 + 2n
                    if (odd_line)
                    {
                        for (i = 0; i < 4; i++)
                        {
                            imy[i] = imy_start[i] + kyp;
                            imx[i] = imx_start[i] + 2;
                            if ((i < col_end3) && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = ROUND(*(im + input_xy * kchp + input_x * imy[i] + imx[i]) * scale);
                            else
                                *cur_col++ = 0;
                            imy[i] = imy_start[i] + ky;
                            if ((i < col_end3) && imx_start[i] >= 0 && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx_start[i]) * scale);
                            else
                                *cur_col++ = 0;
                        }
                        for (i = 0; i < 4; i++)
                            for (k = 0; k < 2; k++)
                            {
                                imx[i] = imx_start[i] + (1 + k);
                                if ((i < col_end3) && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 &&
                                    imy[i] < input_y)
                                    *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx[i]) * scale);
                                else
                                    *cur_col++ = 0;
                            }
                        odd_line = 0;
                    }
                    // even line  2n + 1
                    else
                    {
                        for (i = 0; i < 4; i++)
                            imy[i] = imy_start[i] + ky;
                        for (i = 0; i < 4; i++)
                            for (k = 0; k < 2; k++)
                            {
                                imx[i] = imx_start[i] + k;
                                if (i < col_end3 && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx[i]) * scale);
                                else
                                    *cur_col++ = 0;
                            }
                        kchp = kch;
                        kyp = ky;
                        odd_line = 1;
                    }
            if (kernel_size1)
                for (i = 0; i < 4; i++)
                {
                    imy[i] = imy_start[i] + kyp;
                    imx[i] = imx_start[i] + 2;
                    if ((i < col_end3) && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                        *cur_col++ = ROUND(*(im + input_xy * kchp + input_x * imy[i] + imx[i]) * scale);
                    else
                        *cur_col++ = 0;
                    *cur_col++ = 0;
                }
        }
    }
    // 3x3 dilation
    else if (is_3x3_dilation)
    {
        int kch, kchp, ky, kyp, imx[4], imy[4];
        int odd_line;
        for (col_i = 0; col_i < (col_end & -4); col_i += 4)
        {
            cur_col = col + col_i * kernel_size_aligned2;
            for (i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_w0;
                imy_start[i] = cnt_y[i] * stride_y - pad_h0;
            }
            if ((cnt_y[0] == cnt_y[3]) && cnt_y[0] >= pad_w0 && cnt_x[0] >= pad_w0 && cnt_y[0] < (output_y - pad_w0) &&
                cnt_x[3] < (output_x - pad_w0))
            {
                int in_c = 0;
                for (in_c = 0; in_c + 1 < input_chan; in_c += 2)
                {
                    float* input_start = ( float* )(im + imy_start[0] * input_x + imx_start[0] + input_xy * in_c);
                    float32x4_t c0l0_4 = vld1q_f32(input_start);
                    c0l0_4 = vmulq_n_f32(c0l0_4, scale);
                    float32x4_t c0l1_4 = vld1q_f32(input_start + pad_w0);
                    c0l1_4 = vmulq_n_f32(c0l1_4, scale);
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l0_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l1_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l0_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l1_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l0_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l1_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l0_4, 3));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l1_4, 3));
                    float32x4_t c0l2_4 = vld1q_f32(input_start + pad_w0 * 2);
                    c0l2_4 = vmulq_n_f32(c0l2_4, scale);
                    input_start += input_x * pad_w0;
                    float32x4_t c0l3_4 = vld1q_f32(input_start);
                    c0l3_4 = vmulq_n_f32(c0l3_4, scale);
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l2_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l3_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l2_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l3_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l2_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l3_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l2_4, 3));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l3_4, 3));
                    float32x4_t c0l4_4 = vld1q_f32(input_start + pad_w0);
                    c0l4_4 = vmulq_n_f32(c0l4_4, scale);
                    float32x4_t c0l5_4 = vld1q_f32(input_start + pad_w0 * 2);
                    c0l5_4 = vmulq_n_f32(c0l5_4, scale);
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l4_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l5_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l4_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l5_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l4_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l5_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l4_4, 3));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l5_4, 3));
                    input_start += input_x * pad_w0;
                    float32x4_t c0l6_4 = vld1q_f32(input_start);
                    c0l6_4 = vmulq_n_f32(c0l6_4, scale);
                    float32x4_t c0l7_4 = vld1q_f32(input_start + pad_w0);
                    c0l7_4 = vmulq_n_f32(c0l7_4, scale);
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l6_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l7_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l6_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l7_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l6_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l7_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l6_4, 3));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l7_4, 3));
                    float32x4_t c0l8_4 = vld1q_f32(input_start + pad_w0 * 2);
                    c0l8_4 = vmulq_n_f32(c0l8_4, scale);
                    input_start = ( float* )(im + imy_start[0] * input_x + imx_start[0] + input_xy * in_c + input_xy);
                    float32x4_t c1l0_4 = vld1q_f32(input_start);
                    c1l0_4 = vmulq_n_f32(c1l0_4, scale);
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l8_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l0_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l8_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l0_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l8_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l0_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l8_4, 3));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l0_4, 3));
                    float32x4_t c1l1_4 = vld1q_f32(input_start + pad_w0);
                    c1l1_4 = vmulq_n_f32(c1l1_4, scale);
                    float32x4_t c1l2_4 = vld1q_f32(input_start + pad_w0 * 2);
                    c1l2_4 = vmulq_n_f32(c1l2_4, scale);
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l1_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l2_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l1_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l2_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l1_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l2_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l1_4, 3));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l2_4, 3));
                    input_start += input_x * pad_w0;
                    float32x4_t c1l3_4 = vld1q_f32(input_start);
                    c1l3_4 = vmulq_n_f32(c1l3_4, scale);
                    float32x4_t c1l4_4 = vld1q_f32(input_start + pad_w0);
                    c1l4_4 = vmulq_n_f32(c1l4_4, scale);
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l3_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l4_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l3_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l4_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l3_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l4_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l3_4, 3));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l4_4, 3));
                    float32x4_t c1l5_4 = vld1q_f32(input_start + pad_w0 * 2);
                    c1l5_4 = vmulq_n_f32(c1l5_4, scale);
                    input_start += input_x * pad_w0;
                    float32x4_t c1l6_4 = vld1q_f32(input_start);
                    c1l6_4 = vmulq_n_f32(c1l6_4, scale);
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l5_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l6_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l5_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l6_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l5_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l6_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l5_4, 3));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l6_4, 3));
                    float32x4_t c1l7_4 = vld1q_f32(input_start + pad_w0);
                    c1l7_4 = vmulq_n_f32(c1l7_4, scale);
                    float32x4_t c1l8_4 = vld1q_f32(input_start + pad_w0 * 2);
                    c1l8_4 = vmulq_n_f32(c1l8_4, scale);
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l7_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l8_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l7_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l8_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l7_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l8_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l7_4, 3));
                    *cur_col++ = ROUND(vgetq_lane_f32(c1l8_4, 3));
                }
                if (input_chan & 0x1)
                {
                    float* input_start = ( float* )(im + imy_start[0] * input_x + imx_start[0] + in_c * input_xy);
                    float32x4_t c0l0_4 = vld1q_f32(input_start);
                    c0l0_4 = vmulq_n_f32(c0l0_4, scale);
                    float32x4_t c0l1_4 = vld1q_f32(input_start + pad_w0);
                    c0l1_4 = vmulq_n_f32(c0l1_4, scale);
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l0_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l1_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l0_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l1_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l0_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l1_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l0_4, 3));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l1_4, 3));
                    float32x4_t c0l2_4 = vld1q_f32(input_start + pad_w0 * 2);
                    c0l2_4 = vmulq_n_f32(c0l2_4, scale);
                    input_start += input_x * pad_w0;
                    float32x4_t c0l3_4 = vld1q_f32(input_start);
                    c0l3_4 = vmulq_n_f32(c0l3_4, scale);
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l2_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l3_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l2_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l3_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l2_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l3_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l2_4, 3));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l3_4, 3));
                    float32x4_t c0l4_4 = vld1q_f32(input_start + pad_w0);
                    c0l4_4 = vmulq_n_f32(c0l4_4, scale);
                    float32x4_t c0l5_4 = vld1q_f32(input_start + pad_w0 * 2);
                    c0l5_4 = vmulq_n_f32(c0l5_4, scale);
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l4_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l5_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l4_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l5_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l4_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l5_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l4_4, 3));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l5_4, 3));
                    input_start += input_x * pad_w0;
                    float32x4_t c0l6_4 = vld1q_f32(input_start);
                    c0l6_4 = vmulq_n_f32(c0l6_4, scale);
                    float32x4_t c0l7_4 = vld1q_f32(input_start + pad_w0);
                    c0l7_4 = vmulq_n_f32(c0l7_4, scale);
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l6_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l7_4, 0));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l6_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l7_4, 1));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l6_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l7_4, 2));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l6_4, 3));
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l7_4, 3));
                    float32x4_t c0l8_4 = vld1q_f32(input_start + pad_w0 * 2);
                    c0l8_4 = vmulq_n_f32(c0l8_4, scale);
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l8_4, 0));
                    *cur_col++ = 0;
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l8_4, 1));
                    *cur_col++ = 0;
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l8_4, 2));
                    *cur_col++ = 0;
                    *cur_col++ = ROUND(vgetq_lane_f32(c0l8_4, 3));
                    *cur_col++ = 0;
                }
            }
            else
            {
                odd_line = 0;
                kchp = 0;
                kyp = 0;
                for (kch = 0; kch < input_chan; kch++)
                    for (ky = 0; ky < 3; ky++)
                        if (odd_line)
                        {
                            for (i = 0; i < 4; i++)
                            {
                                imy[i] = imy_start[i] + kyp * dilation_y;
                                imx[i] = imx_start[i] + 2 * dilation_x;
                                if (imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = ROUND(*(im + input_xy * kchp + input_x * imy[i] + imx[i]) * scale);
                                else
                                    *cur_col++ = 0;
                                imy[i] = imy_start[i] + ky * dilation_y;
                                if (imx_start[i] >= 0 && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ =
                                        ROUND(*(im + input_xy * kch + input_x * imy[i] + imx_start[i]) * scale);
                                else
                                    *cur_col++ = 0;
                            }
                            for (i = 0; i < 4; i++)
                                for (k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + (1 + k) * dilation_x;
                                    if (imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                        *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx[i]) * scale);
                                    else
                                        *cur_col++ = 0;
                                }
                            odd_line = 0;
                        }
                        // even line  2n
                        else
                        {
                            for (i = 0; i < 4; i++)
                                imy[i] = imy_start[i] + ky * dilation_y;
                            for (i = 0; i < 4; i++)
                                for (k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + k * dilation_x;
                                    if (imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                        *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx[i]) * scale);
                                    else
                                        *cur_col++ = 0;
                                }
                            kchp = kch;
                            kyp = ky;
                            odd_line = 1;
                        }
                if (kernel_size1)
                    for (i = 0; i < 4; i++)
                    {
                        imy[i] = imy_start[i] + kyp * dilation_y;
                        imx[i] = imx_start[i] + 2 * dilation_x;
                        if (imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                            *cur_col++ = ROUND(*(im + input_xy * kchp + input_x * imy[i] + imx[i]) * scale);
                        else
                            *cur_col++ = 0;
                        *cur_col++ = 0;
                    }
            }
        }
        if (col_end3)
        {
            cur_col = col + col_i * kernel_size_aligned2;
            for (i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_w0;
                imy_start[i] = cnt_y[i] * stride_y - pad_h0;
            }
            odd_line = 0;
            kchp = 0;
            kyp = 0;
            for (kch = 0; kch < input_chan; kch++)
                for (ky = 0; ky < 3; ky++)
                    // odd line 1 + 2n
                    if (odd_line)
                    {
                        for (i = 0; i < 4; i++)
                        {
                            imy[i] = imy_start[i] + kyp * dilation_y;
                            imx[i] = imx_start[i] + 2 * dilation_x;
                            if ((i < col_end3) && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = ROUND(*(im + input_xy * kchp + input_x * imy[i] + imx[i]) * scale);
                            else
                                *cur_col++ = 0;
                            imy[i] = imy_start[i] + ky * dilation_y;
                            if ((i < col_end3) && imx_start[i] >= 0 && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx_start[i]) * scale);
                            else
                                *cur_col++ = 0;
                        }
                        for (i = 0; i < 4; i++)
                            for (k = 0; k < 2; k++)
                            {
                                imx[i] = imx_start[i] + (1 + k) * dilation_x;
                                if ((i < col_end3) && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 &&
                                    imy[i] < input_y)
                                    *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx[i]) * scale);
                                else
                                    *cur_col++ = 0;
                            }
                        odd_line = 0;
                    }
                    // even line  2n + 1
                    else
                    {
                        for (i = 0; i < 4; i++)
                            imy[i] = imy_start[i] + ky * dilation_y;
                        for (i = 0; i < 4; i++)
                            for (k = 0; k < 2; k++)
                            {
                                imx[i] = imx_start[i] + k * dilation_x;
                                if (i < col_end3 && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx[i]) * scale);
                                else
                                    *cur_col++ = 0;
                            }
                        kchp = kch;
                        kyp = ky;
                        odd_line = 1;
                    }
            if (kernel_size1)
                for (i = 0; i < 4; i++)
                {
                    imy[i] = imy_start[i] + kyp * dilation_y;
                    imx[i] = imx_start[i] + 2 * dilation_x;
                    if ((i < col_end3) && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                        *cur_col++ = ROUND(*(im + input_xy * kchp + input_x * imy[i] + imx[i]) * scale);
                    else
                        *cur_col++ = 0;
                    *cur_col++ = 0;
                }
        }
    }
    // general case for kernel size <=3
    else if ((kernel_x) < 4 && (kernel_y < 4))
    {
        int kch[2], kx[2], ky[2], imx[4][2], imy[4][2], col_j;
        for (col_i = 0; col_i < (col_end & -4); col_i += 4)
        {
            for (i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_w0;
                imy_start[i] = cnt_y[i] * stride_y - pad_h0;
            }
            for (col_j = 0; col_j < (kernel_size & -2); col_j += 2)
            {
                for (k = 0; k < 2; k++)
                {
                    kch[k] = (col_j + k) / kernel_xy;
                    ky[k] = (col_j + k - kch[k] * kernel_xy) / kernel_x;
                    kx[k] = (col_j + k - kch[k] * kernel_xy) - ky[k] * kernel_x;
                    ky[k] = ky[k] * dilation_y;
                    kx[k] = kx[k] * dilation_x;
                    for (i = 0; i < 4; i++)
                    {
                        imx[i][k] = imx_start[i] + kx[k];
                        imy[i][k] = imy_start[i] + ky[k];
                    }
                }
                for (i = 0; i < 4; i++)
                    for (k = 0; k < 2; k++)
                        if (imx[i][k] >= 0 && imx[i][k] < input_x && imy[i][k] >= 0 && imy[i][k] < input_y)
                            *cur_col++ = ROUND(*(im + input_xy * kch[k] + input_x * imy[i][k] + imx[i][k]) * scale);
                        else
                            *cur_col++ = 0;
            }
            if (kernel_size1)
            {
                kch[0] = col_j / kernel_xy;
                ky[0] = (col_j - kch[0] * kernel_xy) / kernel_x;
                kx[0] = col_j - kch[0] * kernel_xy - ky[0] * kernel_x;
                ky[0] = ky[0] * dilation_y;
                kx[0] = kx[0] * dilation_x;
                for (i = 0; i < 4; i++)
                {
                    imx[i][0] = imx_start[i] + kx[0];
                    imy[i][0] = imy_start[i] + ky[0];
                    if (imx[i][0] >= 0 && imx[i][0] < input_x && imy[i][0] >= 0 && imy[i][0] < input_y)
                        *cur_col++ = ROUND(*(im + input_xy * kch[0] + input_x * imy[i][0] + imx[i][0]) * scale);
                    else
                        *cur_col++ = 0;
                    *cur_col++ = 0;
                }
            }
        }
        // final 4 input
        if (col_end3)
        {
            for (i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_w0;
                imy_start[i] = cnt_y[i] * stride_y - pad_h0;
            }
            for (col_j = 0; col_j < (kernel_size & -2); col_j += 2)
            {
                for (k = 0; k < 2; k++)
                {
                    kch[k] = (col_j + k) / kernel_xy;
                    ky[k] = (col_j + k - kch[k] * kernel_xy) / kernel_x;
                    kx[k] = (col_j + k - kch[k] * kernel_xy) - ky[k] * kernel_x;
                    ky[k] = ky[k] * dilation_y;
                    kx[k] = kx[k] * dilation_x;
                    for (i = 0; i < 4; i++)
                    {
                        imx[i][k] = imx_start[i] + kx[k];
                        imy[i][k] = imy_start[i] + ky[k];
                    }
                }
                for (i = 0; i < 4; i++)
                    for (k = 0; k < 2; k++)
                        if ((col_i + i) < col_end && imx[i][k] >= 0 && imx[i][k] < input_x && imy[i][k] >= 0 &&
                            imy[i][k] < input_y)
                            *cur_col++ = ROUND(*(im + input_xy * kch[k] + input_x * imy[i][k] + imx[i][k]) * scale);
                        else
                            *cur_col++ = 0;
            }
            if (kernel_size1)
            {
                kch[0] = col_j / kernel_xy;
                ky[0] = (col_j - kch[0] * kernel_xy) / kernel_x;
                kx[0] = col_j - kch[0] * kernel_xy - ky[0] * kernel_x;
                ky[0] = ky[0] * dilation_y;
                kx[0] = kx[0] * dilation_x;
                for (i = 0; i < 4; i++)
                {
                    imx[i][0] = imx_start[i] + kx[0];
                    imy[i][0] = imy_start[i] + ky[0];
                    if ((col_i + i) < col_end && imx[i][0] >= 0 && imx[i][0] < input_x && imy[i][0] >= 0 &&
                        imy[i][0] < input_y)
                        *cur_col++ = ROUND(*(im + input_xy * kch[0] + input_x * imy[i][0] + imx[i][0]) * scale);
                    else
                        *cur_col++ = 0;
                    *cur_col++ = 0;
                }
            }
        }
    }
    // general case for kernel size >=3
    else
    {
        int kch, kx, ky, kchp, kyp, imx[4], imy[4];
        int odd_line;
        int kernel_x1 = kernel_x & 0x1;
        for (col_i = 0; col_i < (col_end & -4); col_i += 4)
        {
            for (i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_w0;
                imy_start[i] = cnt_y[i] * stride_y - pad_h0;
            }
            odd_line = 0;
            kchp = 0;
            kyp = 0;
            for (kch = 0; kch < input_chan; kch++)
            {
                for (ky = 0; ky < kernel_y; ky++)
                    // odd line 2 + 2n
                    if (odd_line)
                    {
                        for (i = 0; i < 4; i++)
                        {
                            imy[i] = imy_start[i] + kyp * dilation_y;
                            imx[i] = imx_start[i] + (kernel_x - 1) * dilation_x;
                            if (imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = ROUND(*(im + input_xy * kchp + input_x * imy[i] + imx[i]) * scale);
                            else
                                *cur_col++ = 0;
                            imy[i] = imy_start[i] + ky * dilation_y;
                            if (imx_start[i] >= 0 && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx_start[i]) * scale);
                            else
                                *cur_col++ = 0;
                        }
                        for (kx = 1; kx < kernel_x; kx += 2)
                            for (i = 0; i < 4; i++)
                                for (k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + (kx + k) * dilation_x;
                                    if (imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                        *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx[i]) * scale);
                                    else
                                        *cur_col++ = 0;
                                }
                        odd_line = 0;
                    }
                    // even line  2n
                    else
                    {
                        for (i = 0; i < 4; i++)
                            imy[i] = imy_start[i] + ky * dilation_y;
                        for (kx = 0; kx < (kernel_x - 1); kx += 2)
                            for (i = 0; i < 4; i++)
                                for (k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + (kx + k) * dilation_x;
                                    if (imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                        *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx[i]) * scale);
                                    else
                                        *cur_col++ = 0;
                                }
                        kchp = kch;
                        kyp = ky;
                        odd_line = kernel_x1;
                    }
            }
            if (kernel_size1)
                for (i = 0; i < 4; i++)
                {
                    imy[i] = imy_start[i] + kyp * dilation_y;
                    imx[i] = imx_start[i] + (kernel_x - 1) * dilation_x;
                    if (imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                        *cur_col++ = ROUND(*(im + input_xy * kchp + input_x * imy[i] + imx[i]) * scale);
                    else
                        *cur_col++ = 0;
                    *cur_col++ = 0;
                }
        }
        // final 4 input
        if (col_end3)
        {
            for (i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_w0;
                imy_start[i] = cnt_y[i] * stride_y - pad_h0;
            }
            odd_line = 0;
            kchp = 0;
            kyp = 0;
            for (kch = 0; kch < input_chan; kch++)
                for (ky = 0; ky < kernel_y; ky++)
                    // odd line 1 + 2n
                    if (odd_line)
                    {
                        for (i = 0; i < 4; i++)
                        {
                            imy[i] = imy_start[i] + kyp * dilation_y;
                            imx[i] = imx_start[i] + (kernel_x - 1) * dilation_x;
                            if ((i < col_end3) && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = ROUND(*(im + input_xy * kchp + input_x * imy[i] + imx[i]) * scale);
                            else
                                *cur_col++ = 0;
                            imy[i] = imy_start[i] + ky * dilation_y;
                            if ((i < col_end3) && imx_start[i] >= 0 && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx_start[i]) * scale);
                            else
                                *cur_col++ = 0;
                        }
                        for (kx = 1; kx < kernel_x; kx += 2)
                            for (i = 0; i < 4; i++)
                                for (k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + (kx + k) * dilation_x;
                                    if ((i < col_end3) && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 &&
                                        imy[i] < input_y)
                                        *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx[i]) * scale);
                                    else
                                        *cur_col++ = 0;
                                }
                        odd_line = 0;
                    }
                    // even line  2n + 1
                    else
                    {
                        for (i = 0; i < 4; i++)
                            imy[i] = imy_start[i] + ky * dilation_y;
                        for (kx = 0; kx < (kernel_x - 1); kx += 2)
                            for (i = 0; i < 4; i++)
                                for (k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + (kx + k) * dilation_x;
                                    if (i < col_end3 && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 &&
                                        imy[i] < input_y)
                                        *cur_col++ = ROUND(*(im + input_xy * kch + input_x * imy[i] + imx[i]) * scale);
                                    else
                                        *cur_col++ = 0;
                                }
                        kchp = kch;
                        kyp = ky;
                        odd_line = kernel_x1;
                    }
            if (kernel_size1)
                for (i = 0; i < 4; i++)
                {
                    imy[i] = imy_start[i] + kyp * dilation_y;
                    imx[i] = imx_start[i] + (kernel_x - 1) * dilation_x;
                    if ((i < col_end3) && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                        *cur_col++ = ROUND(*(im + input_xy * kchp + input_x * imy[i] + imx[i]) * scale);
                    else
                        *cur_col++ = 0;
                    *cur_col++ = 0;
                }
        }
    }

    return;
}

static void i8gemm_set(int8_t* col, int8_t* kernel, float* biases, float* output, float* scale, int kernel_size,
                       int ch_start, int ch_end, int output_xy, int activation, int num_thread, int cpu_affinity)
{
    int nn_outch = ch_end / PER_OUT_CHAN;
    int col_end3 = output_xy & 0x3;

    if (col_end3)
    {
//#pragma omp parallel for num_threads(num_thread)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * PER_OUT_CHAN;

            float* cur_scale = scale + p;
            float* biasptr = biases ? ( float* )(biases + p) : NULL;
            int8_t* kernel_tmp = (kernel + p * kernel_size);
            float* output_tmp = ( float* )(output + p * output_xy);

            int col_line = 0;
            for (col_line = 0; col_line + 3 < output_xy; col_line += 4)
#ifdef __aarch64__
            {
                int8_t* col_tmp = (col + col_line * kernel_size);
                if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                    i8gemm_4x16_a53_chan(biasptr, col_tmp, kernel_tmp, kernel_size, output_tmp + col_line, cur_scale, output_xy, activation, 0);
                else
                    i8gemm_4x16_a72_chan(biasptr, col_tmp, kernel_tmp, kernel_size, output_tmp + col_line, cur_scale, output_xy, activation, 0);                    
            }
            {
                float result[64];
                int8_t* col_tmp = (col + col_line * kernel_size);

                if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                    i8gemm_4x16_a53_chan(biasptr, col_tmp, kernel_tmp, kernel_size, result, cur_scale, 0, activation, 0);
                else
                    i8gemm_4x16_a72_chan(biasptr, col_tmp, kernel_tmp, kernel_size, result, cur_scale, 0, activation, 0);

                float* output_line[4];
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        output_line[j] = output + (p + i * 4 + j) * output_xy + col_line;
                    }
                    output_line[0][0] = result[i * 16 + 0];
                    output_line[1][0] = result[i * 16 + 5];
                    output_line[2][0] = result[i * 16 + 10];
                    output_line[3][0] = result[i * 16 + 15];
                    if (col_end3 > 1)
                    {
                        output_line[0][1] = result[i * 16 + 4];
                        output_line[1][1] = result[i * 16 + 1];
                        output_line[2][1] = result[i * 16 + 14];
                        output_line[3][1] = result[i * 16 + 11];
                    }
                    if (col_end3 > 2)
                    {
                        output_line[0][2] = result[i * 16 + 8];
                        output_line[1][2] = result[i * 16 + 13];
                        output_line[2][2] = result[i * 16 + 2];
                        output_line[3][2] = result[i * 16 + 7];
                    }
                }
            }
#else
            {
                int8_t* col_tmp = (col + col_line * kernel_size);
                if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                    i8gemm_4x8_a7_chan(biasptr, col_tmp, kernel_tmp, kernel_size, output_tmp + col_line, cur_scale, output_xy, activation, 0);
                else
                    i8gemm_4x8_a17_chan(biasptr, col_tmp, kernel_tmp, kernel_size, output_tmp + col_line, cur_scale, output_xy, activation, 0);
            }
            {
                float result[32];
                int8_t* col_tmp = (col + col_line * kernel_size);

                if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                    i8gemm_4x8_a7_chan(biasptr, col_tmp, kernel_tmp, kernel_size, result, cur_scale, 0, activation, 0);
                else
                    i8gemm_4x8_a17_chan(biasptr, col_tmp, kernel_tmp, kernel_size, result, cur_scale, 0, activation, 0);

                float* output_line[4];
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        output_line[j] = output + (p + i * 4 + j) * output_xy + col_line;
                    }
                    output_line[0][0] = result[i * 16 + 0];
                    output_line[1][0] = result[i * 16 + 5];
                    output_line[2][0] = result[i * 16 + 10];
                    output_line[3][0] = result[i * 16 + 15];
                    if (col_end3 > 1)
                    {
                        output_line[0][1] = result[i * 16 + 4];
                        output_line[1][1] = result[i * 16 + 1];
                        output_line[2][1] = result[i * 16 + 14];
                        output_line[3][1] = result[i * 16 + 11];
                    }
                    if (col_end3 > 2)
                    {
                        output_line[0][2] = result[i * 16 + 8];
                        output_line[1][2] = result[i * 16 + 13];
                        output_line[2][2] = result[i * 16 + 2];
                        output_line[3][2] = result[i * 16 + 7];
                    }
                }
            }
#endif
        }
    }
    else
    {
#pragma omp parallel for num_threads(num_thread)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * PER_OUT_CHAN;

            float* cur_scale = scale + p;
            float* biasptr = biases ? ( float* )(biases + p) : NULL;
            int8_t* kernel_tmp = (kernel + p * kernel_size);
            float* output_tmp = ( float* )(output + p * output_xy);

            for (int col_line = 0; col_line + 3 < output_xy; col_line += 4)
            {
                int8_t* col_tmp = (col + col_line * kernel_size);
#ifdef __aarch64__
                if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                    i8gemm_4x16_a53_chan(biasptr, col_tmp, kernel_tmp, kernel_size, output_tmp + col_line, cur_scale, output_xy, activation, 0);
                else
                    i8gemm_4x16_a72_chan(biasptr, col_tmp, kernel_tmp, kernel_size, output_tmp + col_line, cur_scale, output_xy, activation, 0);
#else
                if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                    i8gemm_4x8_a7_chan(biasptr, col_tmp, kernel_tmp, kernel_size, output_tmp + col_line, cur_scale, output_xy, activation, 0);
                else
                    i8gemm_4x8_a17_chan(biasptr, col_tmp, kernel_tmp, kernel_size, output_tmp + col_line, cur_scale, output_xy, activation, 0);
#endif
            }
        }
    }
}

static void i8gemm4x4(int8_t* col, int8_t* kernel, float* biases, float* output, float* scale, int kernel_size,
                      int ch_start, int ch_end, int output_xy, int activation, int num_thread, int cpu_affinity)
{
    float result[16];
    float* output_line[4];
    float* cur_biases = NULL;
    int col_line, kernel_num;
    int8_t *cur_col, *cur_kernel;
    float* cur_output;
    int i=0;
    int col_end3 = output_xy & 0x3;
    int kernel_end3 = ch_end & 0x3;

    for (kernel_num = ch_start; kernel_num + 3 < (ch_end & -4); kernel_num += 4)
    {
        float* cur_scale = scale + kernel_num;
        if (biases)
            cur_biases = ( float* )(biases + kernel_num);
        cur_kernel = (kernel + kernel_num * kernel_size);
        cur_output = ( float* )(output + kernel_num * output_xy);
        for (col_line = 0; col_line < (output_xy & -4); col_line += 4)
        {
            cur_col = (col + col_line * kernel_size);
#ifdef __aarch64__
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                i8gemm_4x4_a53_chan(cur_biases, cur_col, cur_kernel, kernel_size, cur_output + col_line, cur_scale, output_xy, activation, 0);
            else
                i8gemm_4x4_a72_chan(cur_biases, cur_col, cur_kernel, kernel_size, cur_output + col_line, cur_scale, output_xy, activation, 0);
#else
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                i8gemm_4x4_a7_chan(cur_biases, cur_col, cur_kernel, kernel_size, cur_output + col_line, cur_scale, output_xy, activation, 0);
            else
                i8gemm_4x4_a17_chan(cur_biases, cur_col, cur_kernel, kernel_size, cur_output + col_line, cur_scale, output_xy, activation, 0);
#endif
        }
        if (col_end3)
        {
            cur_col = (col + col_line * kernel_size);
#ifdef __aarch64__
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                i8gemm_4x4_a53_chan(cur_biases, cur_col, cur_kernel, kernel_size, result, cur_scale, 0, activation, 0);
            else
                i8gemm_4x4_a72_chan(cur_biases, cur_col, cur_kernel, kernel_size, result, cur_scale, 0, activation, 0);
#else
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                i8gemm_4x4_a7_chan(cur_biases, cur_col, cur_kernel, kernel_size, result, cur_scale, 0, activation, 0);
            else
                i8gemm_4x4_a17_chan(cur_biases, cur_col, cur_kernel, kernel_size, result, cur_scale, 0, activation, 0);
#endif
            for (int j = 0; j < 4; j++)
            {
                output_line[j] = output + (kernel_num + j) * output_xy + col_line;
            }
            output_line[0][0] = result[i * 16 + 0];
            output_line[1][0] = result[i * 16 + 5];
            output_line[2][0] = result[i * 16 + 10];
            output_line[3][0] = result[i * 16 + 15];
            if (col_end3 > 1)
            {
                output_line[0][1] = result[i * 16 + 4];
                output_line[1][1] = result[i * 16 + 1];
                output_line[2][1] = result[i * 16 + 14];
                output_line[3][1] = result[i * 16 + 11];
            }
            if (col_end3 > 2)
            {
                output_line[0][2] = result[i * 16 + 8];
                output_line[1][2] = result[i * 16 + 13];
                output_line[2][2] = result[i * 16 + 2];
                output_line[3][2] = result[i * 16 + 7];
            }
        }
    }
    if (kernel_end3)
    {
        float* cur_scale = scale + kernel_num;
        if (biases)
            cur_biases = ( float* )(biases + kernel_num);
        cur_kernel = (kernel + kernel_num * kernel_size);
        for (col_line = 0; col_line < (output_xy & -4); col_line += 4)
        {
            cur_col = (col + col_line * kernel_size);
#ifdef __aarch64__
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                i8gemm_4x4_a53_chan(cur_biases, cur_col, cur_kernel, kernel_size, result, cur_scale, 0, activation, 0);
            else
                i8gemm_4x4_a72_chan(cur_biases, cur_col, cur_kernel, kernel_size, result, cur_scale, 0, activation, 0);
#else
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                i8gemm_4x4_a7_chan(cur_biases, cur_col, cur_kernel, kernel_size, result, cur_scale, 0, activation, 0);
            else
                i8gemm_4x4_a17_chan(cur_biases, cur_col, cur_kernel, kernel_size, result, cur_scale, 0, activation, 0);
#endif
            for (int j = 0; j < 4; j++)
            {
                output_line[j] = output + (kernel_num + j) * output_xy + col_line;
            }
            output_line[0][0] = result[i * 16 + 0];
            output_line[0][1] = result[i * 16 + 4];
            output_line[0][2] = result[i * 16 + 8];
            output_line[0][3] = result[i * 16 + 12];
            if (kernel_end3 > 1)
            {
                output_line[1][0] = result[i * 16 + 5];
                output_line[1][1] = result[i * 16 + 1];
                output_line[1][2] = result[i * 16 + 13];
                output_line[1][3] = result[i * 16 + 9];
            }
            if (kernel_end3 > 2)
            {
                output_line[2][0] = result[i * 16 + 10];
                output_line[2][1] = result[i * 16 + 14];
                output_line[2][2] = result[i * 16 + 2];
                output_line[2][3] = result[i * 16 + 6];
            }
        }
        if (col_end3)
        {
            cur_col = (col + col_line * kernel_size);
#ifdef __aarch64__
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                i8gemm_4x4_a53_chan(cur_biases, cur_col, cur_kernel, kernel_size, result, cur_scale, 0, activation, 0);
            else
                i8gemm_4x4_a72_chan(cur_biases, cur_col, cur_kernel, kernel_size, result, cur_scale, 0, activation, 0);
#else
            if (cpu_affinity == TENGINE_CLUSTER_LITTLE)
                i8gemm_4x4_a7_chan(cur_biases, cur_col, cur_kernel, kernel_size, result, cur_scale, 0, activation, 0);
            else
                i8gemm_4x4_a17_chan(cur_biases, cur_col, cur_kernel, kernel_size, result, cur_scale, 0, activation, 0);
#endif
            for (int j = 0; j < 4; j++)
            {
                output_line[j] = output + (kernel_num + j) * output_xy + col_line;
            }
            output_line[0][0] = result[i * 16 + 0];
            if (col_end3 > 1)
            {
                output_line[0][1] = result[i * 16 + 4];
            }
            if (col_end3 > 2)
            {
                output_line[0][2] = result[i * 16 + 8];
            }
            if (kernel_end3 > 1)
            {
                output_line[1][0] = result[i * 16 + 5];
                if (col_end3 > 1)
                {
                    output_line[1][1] = result[i * 16 + 1];
                }
                if (kernel_end3 > 2)
                {
                    output_line[1][2] = result[i * 16 + 13];
                }
            }
            if (kernel_end3 > 2)
            {
                output_line[2][0] = result[i * 16 + 10];
                if (col_end3 > 1)
                {
                    output_line[2][1] = result[i * 16 + 14];
                }
                if (kernel_end3 > 2)
                {
                    output_line[2][2] = result[i * 16 + 2];
                }
            }
        }
    }
    return;
}

int hybrid_conv_hcl_get_shared_mem_size(struct tensor* input, struct tensor* output, struct conv_param* param)
{
    int group = param->group;
    int input_chan = param->input_channel / group;
    int kernel_size = input_chan * param->kernel_h * param->kernel_w;
    int kernel_size_align = ((kernel_size + 1) & -2);

    int output_xy = output->dims[2] * output->dims[3];

    int mem_size = sizeof(int8_t) * kernel_size_align * ((output_xy + 3) & -4) + 128;

    return mem_size;
}

static int get_private_mem_size(struct tensor* filter, struct conv_param* param)
{
    int group = param->group;
    int out_chan = filter->dims[0] / group;
    int kernel_size = filter->dims[1] * filter->dims[2] * filter->dims[3];
    int kernel_size_align = ((kernel_size + 1) & -2);

    int mem_size = sizeof(int8_t) * kernel_size_align * ((out_chan + 3) & -4) * group + 128;

    return mem_size;
}

int hybrid_conv_hcl_prerun(struct tensor* input_tensor, struct tensor* filter_tensor,
                           struct tensor* output_tensor, struct conv_priv_info* priv_info, struct conv_param* param)
{
    if (!priv_info->external_im2col_mem)
    {
        int mem_size = hybrid_conv_hcl_get_shared_mem_size(input_tensor, output_tensor, param);
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

    if (!priv_info->p_kernel_max)
    {
        int out_chan = param->output_channel * sizeof(float);
        void* mem = sys_malloc(out_chan);
        priv_info->p_kernel_max = mem;
    }

    interleave(filter_tensor, priv_info, param);

    return 0;
}

int hybrid_conv_hcl_postrun(struct conv_priv_info* priv_info)
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

    if (priv_info->p_kernel_max != NULL)
    {
        sys_free(priv_info->p_kernel_max);
        priv_info->p_kernel_max = NULL;
    }

    return 0;
}

int hybrid_conv_hcl_run(struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* bias_tensor,
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
    int kernel_size_align = (kernel_size + 1) & -2;

    int out_c = output_tensor->dims[1] / group;
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int out_hw = out_h * out_w;
    int output_size = out_c * out_h * out_w;
    int out_c_align = ((out_c + 3) & -4);

    /* buffer addr */
    float* input_buf = ( float* )input_tensor->data;
    float* output_buf = ( float* )output_tensor->data;
    float* bias_buf = NULL;
    int8_t* col_buf = ( int8_t* )priv_info->im2col_buffer;
    int8_t* interleave_buf = ( int8_t* )priv_info->interleave_buffer;
    float* kernel_max = ( float* )priv_info->p_kernel_max;

    if (bias_tensor)
        bias_buf = ( float* )bias_tensor->data;

    int sgemm_set_chan = out_c / PER_OUT_CHAN * PER_OUT_CHAN;
    int sgemm_set_remain = out_c % PER_OUT_CHAN;
    float out_scale[out_c_align];
    for (int n = 0; n < batch; n++)    // batch size
    {
        for (int g = 0; g < group; g++)
        {
            /* im2col */
            float* cur_input = input_buf + (n * group + g) * input_size;
            float input_max;

            /* get input scale for quantizaion input data from fp32 to int8 */
            get_max(cur_input, input_size, &input_max);

            /* get output scales for dequantization output data from int32 to fp32 */
            float* cur_kernel_max = kernel_max + g * out_c;
            for (int k = 0; k < out_c; k++)
                out_scale[k] = input_max * cur_kernel_max[k] / (127 * 127);

            im2col_int8(cur_input, col_buf, input_max, in_c, in_w, in_h, kernel_w, kernel_h, stride_w, stride_h,
                        dilation_w, dilation_h, pad_w0, pad_w1, pad_h0, pad_h1, out_w, out_h);

            /* gemm */
            int8_t* cur_kernel = interleave_buf + g * kernel_size_align * out_c_align;
            float* cur_output = output_buf + (n * group + g) * output_size;
            float* cur_bias = bias_buf ? (bias_buf + g * out_c) : NULL;

            i8gemm_set(col_buf, cur_kernel, cur_bias, cur_output, out_scale, kernel_size_align, 0, sgemm_set_chan,
                       out_hw, act_type, num_thread, cpu_affinity);
            if (sgemm_set_remain)
                i8gemm4x4(col_buf, cur_kernel, cur_bias, cur_output, out_scale, kernel_size_align, sgemm_set_chan,
                          out_c, out_hw, act_type, num_thread, cpu_affinity);
        }
    }

    return 0;
}
