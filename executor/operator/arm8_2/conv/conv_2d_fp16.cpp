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
 * Author: xiaowei@openailab.com
 */

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <arm_neon.h>
#include <sys/time.h>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"

#include "graph.hpp"
#include "operator/convolution.hpp"
#include "compiler_fp16.h"

extern "C" void hgemm_4x16_a55(__fp16* biases, __fp16* input, __fp16* kernel, long kernel_size, __fp16* output,
                               long output_xy, long fused_relu);
extern "C" void hgemm_4x4_a55(__fp16* biases, __fp16* input, __fp16* kernel, long kernel_size, __fp16* output,
                              long output_xy, long fused_relu);

extern "C" void im2col_fp16_1x1(__fp16* input, long input_xy, __fp16* col, long col_cnt, long input_chan);
extern "C" void im2col_fp16_3x3(__fp16* input, long input_x, long input_y, long input_chan, __fp16* col, long stride);

namespace TEngine {

static inline unsigned long get_cur_time(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);

    return (tv.tv_sec * 1000000 + tv.tv_usec);
}

void ConvertF32toF16(void* fp32, void* fp16, int size)
{
    float* mem_f32 = ( float* )fp32;
    __fp16* mem_f16 = ( __fp16* )fp16;
    for(int i = 0; i < size; i++)
        mem_f16[i] = fp32_to_fp16(mem_f32[i]);
}

void ConvertF16toF32(void* fp16, void* fp32, int size)
{
    float* mem_f32 = ( float* )fp32;
    __fp16* mem_f16 = ( __fp16* )fp16;
    for(int i = 0; i < size; i++)
        mem_f32[i] = fp16_to_fp32(mem_f16[i]);
}

namespace conv_fp16 {

#define TYPE_A55 0
#define TYPE_A76 1

const char* conv_name = "CONV_FP16";
const int default_prio = 1500;

void im2col(__fp16* im, __fp16* col, int input_chan, int input_x, int input_y, int kernel_x, int kernel_y, int stride_x,
            int stride_y, int dilation_x, int dilation_y, int pad_w0, int pad_w1, int pad_h0, int pad_h1, int output_x,
            int output_y, int col_start, int col_end)
{
    int kernel_size = kernel_x * kernel_y * input_chan;
    int input_xy = input_x * input_y;
    int pad_x = pad_w0;
    int pad_y = pad_h0;
    __fp16* cur_col = col + col_start * kernel_size;
    bool is_1x1 = (kernel_x == 1) && (kernel_y == 1) && (stride_x == 1) && (stride_y == 1);
    bool is_dilation = (dilation_x != 1) || (dilation_y != 1);
    bool is_3x3 = (kernel_x == 3) && (kernel_y == 3) && (!is_dilation);
    int col_i, col_j, kch, ky, kx, i;

    if(is_1x1)
    {
        {
            int col_cnt = (col_end & -4) - (col_start & -4);
            im2col_fp16_1x1(im + col_start, input_xy, cur_col, col_cnt, input_chan);
            cur_col += col_cnt * kernel_size;
            col_i = col_end & -4;
        }
        // final 4 input
        if(col_end & 0x3)
        {
            for(col_j = 0; col_j < kernel_size; col_j++)
            {
                for(i = 0; i < 4; i++)
                {
                    if((col_i + i) < col_end)
                        *cur_col++ = *(im + input_xy * col_j + col_i + i);
                    else
                        *cur_col++ = 0.0;
                }
            }
        }
    }
    else if(is_3x3)
    {
        bool is_pad0 = (pad_w0 == 0) && (pad_h0 == 0) && (pad_w1 == 0) && (pad_h1 == 0);
        for(col_i = (col_start & -4); col_i < (col_end & -4); col_i += 4)
        {
            cur_col = col + col_i * kernel_size;
            int imy0 = col_i / output_x;
            int imy3 = (col_i + 3) / output_x;
            int imx0 = col_i - imy0 * output_x;
            int imx3 = (col_i + 3) - imy3 * output_x;
            if((imy0 == imy3) &&
               (is_pad0 || (imy0 != 0 && imx0 != 0 && imy0 != (output_y - 1) && imx3 != (output_x - 1))))
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
                for(kch = 0; kch < input_chan; kch++)
                    for(ky = 0; ky < 3; ky++)
                        for(kx = 0; kx < 3; kx++)
                        {
                            int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                            int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                            for(i = 0; i < 4; i++)
                            {
                                if(imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                else
                                    *cur_col++ = 0.0;
                            }
                        }
            }
        }
        // final 4 input
        if(col_end & 0x3)
        {
            int cnt_y[4] = {col_i / output_x, (col_i + 1) / output_x, (col_i + 2) / output_x, (col_i + 3) / output_x};
            int cnt_x[4] = {col_i - cnt_y[0] * output_x, col_i - cnt_y[1] * output_x + 1,
                            col_i - cnt_y[2] * output_x + 2, col_i - cnt_y[3] * output_x + 3};
            int imx_start[4] = {cnt_x[0] * stride_x - pad_x, cnt_x[1] * stride_x - pad_x, cnt_x[2] * stride_x - pad_x,
                                cnt_x[3] * stride_x - pad_x};
            int imy_start[4] = {cnt_y[0] * stride_y - pad_y, cnt_y[1] * stride_y - pad_y, cnt_y[2] * stride_y - pad_y,
                                cnt_y[3] * stride_y - pad_y};
            for(kch = 0; kch < input_chan; kch++)
                for(ky = 0; ky < 3; ky++)
                    for(kx = 0; kx < 3; kx++)
                    {
                        int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                        int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                        for(i = 0; i < 4; i++)
                        {
                            if((col_i + i) < col_end && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 &&
                               imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0.0;
                        }
                    }
        }
    }
    else
    {    // for general cases
        for(col_i = (col_start & -4); col_i < (col_end & -4); col_i += 4)
        {
            int cnt_y[4] = {col_i / output_x, (col_i + 1) / output_x, (col_i + 2) / output_x, (col_i + 3) / output_x};
            int cnt_x[4] = {col_i - cnt_y[0] * output_x, col_i - cnt_y[1] * output_x + 1,
                            col_i - cnt_y[2] * output_x + 2, col_i - cnt_y[3] * output_x + 3};
            int imx_start[4] = {cnt_x[0] * stride_x - pad_x, cnt_x[1] * stride_x - pad_x, cnt_x[2] * stride_x - pad_x,
                                cnt_x[3] * stride_x - pad_x};
            int imy_start[4] = {cnt_y[0] * stride_y - pad_y, cnt_y[1] * stride_y - pad_y, cnt_y[2] * stride_y - pad_y,
                                cnt_y[3] * stride_y - pad_y};
            for(kch = 0; kch < input_chan; kch++)
                for(ky = 0; ky < (kernel_y * dilation_y); ky += dilation_y)
                    for(kx = 0; kx < (kernel_x * dilation_x); kx += dilation_x)
                    {
                        int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                        int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                        for(i = 0; i < 4; i++)
                        {
                            if(imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0.0;
                        }
                    }
        }
        // final 4 input
        if(col_end & 0x3)
        {
            int cnt_y[4] = {col_i / output_x, (col_i + 1) / output_x, (col_i + 2) / output_x, (col_i + 3) / output_x};
            int cnt_x[4] = {col_i - cnt_y[0] * output_x, col_i - cnt_y[1] * output_x + 1,
                            col_i - cnt_y[2] * output_x + 2, col_i - cnt_y[3] * output_x + 3};
            int imx_start[4] = {cnt_x[0] * stride_x - pad_x, cnt_x[1] * stride_x - pad_x, cnt_x[2] * stride_x - pad_x,
                                cnt_x[3] * stride_x - pad_x};
            int imy_start[4] = {cnt_y[0] * stride_y - pad_y, cnt_y[1] * stride_y - pad_y, cnt_y[2] * stride_y - pad_y,
                                cnt_y[3] * stride_y - pad_y};
            for(kch = 0; kch < input_chan; kch++)
                for(ky = 0; ky < (kernel_y * dilation_y); ky += dilation_y)
                    for(kx = 0; kx < (kernel_x * dilation_x); kx += dilation_x)
                    {
                        int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                        int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                        for(i = 0; i < 4; i++)
                        {
                            if((col_i + i) < col_end && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 &&
                               imy[i] < input_y)
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
    for(i = 0; i < (kernel_chan & -16); i += 16)
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
        for(j = 0; j < kernel_size; j++)
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

    for(i = (kernel_chan & -16); i < (kernel_chan & -4); i += 4)
    {
        cur_kernel0 = kernel + kernel_size * i;
        cur_kernel1 = kernel + kernel_size * (i + 1);
        cur_kernel2 = kernel + kernel_size * (i + 2);
        cur_kernel3 = kernel + kernel_size * (i + 3);
        for(j = 0; j < kernel_size; j++)
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
    if((kernel_chan & 0x3) == 3)
    {
        for(j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel0[j];
            *(cur_kernel_interleaved++) = cur_kernel1[j];
            *(cur_kernel_interleaved++) = cur_kernel2[j];
            *(cur_kernel_interleaved++) = 0.0;
        }
    }
    else if((kernel_chan & 0x3) == 2)
    {
        for(j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel0[j];
            *(cur_kernel_interleaved++) = cur_kernel1[j];
            *(cur_kernel_interleaved++) = 0.0;
            *(cur_kernel_interleaved++) = 0.0;
        }
    }
    else if((kernel_chan & 0x3) == 1)
    {
        for(j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel0[j];
            *(cur_kernel_interleaved++) = 0.0;
            *(cur_kernel_interleaved++) = 0.0;
            *(cur_kernel_interleaved++) = 0.0;
        }
    }

    return;
}

static void hgemm4x16(__fp16* col, __fp16* kernel, __fp16* biases, bool bias_term, __fp16* output, int kernel_size,
                      int col_start, int col_end, int kernel_start, int kernel_end, int output_xy, int relu_fused,
                      int cpu_type)
{
    __fp16 result[64];
    __fp16* cur_biases = nullptr;
    __fp16 *cur_col, *cur_kernel, *cur_output;
    int col_line, kernel_num;
    int i, j;
    int col_end3 = col_end & 0x3;

    for(kernel_num = (kernel_start & -16); kernel_num < (kernel_end & -16); kernel_num += 16)
    {
        if(bias_term)
            cur_biases = biases + kernel_num;
        cur_kernel = kernel + kernel_num * kernel_size;
        cur_output = output + kernel_num * output_xy;

        for(col_line = col_start; col_line < (col_end & -4); col_line += 4)
        {
            cur_col = col + col_line * kernel_size;
            hgemm_4x16_a55(cur_biases, cur_col, cur_kernel, kernel_size, cur_output + col_line, output_xy, relu_fused);
        }
        if(col_end3)
        {
            cur_col = col + col_line * kernel_size;
            hgemm_4x16_a55(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, relu_fused);

            for(i = 0; i < 16; i++)
                for(j = 0; j < (col_end3); j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
        }
    }

    return;
}

static void hgemm4x4(__fp16* col, __fp16* kernel, __fp16* biases, bool bias_term, __fp16* output, int kernel_size,
                     int col_start, int col_end, int kernel_start, int kernel_end, int output_xy, int relu_fused,
                     int cpu_type)
{
    __fp16 result[16];
    __fp16* cur_biases = nullptr;
    int col_line, kernel_num;
    __fp16 *cur_col, *cur_kernel, *cur_output;
    int i, j;
    int col_end3 = col_end & 0x3;
    int kernel_end3 = kernel_end & 0x3;

    for(kernel_num = kernel_start & -4; kernel_num < (kernel_end & -4); kernel_num += 4)
    {
        if(bias_term)
            cur_biases = biases + kernel_num;
        cur_kernel = kernel + kernel_num * kernel_size;
        cur_output = output + kernel_num * output_xy;
        for(col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            cur_col = col + col_line * kernel_size;
            hgemm_4x4_a55(cur_biases, cur_col, cur_kernel, kernel_size, cur_output + col_line, output_xy, relu_fused);
        }
        if(col_end3)
        {
            cur_col = col + col_line * kernel_size;
            hgemm_4x4_a55(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, relu_fused);

            for(i = 0; i < 4; i++)
            {
                for(j = 0; j < (col_end3); j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
            }
        }
    }
    if(kernel_end3)
    {
        if(bias_term)
            cur_biases = biases + kernel_num;
        cur_kernel = kernel + kernel_num * kernel_size;
        for(col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            cur_col = col + col_line * kernel_size;
            hgemm_4x4_a55(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, relu_fused);
            for(i = 0; i < kernel_end3; i++)
                for(j = 0; j < 4; j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
        }
        if(col_end3)
        {
            cur_col = col + col_line * kernel_size;
            hgemm_4x4_a55(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, relu_fused);
            for(i = 0; i < (kernel_end3); i++)
            {
                for(j = 0; j < (col_end3); j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
            }
        }
    }
    return;
}

struct im2col_param
{
    __fp16* im;
    __fp16* col;
    int input_chan;
    int input_x;
    int input_y;
    int kernel_x;
    int kernel_y;
    int stride_x;
    int stride_y;
    int dilation_x;
    int dilation_y;
    int pad_w0;
    int pad_w1;
    int pad_h0;
    int pad_h1;
    int output_x;
    int output_y;
    int col_start;
    int col_end;
};

struct hgemm_param
{
    __fp16* col;
    __fp16* kernel;
    __fp16* biases;
    bool bias_term;
    __fp16* output;
    int kernel_size;
    int col_start;
    int col_end;
    int kernel_start;
    int kernel_end;
    int output_xy;
};

struct ConvFP16 : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Reshape(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    bool GetSharedMemorySize(Node*, unsigned int& mem_size) override;
    bool SetSharedMemoryAddr(Node*, void* mem_addr, int mem_size) override;
    bool __fp16_mode;
    bool im2col_aider(int cpu, int seq, void* data /* im2col_param * param */);
    bool hgemm_aider(int cpu, int seq, void* data /* hgemm_param * param */);
    bool hgemm4x4_aider(int cpu, int seq, void* data /* hgemm_param * param */);
    long activation = -1;
    bool dynamic_shape;
};

bool ConvFP16::im2col_aider(int cpu, int seq, void* data)
{
    im2col_param* param = ( im2col_param* )(data);
    im2col(param->im, param->col, param->input_chan, param->input_x, param->input_y, param->kernel_x, param->kernel_y,
           param->stride_x, param->stride_y, param->dilation_x, param->dilation_y, param->pad_w0, param->pad_w1,
           param->pad_h0, param->pad_h1, param->output_x, param->output_y, param->col_start, param->col_end);

    return true;
}

bool ConvFP16::hgemm4x4_aider(int cpu, int seq, void* data)
{
    int cpu_type = -1;
    hgemm_param* param = ( hgemm_param* )(data);
    if(cpu_info->GetCPUModel(cpu)==CPU_A55)
        cpu_type=TYPE_A55;
    else
        cpu_type = TYPE_A76;
    hgemm4x4(param->col, param->kernel, param->biases, param->bias_term, param->output, param->kernel_size,
             param->col_start, param->col_end, param->kernel_start, param->kernel_end, param->output_xy, activation,
             cpu_type);

    return true;
}

bool ConvFP16::hgemm_aider(int cpu, int seq, void* data)
{
    int cpu_type = -1;
    hgemm_param* param = ( hgemm_param* )(data);

    if(cpu_info->GetCPUModel(cpu)==CPU_A55)
        cpu_type=TYPE_A55;
    else
        cpu_type = TYPE_A76;

    hgemm4x16(param->col, param->kernel, param->biases, param->bias_term, param->output, param->kernel_size,
              param->col_start, param->col_end, param->kernel_start, param->kernel_end, param->output_xy, activation,
              cpu_type);

    return true;
}

bool ConvFP16::Prerun(Node* node)
{
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();
    int group = param->group;

    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& output_shape = output_tensor->GetShape();
    int output_chan = output_shape.GetC() / group;

    /* pre-allocate col_buf */
    Tensor* input_tensor = node->GetInputTensor(0);
    TShape& input_shape = input_tensor->GetShape();

    int input_chan = input_shape.GetC() / group;
    int kernel_size = input_chan * param->kernel_h * param->kernel_w;

    if(!dynamic_shape)
    {
        __fp16* col_buf = NULL;
        if(node->ExistAttr("shared_col_buf"))
        {
            col_buf = ( __fp16* )any_cast<void*>(node->GetAttr("shared_col_buf"));
        }
        else
        {
            unsigned int col_size;
            GetSharedMemorySize(node, col_size);
            col_buf = ( __fp16* )mem_alloc(col_size);
            node->SetAttr("col_buf_allocated", col_size);
        }
        (*node)["col_buf"] = col_buf;
    }

    /* packing kernel data */
    Tensor* kernel_tensor = node->GetInputTensor(1);

    // fp32 to fp16
    float* kernel_fp32 = (float* )get_tensor_mem(kernel_tensor);
    const TShape& kernel_shape = kernel_tensor->GetShape();
    ConvertF32toF16(kernel_fp32, kernel_fp32, kernel_shape.GetSize());

    __fp16* kernel_org = (__fp16* )kernel_fp32;
    __fp16* kernel_interleaved = NULL;

    int kernel_interleaved_size_g = kernel_size * ((output_chan + 3) & -4);
    int kernel_size_g = kernel_size * output_chan;
    kernel_interleaved = (__fp16* )mem_alloc(sizeof(__fp16) * (kernel_interleaved_size_g * group) + 128);

    for(int g = 0; g < group; ++g)
    {
        __fp16* kernel = kernel_org + g * kernel_size_g;
        __fp16* kernel_interleaved_g = kernel_interleaved + g * kernel_interleaved_size_g;
        interleave_kernel(kernel, kernel_interleaved_g, output_chan, kernel_size);
    }

    (*node)["kernel_interleaved"] = kernel_interleaved;
    (*node)["input_fp16"] = (__fp16* )malloc(input_tensor->GetTotalSize() / 2);
    (*node)["output_fp16"] = (__fp16* )malloc(output_tensor->GetTotalSize() / 2);

    if(exec_attr->low_mem_mode)
        kernel_tensor->FreeMem();

    // bias fp32 to fp16
    bool have_biases = (node->GetInputNum() > 2);
    if(have_biases)
    {
        Tensor* bias_tensor = node->GetInputTensor(2);
        float* bias_fp32 = (float* )get_tensor_mem(bias_tensor);
        const TShape& bias_shape = bias_tensor->GetShape();
        ConvertF32toF16(bias_fp32, bias_fp32, bias_shape.GetSize());
    }

    return true;
}

bool ConvFP16::Reshape(Node* node)
{
    // TODO: add wino_grad  support
    unsigned int new_col_size;

    GetSharedMemorySize(node, new_col_size);
    if(node->ExistAttr("col_buf_allocated"))
    {
        unsigned int col_size = any_cast<unsigned int>(node->GetAttr("col_buf_allocated"));

        if(new_col_size == col_size)
            return true;

        __fp16* addr = any_cast<__fp16*>(node->GetAttr("col_buf"));
        mem_free(addr);
    }
    __fp16* col_buf = ( __fp16* )mem_alloc(new_col_size);
    (*node)["col_buf"] = col_buf;

    node->SetAttr("col_buf_allocated", new_col_size);
    return true;
}

struct TimeLogger
{
    uint64_t im2col_used;
    uint64_t gemm_used;
    int count;

    TimeLogger()
    {
        im2col_used = gemm_used = count = 0;
    }
    void add_log(uint64_t im2col, uint64_t gemm)
    {
        im2col_used += im2col;
        gemm_used += gemm;
        count++;
    }
    ~TimeLogger()
    {
        const char* env = std::getenv("DEBUG");
        if(count && env)
        {
            uint64_t total = im2col_used + gemm_used;
            printf("count [%d] total time: %lu im2col: %lu (%.2f) gemm: %lu (%.2f)\n", count, total, im2col_used,
                   1.0 * im2col_used / total, gemm_used, 1.0 * gemm_used / total);
            printf("per run time: im2col: %.2f us gemm: %.2f us\n", 1.0 * im2col_used / count, 1.0 * gemm_used / count);
        }
    }
};

TimeLogger time_log;

static double get_current_time(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + (tv.tv_usec / 1000.0);
}

bool ConvFP16::Run(Node* node)
{
    /* input */
    Tensor* input_tensor = node->GetInputTensor(0);
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();

    const TShape& input_shape = input_tensor->GetShape();

    int group = param->group;
    int input_chan = input_shape.GetC() / group;
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();
    int input_size = input_w * input_h * input_chan;
    int pad_w0 = param->pad_w0;    // left padding columns
    int pad_w1 = param->pad_w1;    // right padding columns
    int pad_h0 = param->pad_h0;    // top padding rows
    int pad_h1 = param->pad_h1;    // bottom padding rows
    int stride_x = param->stride_w;
    int stride_y = param->stride_h;
    int dilation_x = param->dilation_w;
    int dilation_y = param->dilation_h;

    // cast data fp32 to fp16
    float* input_data = (float* )get_tensor_mem(input_tensor);
    __fp16* input_org = any_cast<__fp16*>(node->GetAttr("input_fp16"));
    ConvertF32toF16(input_data, input_org, input_shape.GetSize());

    /* output */
    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& output_shape = output_tensor->GetShape();
    __fp16* output_org = any_cast<__fp16*>(node->GetAttr("output_fp16"));

    // __fp16* output_org1 = output_org;
    int output_y = output_shape.GetH();
    int output_x = output_shape.GetW();
    int output_xy = output_x * output_y;
    int output_chan = output_shape.GetC() / group;
    int output_n = output_shape.GetN();

    /* kernel */
    int kernel_x = param->kernel_w;
    int kernel_y = param->kernel_h;
    int kernel_size = input_chan * kernel_x * kernel_y;

    __fp16* kernel_interleaved = any_cast<__fp16*>(node->GetAttr("kernel_interleaved"));
    __fp16* col = any_cast<__fp16*>(node->GetAttr("col_buf"));

    int cpu_number = cpu_info->GetCPUNumber();

    /* biases */
    __fp16* biases = NULL;
    bool have_biases = (node->GetInputNum() > 2);
    if(have_biases)
        biases = ( __fp16* )get_tensor_mem(node->GetInputTensor(2));

    int cpu_type;
    if(cpu_info->GetCPUModel(cpu_info->GetMasterCPU()) == CPU_A55)
        cpu_type = TYPE_A55;
    else
        cpu_type = TYPE_A76;

    /* block size split parameter */
    int L3_CACHE_SIZE = (cpu_type == TYPE_A55) ? 512 * 1024 : 4* 1024 * 1024;
    int kernel_size_l2 = kernel_size;
    int col_cnt_l3 = L3_CACHE_SIZE / sizeof(__fp16) / kernel_size_l2 * 7 / 8;
    col_cnt_l3 = col_cnt_l3 > 4 ? (col_cnt_l3 & -4) : 4;

    /* one image per time */
    for(int i = 0; i < output_n; i++)
    {
        __fp16* input = input_org + i * input_size * group;
        __fp16* output = output_org + i * output_xy * output_chan * group;

        for(int g = 0; g < group; g++)
        {
            __fp16* input_g = input + g * input_size;
            long im2col_start = get_cur_time();

            if(cpu_number == 1)
                im2col(input_g, col, input_chan, input_w, input_h, kernel_x, kernel_y, stride_x, stride_y, dilation_x,
                       dilation_y, pad_w0, pad_w1, pad_h0, pad_h1, output_x, output_y, 0, output_xy);
            else
            {
                std::vector<sub_op_task> task_list;
                std::vector<im2col_param> param_list;

                auto f = std::bind(&ConvFP16::im2col_aider, this, std::placeholders::_1, std::placeholders::_2,
                                   std::placeholders::_3);

                int steps = output_xy / cpu_number;

                steps = (steps + 3) & (~0x3);

                task_list.resize(cpu_number);
                param_list.resize(cpu_number);

                for(int i = 0; i < cpu_number; i++)
                {
                    im2col_param* param = &param_list[i];
                    sub_op_task* task = &task_list[i];

                    task->exec_func = f;
                    task->seq = i;
                    task->data = param;

                    param->im = input_g;
                    param->col = col;
                    param->input_chan = input_chan;
                    param->input_x = input_w;
                    param->input_y = input_h;
                    param->kernel_x = kernel_x;
                    param->kernel_y = kernel_y;
                    param->stride_x = stride_x;
                    param->stride_y = stride_y;
                    param->dilation_x = dilation_x;
                    param->dilation_y = dilation_y;
                    param->pad_w0 = pad_w0;
                    param->pad_w1 = pad_w1;
                    param->pad_h0 = pad_h0;
                    param->pad_h1 = pad_h1;
                    param->output_x = output_x;
                    param->output_y = output_y;
                    param->col_start = i * steps;
                    param->col_end = param->col_start + steps;
                }

                param_list[cpu_number - 1].col_end = output_xy;

                task_dispatch(task_list, -1);
                wait_done();
            }

            long im2col_end = get_cur_time();

            __fp16* kernel_g = kernel_interleaved + g * (kernel_size * ((output_chan + 3) & -4));
            __fp16* output_g = output + g * output_xy * output_chan;
            __fp16* bias_g = biases + g * output_chan;

            std::vector<sub_op_task> task_list;
            std::vector<hgemm_param> param_list;

            int chan_16_num = output_chan / 16;
            int chan_4_num = (output_chan & 0xf) ? 1 : 0;
            int l3_loop = (output_xy - 1) / col_cnt_l3 + 1;
            int max_task_num = l3_loop * (chan_16_num + chan_4_num);

            if(cpu_number > 1)
                param_list.resize(max_task_num);

            // for input block of L2 cache size
            for(int col_i = 0; col_i < output_xy; col_i += col_cnt_l3)
            {
                int col_start = col_i;
                int col_end = col_i + col_cnt_l3;
                col_end = col_end > output_xy ? output_xy : col_end;

                if(cpu_number == 1)
                {
                    hgemm4x16(col, kernel_g, bias_g, have_biases, output_g, kernel_size, col_start, col_end, 0,
                              output_chan & -16, output_xy, activation, cpu_type);
                    if(output_chan & 0xf)
                        hgemm4x4(col, kernel_g, bias_g, have_biases, output_g, kernel_size, col_start, col_end,
                                 output_chan & -16, output_chan, output_xy, activation, cpu_type);
                }
                else
                {
                    auto f = std::bind(&ConvFP16::hgemm_aider, this, std::placeholders::_1, std::placeholders::_2,
                                       std::placeholders::_3);

                    for(int i = 0; i < chan_16_num; i++)
                    {
                        sub_op_task tmp_task;
                        hgemm_param* param = &param_list[task_list.size()];
                        sub_op_task* task = &tmp_task;
                        task->exec_func = f;
                        task->seq = i;
                        task->data = param;

                        param->col = col;
                        param->kernel = kernel_g;
                        param->biases = bias_g;
                        param->bias_term = have_biases;
                        param->output = output_g;
                        param->kernel_size = kernel_size;
                        param->col_start = col_start;
                        param->col_end = col_end;
                        param->kernel_start = i * 16;
                        param->kernel_end = param->kernel_start + 16;
                        param->output_xy = output_xy;

                        task_list.emplace_back(tmp_task);
                    }

                    if(output_chan & 0xf)
                    {
                        auto f = std::bind(&ConvFP16::hgemm4x4_aider, this, std::placeholders::_1,
                                           std::placeholders::_2, std::placeholders::_3);
                        sub_op_task tmp_task;
                        hgemm_param* param = &param_list[task_list.size()];
                        sub_op_task* task = &tmp_task;
                        task->exec_func = f;
                        task->seq = task_list.size() - 1;
                        task->data = param;

                        param->col = col;
                        param->kernel = kernel_g;
                        param->biases = bias_g;
                        param->bias_term = have_biases;
                        param->output = output_g;
                        param->kernel_size = kernel_size;
                        param->col_start = col_start;
                        param->col_end = col_end;
                        param->kernel_start = output_chan & -16;
                        param->kernel_end = output_chan;
                        param->output_xy = output_xy;

                        task_list.emplace_back(tmp_task);
                    }
                }
            }

            if(cpu_number > 1)
            {
                task_dispatch(task_list, -1);
                wait_done();
            }
            long gemm_end = get_cur_time();

            time_log.add_log(im2col_end - im2col_start, gemm_end - im2col_end);
        }
    }

    // cast output data fp16 to fp32
    // start = get_current_time();
    float* output_fp32 = (float* )get_tensor_mem(output_tensor);
    ConvertF16toF32(output_org, output_fp32, output_tensor->GetTotalSize() / 4);

    return true;
}

bool ConvFP16::Postrun(Node* node)
{
    __fp16* addr;

    if(node->ExistAttr("kernel_interleaved"))
    {
        addr = any_cast<__fp16*>(node->GetAttr("kernel_interleaved"));
        mem_free(addr);

        node->RemoveAttr("kernel_interleaved");
    }

    if(node->ExistAttr("col_buf_allocated"))
    {
        addr = any_cast<__fp16*>(node->GetAttr("col_buf"));
        mem_free(addr);
        node->RemoveAttr("col_buf_allocated");
    }

    if(node->ExistAttr("input_fp16"))
    {
        addr = any_cast<__fp16*>(node->GetAttr("input_fp16"));
        mem_free(addr);
        node->RemoveAttr("input_fp16");
    }

    if(node->ExistAttr("output_fp16"))
    {
        addr = any_cast<__fp16*>(node->GetAttr("output_fp16"));
        mem_free(addr);
        node->RemoveAttr("output_fp16");
    }        

    if(node->ExistAttr("col_buf"))
    {
        node->RemoveAttr("col_buf");
    }

    activation = -1;
    return true;
}

bool ConvFP16::GetSharedMemorySize(Node* node, unsigned int& mem_size)
{
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();
    int group = param->group;

    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& output_shape = output_tensor->GetShape();
    int output_y = output_shape.GetH();
    int output_x = output_shape.GetW();

    Tensor* input_tensor = node->GetInputTensor(0);
    TShape& input_shape = input_tensor->GetShape();

    int input_chan = input_shape.GetC() / group;
    int kernel_size = input_chan * param->kernel_h * param->kernel_w;
    int output_xy = output_x * output_y;

    mem_size = (sizeof(__fp16) * (kernel_size * ((output_xy + 3) & -4)) + 128);

    return true;
}

bool ConvFP16::SetSharedMemoryAddr(Node* node, void* mem_addr, int mem_size)
{
    (*node)["shared_col_buf"] = mem_addr;
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));

    if(exec_attr->graph_layout != TENGINE_LAYOUT_NCHW || exec_attr->kernel_mode != EXEC_KERNEL_FP16)
        return nullptr;

    ConvFP16* ops = new ConvFP16();

    ops->need_free = true;

    if(node->IsDynamicShape())
        ops->dynamic_shape = true;
    else
        ops->dynamic_shape = false;

    Operator* op = node->GetOp();
    Convolution* conv_op = dynamic_cast<Convolution*>(op);
    ConvParam* param = conv_op->GetParam();

    ops->activation = param->activation;

    return ops;
}

}    // conv_fp16

void RegisterConv2dFP16(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Convolution", conv_fp16::SelectFunc,
                                                      conv_fp16::default_prio))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio[" << conv_fp16::default_prio << "]\n";
}

}    // namespace TEngine
