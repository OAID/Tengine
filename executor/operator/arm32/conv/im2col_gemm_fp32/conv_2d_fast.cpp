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

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/convolution.hpp"
#include "sys/time.h"
#include "op_utils.hpp"

#include <sys/time.h>

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

static inline unsigned long get_cur_time(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);

    return (tv.tv_sec * 1000000 + tv.tv_usec);
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
            double avg_time = ( double )total / count / 1000.;
            double im2col_ = ( double )im2col_used / count / 1000.;
            double gemm_used_ = ( double )gemm_used / count / 1000.;
            printf("time %2.2f\t im2tol:%2.2f [%2.2f] percent\t gemm:%2.2f [%2.2f] percent\n ", avg_time, im2col_,
                   im2col_ / avg_time * 100., gemm_used_, gemm_used_ / avg_time * 100.);
        }
    }
};

#include "conv_2d_fast_kernel/A17.inl"

extern "C" void im2col_fp32_1x1(float* input, int input_xy, float* col, int col_cnt, int input_chan);

namespace TEngine {

namespace conv_fast {

typedef void (*sgemm_kernel_t)(float* biases, float* input, float* kernel, int kernel_size, float* output,
                               int output_xy, int activation, int layout);
typedef void (*direct_3x3_kernel_t)(float* biases, float* input, float* kernel, float* output, int input_chan,
                                    int input_w, int input_h, int activation);

void direct_k1s1p0_4x12(float* input_data, float* output, float* kernel, float* bias, const int c_in, int hw,
                        int ker_start, int ker_end, bool bias_term, int activation)
{
    float* bias_ptr = NULL;
    int block_hw = hw >> 2;

    // int activation=false;

    // if(activation>=0)
    //    activation=true;

    for(int i = ker_start; i < ker_end; i += 12)
    {
        float* out_ptr = output + i * hw;
        float* ker_ptr = kernel + i * c_in;
        float* inp_ptr = input_data;
        if(bias_term)
        {
            bias_ptr = bias + i;
        }
        for(int k = 0; k < block_hw; k++)
        {
            direct_k1s1p0_4x12_a17(bias_ptr, inp_ptr, ker_ptr, out_ptr, hw, c_in, activation);
            out_ptr += 4;
            inp_ptr += 4;
        }
    }
}
void direct_k1s1p0_4x4(float* input_data, float* output, float* kernel, float* bias, const int c_in, const int hw,
                       int ker_start, int ker_end, bool bias_term, int activation)
{
    float* bias_ptr = NULL;
    int block_hw = hw >> 2;
    // int activation=false;

    // if(activation>=0)
    //    activation=true;

    for(int i = ker_start; i < ker_end; i += 4)
    {
        float* out_ptr = output + i * hw;
        float* ker_ptr = kernel + i * c_in;
        float* inp_ptr = input_data;
        if(bias_term)
        {
            bias_ptr = bias + i;
        }
        for(int k = 0; k < block_hw; k++)
        {
            direct_k1s1p0_4x4_a17(bias_ptr, inp_ptr, ker_ptr, out_ptr, hw, c_in, activation);
            out_ptr += 4;
            inp_ptr += 4;
        }
    }
}
static void direct_k3s1p1_4x12(float* biases, float* input, float* kernel, float* output, int input_chan, int input_w,
                               int input_h, int activation, direct_3x3_kernel_t kernel_func)
{
    kernel_func(biases, input, kernel, output, input_chan, input_w, input_h, activation);
}

static void direct_k3s1p1_4x4(float* biases, float* input, float* kernel, float* output, int input_chan, int input_w,
                              int input_h, int activation, direct_3x3_kernel_t kernel_func)
{
    kernel_func(biases, input, kernel, output, input_chan, input_w, input_h, activation);
}

const char* conv_name = "CONV_FAST";
const int default_prio = 500;

void im2col(float* im, float* col, int input_chan, int input_x, int input_y, int kernel_x, int kernel_y, int stride_x,
            int stride_y, int dilation_x, int dilation_y, int pad_w0, int pad_w1, int pad_h0, int pad_h1, int output_x,
            int output_y, int col_start, int col_end)
{
    int kernel_size = kernel_x * kernel_y * input_chan;
    int input_xy = input_x * input_y;
    int pad_x = pad_w0;
    int pad_y = pad_h0;
    float* cur_col = col + col_start * kernel_size;
    bool is_1x1 = (kernel_x == 1) && (kernel_y == 1) && (stride_x == 1) && (stride_y == 1);
    bool is_dilation = (dilation_x != 1) || (dilation_y != 1);
    bool is_3x3 = (kernel_x == 3) && (kernel_y == 3) && (!is_dilation);
    bool is_3x3_dilation = (dilation_x != 1) && (dilation_x == dilation_y) && (stride_x == 1) && (stride_y == 1) &&
                           (dilation_x == pad_x) && (dilation_x == pad_y) && (kernel_x == 3) && (kernel_y == 3);
    int col_i, col_j, kch, ky, kx, i, j;
    int col_end3 = col_end & 0x3;

    if(is_1x1)
    {
        int col_cnt = (col_end & -4) - (col_start & -4);
        im2col_fp32_1x1(( float* )im + col_start, input_xy, cur_col, col_cnt, input_chan);
        cur_col += col_cnt * kernel_size;
        col_i = col_end & -4;
        // final 4 input
        if(col_end3)
        {
            for(col_j = 0; col_j < kernel_size; col_j++)
            {
                for(i = 0; i < col_end3; i++)
                    *cur_col++ = *(im + input_xy * col_j + col_i + i);
                for(i = col_end3; i < 4; i++)
                    *cur_col++ = 0;
            }
        }
    }
    else if(is_3x3)
    {
        int stride_x2 = stride_x * 2;
        int stride_x3 = stride_x * 3;
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
                float* l0 = im + (imy0 * stride_y - pad_y) * input_x + (imx0 * stride_x - pad_x);
                float* l1 = l0 + input_x;
                float* l2 = l0 + input_x * 2;
                for(i = 0; i < input_chan; i++)
                {
                    for(j = 0; j < 3; j++)
                    {
                        cur_col[j * 4 + 0] = l0[j];
                        cur_col[j * 4 + 1] = l0[j + stride_x];
                        cur_col[j * 4 + 2] = l0[j + stride_x2];
                        cur_col[j * 4 + 3] = l0[j + stride_x3];
                        cur_col[j * 4 + 12] = l1[j];
                        cur_col[j * 4 + 13] = l1[j + stride_x];
                        cur_col[j * 4 + 14] = l1[j + stride_x2];
                        cur_col[j * 4 + 15] = l1[j + stride_x3];
                        cur_col[j * 4 + 24] = l2[j];
                        cur_col[j * 4 + 25] = l2[j + stride_x];
                        cur_col[j * 4 + 26] = l2[j + stride_x2];
                        cur_col[j * 4 + 27] = l2[j + stride_x3];
                    }
                    cur_col += 36;
                    l0 += input_xy;
                    l1 += input_xy;
                    l2 += input_xy;
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
        if(col_end3)
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
    else if(is_3x3_dilation)
    {
        for(col_i = (col_start & -4); col_i < (col_end & -4); col_i += 4)
        {
            cur_col = col + col_i * kernel_size;
            int imy0 = col_i / output_x;
            int imy3 = (col_i + 3) / output_x;
            int imx0 = col_i - imy0 * output_x;
            int imx3 = (col_i + 3) - imy3 * output_x;
            if(imy3 == imy0 && (imy0 >= pad_x) && (imx0 >= pad_x) && (imx3 < output_x - pad_x) &&
               (imy3 < output_y - pad_x))
            {
                for(i = 0; i < input_chan; i++)
                {
                    float* im_00 = im + i * input_xy + (imy0 - pad_x) * input_x + imx0 - pad_x;
                    float32x4_t col_4 = vld1q_f32(im_00);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    col_4 = vld1q_f32(im_00 + pad_x);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    col_4 = vld1q_f32(im_00 + pad_x * 2);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    im_00 += input_x * pad_x;
                    col_4 = vld1q_f32(im_00);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    col_4 = vld1q_f32(im_00 + pad_x);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    col_4 = vld1q_f32(im_00 + pad_x * 2);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    im_00 += input_x * pad_x;
                    col_4 = vld1q_f32(im_00);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    col_4 = vld1q_f32(im_00 + pad_x);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    col_4 = vld1q_f32(im_00 + pad_x * 2);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                }
            }
            else
            {
                int cnt_y[4] = {col_i / output_x, (col_i + 1) / output_x, (col_i + 2) / output_x,
                                (col_i + 3) / output_x};
                int cnt_x[4] = {col_i - cnt_y[0] * output_x, col_i - cnt_y[1] * output_x + 1,
                                col_i - cnt_y[2] * output_x + 2, col_i - cnt_y[3] * output_x + 3};
                int imx_start[4] = {cnt_x[0] * stride_x - pad_x, cnt_x[1] * stride_x - pad_x,
                                    cnt_x[2] * stride_x - pad_x, cnt_x[3] * stride_x - pad_x};
                int imy_start[4] = {cnt_y[0] * stride_y - pad_y, cnt_y[1] * stride_y - pad_y,
                                    cnt_y[2] * stride_y - pad_y, cnt_y[3] * stride_y - pad_y};
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
        }
        if(col_end3)
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
        if(col_end3)
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

// interleave 0 ~ (output_chan & -12) kernels with 12 in form of k[0-11][0],k[0-11][1],k[0-11][2]..
// interleave (output_chan & -12) ~ ((output_chan + 3) & -4) tail kernls with 4 in form of
// k[0-3][0],k[0-3][1],k[0-3][2]..
void interleave_kernel(float* kernel, float* kernel_interleaved, int kernel_chan, int kernel_size)
{
    int i, j;
    float *cur_kernel0, *cur_kernel1, *cur_kernel2, *cur_kernel3, *cur_kernel4, *cur_kernel5, *cur_kernel6,
        *cur_kernel7;
    float *cur_kernel8, *cur_kernel9, *cur_kernel10, *cur_kernel11;
    float* cur_kernel_interleaved = kernel_interleaved;

    int kernel_chan12 = (kernel_chan / 12) * 12;

    // interleave 12 kernels
    for(i = 0; i < kernel_chan12; i += 12)
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
        }
    }

    for(; i < (kernel_chan & -4); i += 4)
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

static void sgemm4x12(float* col, float* kernel, float* biases, bool bias_term, float* output, int kernel_size,
                      int col_start, int col_end, int kernel_start, int kernel_end, int output_xy, int activation,
                      sgemm_kernel_t kernel_func)
{
    // int activation=false;

    // if(activation>=0)
    //     activation=true;

    float result[48];
    float* cur_biases = nullptr;
    float *cur_col, *cur_kernel, *cur_output;
    int col_line, kernel_num;
    int i, j;
    int col_end3 = col_end & 0x3;

    for(kernel_num = (kernel_start); kernel_num < (kernel_end); kernel_num += 12)
    {
        if(bias_term)
            cur_biases = ( float* )(biases + kernel_num);
        cur_kernel = ( float* )(kernel + kernel_num * kernel_size);
        cur_output = ( float* )(output + kernel_num * output_xy);
        for(col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            kernel_func(cur_biases, cur_col, cur_kernel, kernel_size, cur_output + col_line, output_xy, activation, 0);
        }
        if(col_end3)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            kernel_func(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);

            for(i = 0; i < 12; i++)
                for(j = 0; j < (col_end3); j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[i * 4 + j];
        }
    }
    return;
}

static void sgemm4x4(float* col, float* kernel, float* biases, bool bias_term, float* output, int kernel_size,
                     int col_start, int col_end, int kernel_start, int kernel_end, int output_xy, int activation,
                     sgemm_kernel_t kernel_func)
{
    float result[16];
    float* cur_biases = nullptr;
    int col_line, kernel_num;
    float *cur_col, *cur_kernel, *cur_output;
    int i, j;
    int col_end3 = col_end & 0x3;
    int kernel_end3 = kernel_end & 0x3;

    for(kernel_num = (kernel_start & -4); kernel_num < (kernel_end & -4); kernel_num += 4)
    {
        if(bias_term)
            cur_biases = ( float* )(biases + kernel_num);
        cur_kernel = ( float* )(kernel + kernel_num * kernel_size);
        cur_output = ( float* )(output + kernel_num * output_xy);
        for(col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            kernel_func(cur_biases, cur_col, cur_kernel, kernel_size, cur_output + col_line, output_xy, activation, 0);
        }
        if(col_end3)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            kernel_func(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
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
            cur_biases = ( float* )(biases + kernel_num);
        cur_kernel = ( float* )(kernel + kernel_num * kernel_size);
        for(col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            kernel_func(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
            for(i = 0; i < kernel_end3; i++)
                for(j = 0; j < 4; j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
        }
        if(col_end3)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            kernel_func(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
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
    float* im;
    float* col;
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

struct sgemm_param
{
    float* col;
    float* kernel;
    float* biases;
    bool bias_term;
    float* output;
    int kernel_size;
    int col_start;
    int col_end;
    int kernel_start;
    int kernel_end;
    int output_xy;
    sgemm_kernel_t kernel_func;
};
struct direct_k1s1p0_param
{
    float* inp;
    float* ker;
    float* bias;
    bool bias_term;
    float* out;
    int hw;
    int k_start;
    int k_end;
    int c_in;
    int activation;
};
struct direct_3x3_param
{
    float* input;
    float* kernel;
    float* bias;
    float* output;
    int input_c;
    int input_w;
    int input_h;
    int step;
    int activation;
    direct_3x3_kernel_t kernel_func;
};

struct ConvFast : public MTNodeOps
{
    ConvFast()
    {
        name_ = "arm_gemm_conv_fp32";
    }
    int cpu_type;
    int cpu_number;

    sgemm_kernel_t kernel_run_4x12 = nullptr;
    sgemm_kernel_t kernel_run_4x4 = nullptr;
    direct_3x3_kernel_t direct_run_4x12 = nullptr;
    direct_3x3_kernel_t direct_run_4x4 = nullptr;

    bool Prerun(Node* node) override;
    bool Reshape(Node* node) override;
    bool Run(Node* node) override;
    bool RealRun(Node* node);
    bool Postrun(Node* node) override;

    bool GetSharedMemorySize(Node*, unsigned int& mem_size) override;
    bool SetSharedMemoryAddr(Node*, void* mem_addr, int mem_size) override;

    bool im2col_aider(int cpu, int seq, void* data /* im2col_param * param */);
    bool sgemm_aider(int cpu, int seq, void* data /* sgemm_param * param*/);
    bool sgemm4x4_aider(int cpu, int seq, void* data /* sgemm_param * param*/);
    bool direct_k1s1p0_4x12_aider(int cpu, int seq, void* data /* sgemm_param * param*/);
    bool direct_k1s1p0_4x4_aider(int cpu, int seq, void* data /* sgemm_param * param*/);
    bool direct_3x3_aider(int cpu, int seq, void* data /* direct_3x3_param * param */);
    bool direct_3x3_run(Node* node);

    bool use_direct_k1s1p0;
    int activation;

    bool DynamicProcess(Node* node);
    bool PostDynamicProcess(Node* node);
    void im2col_gemm_kernel(const int i, const int tid, const void* step, int col_4, float* col, float* input_g,
                            float* kernel_g, float* bias_g, float* output_g, int output_xy, int input_chan, int output_chan12,
                            int output_chan, int input_w, int input_h, int kernel_x, int kernel_y, int stride_x,
                            int stride_y, int dilation_x, int dilation_y, int pad_w0, int pad_w1, int pad_h0,
                            int pad_h1, int output_x, int output_y, bool have_biases, int kernel_size, int activation);
};
void ConvFast::im2col_gemm_kernel(const int i, const int tid, const void* step, int block_4, float* col, float* input_g,
                                  float* kernel_g, float* bias_g, float* output_g, int output_xy, int input_chan, int output_chan12,
                                  int output_chan, int input_w, int input_h, int kernel_x, int kernel_y, int stride_x,
                                  int stride_y, int dilation_x, int dilation_y, int pad_w0, int pad_w1, int pad_h0,
                                  int pad_h1, int output_x, int output_y, bool have_biases, int kernel_size,
                                  int activation)
{
    int my_step = (( int* )step)[0];
    for(int idx=tid;idx<block_4;idx+=my_step)
    {
        int col_start=idx*16;
        int col_end = col_start + 16;
        col_end = (col_end>output_xy)?output_xy:col_end;
        im2col(input_g, col, input_chan, input_w, input_h, kernel_x, kernel_y, stride_x, stride_y, dilation_x,
            dilation_y, pad_w0, pad_w1, pad_h0, pad_h1, output_x, output_y, col_start, col_end);

        sgemm4x12(col, kernel_g, bias_g, have_biases, output_g, kernel_size, col_start, col_end, 0, output_chan12,
                output_xy, activation, kernel_run_4x12);
        if(output_chan != output_chan12)
            sgemm4x4(col, kernel_g, bias_g, have_biases, output_g, kernel_size, col_start, col_end, output_chan12,
                    output_chan, output_xy, activation, kernel_run_4x4);
    }
}

bool ConvFast::direct_3x3_aider(int cpu, int seq, void* data)
{
    direct_3x3_param* param = ( direct_3x3_param* )(data);
    // printf("step:%d\n",param->step);
    if(param->step == 12)
    {
        direct_k3s1p1_4x12(param->bias, param->input, param->kernel, param->output, param->input_c, param->input_w,
                           param->input_h, param->activation, param->kernel_func);
    }
    else
    {
        for(int i = 0; i < param->step; i += 4)
        {
            float* bias = nullptr;
            if(param->bias)
                bias = param->bias + i;
            float* kernel = param->kernel + i * 9 * param->input_c;
            float* output = param->output + i * param->input_w * param->input_h;
            direct_k3s1p1_4x4(bias, param->input, kernel, output, param->input_c, param->input_w, param->input_h,
                              param->activation, param->kernel_func);
        }
    }
    return true;
}

bool ConvFast::im2col_aider(int cpu, int seq, void* data)
{
    im2col_param* param = ( im2col_param* )(data);
    im2col(param->im, param->col, param->input_chan, param->input_x, param->input_y, param->kernel_x, param->kernel_y,
           param->stride_x, param->stride_y, param->dilation_x, param->dilation_y, param->pad_w0, param->pad_w1,
           param->pad_h0, param->pad_h1, param->output_x, param->output_y, param->col_start, param->col_end);

    return true;
}

bool ConvFast::sgemm_aider(int cpu, int seq, void* data)
{
    sgemm_param* param = ( sgemm_param* )(data);

    sgemm4x12(param->col, param->kernel, param->biases, param->bias_term, param->output, param->kernel_size,
              param->col_start, param->col_end, param->kernel_start, param->kernel_end, param->output_xy, activation,
              param->kernel_func);

    return true;
}
bool ConvFast::direct_k1s1p0_4x12_aider(int cpu, int seq, void* data)
{
    direct_k1s1p0_param* param = ( direct_k1s1p0_param* )(data);

    direct_k1s1p0_4x12(param->inp, param->out, param->ker, param->bias, param->c_in, param->hw, param->k_start,
                       param->k_end, param->bias_term, activation);
    return true;
}

bool ConvFast::direct_k1s1p0_4x4_aider(int cpu, int seq, void* data)
{
    direct_k1s1p0_param* param = ( direct_k1s1p0_param* )(data);

    direct_k1s1p0_4x4(param->inp, param->out, param->ker, param->bias, param->c_in, param->hw, param->k_start,
                      param->k_end, param->bias_term, activation);

    return true;
}
bool ConvFast::sgemm4x4_aider(int cpu, int seq, void* data)
{
    sgemm_param* param = ( sgemm_param* )(data);

    sgemm4x4(param->col, param->kernel, param->biases, param->bias_term, param->output, param->kernel_size,
             param->col_start, param->col_end, param->kernel_start, param->kernel_end, param->output_xy, activation,
             param->kernel_func);

    return true;
}

bool ConvFast::Prerun(Node* node)
{
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();

    int group = param->group;

    int pad_x = param->pad_w0;
    int pad_y = param->pad_h0;
    int stride_x = param->stride_w;
    int stride_y = param->stride_h;
    int dilation_x = param->dilation_w;
    int dilation_y = param->dilation_h;

    int kernel_x = param->kernel_w;
    int kernel_y = param->kernel_h;

    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& output_shape = output_tensor->GetShape();
    int output_chan = output_shape.GetC() / group;
    int output_y = output_shape.GetH();
    int output_x = output_shape.GetW();

    /* pre-allocate col_buf */
    Tensor* input_tensor = node->GetInputTensor(0);
    TShape& input_shape = input_tensor->GetShape();

    int input_chan = input_shape.GetC() / group;
    int kernel_size = input_chan * param->kernel_h * param->kernel_w;
    int output_xy = output_x * output_y;

    if(!node->IsDynamicShape())
    {
        if(node->ExistAttr("shared_col_buf"))
        {
            float* addr = ( float* )any_cast<void*>(node->GetAttr("shared_col_buf"));

            (*node)["col_buf"] = addr;
        }
        else
        {
            unsigned int col_size;

            GetSharedMemorySize(node, col_size);

            float* col_buf = ( float* )mem_alloc(col_size);

            (*node)["col_buf"] = col_buf;

            node->SetAttr("col_buf_allocated", col_size);
        }
    }

    /* packing kernel data */
    Tensor* kernel_tensor = node->GetInputTensor(1);
    float* kernel_org = ( float* )get_tensor_mem(kernel_tensor);
    float* kernel_interleaved = NULL;

    if(0 && kernel_y == 1 && kernel_x == 1 && stride_y == 1 && stride_x == 1 && dilation_y == 1 && dilation_x == 1 &&
       pad_y == 0 && pad_x == 0 && param->group == 1 && output_chan % 4 == 0 && output_chan <= 48 && output_xy % 4 == 0)
    {
        use_direct_k1s1p0 = true;
        kernel_interleaved = ( float* )mem_alloc(sizeof(float) * (kernel_size * output_chan));
        interleave_kernel(kernel_org, kernel_interleaved, output_chan, input_chan);
    }
    else
    {
        int kernel_interleaved_size_g = kernel_size * ((output_chan + 3) & -4);
        int kernel_size_g = kernel_size * output_chan;
        kernel_interleaved = ( float* )mem_alloc(sizeof(float) * (kernel_interleaved_size_g * group) + 128);

        for(int g = 0; g < group; ++g)
        {
            float* kernel = kernel_org + g * kernel_size_g;
            float* kernel_interleaved_g = kernel_interleaved + g * kernel_interleaved_size_g;
            interleave_kernel(kernel, kernel_interleaved_g, output_chan, kernel_size);
        }
    }
    (*node)["kernel_interleaved"] = kernel_interleaved;

    if(exec_attr->low_mem_mode)
        kernel_tensor->FreeMem();
#ifdef ON_A17
    if(cpu_type == CPU_A17)
    {
        kernel_run_4x12 = sgemm_4x12_a17;
        kernel_run_4x4 = sgemm_4x4_a17;
        direct_run_4x12 = direct_k3s1p1_4x12_a17;
        direct_run_4x4 = direct_k3s1p1_4x4_a17;
    }
#endif

    return true;
}

bool ConvFast::Reshape(Node* node)
{
    unsigned int new_col_size;

    GetSharedMemorySize(node, new_col_size);

    if(node->ExistAttr("col_buf_allocated"))
    {
        unsigned int col_size = any_cast<unsigned int>(node->GetAttr("col_buf_allocated"));

        if(new_col_size == col_size)
            return true;

        float* addr = any_cast<float*>(node->GetAttr("col_buf"));

        mem_free(addr);
    }

    float* col_buf = ( float* )mem_alloc(new_col_size);
    (*node)["col_buf"] = col_buf;

    node->SetAttr("col_buf_allocated", new_col_size);

    return true;
}

TimeLogger time_log;
bool ConvFast::direct_3x3_run(Node* node)
{
    // printf("run  direct_3x3_run\n");
    Tensor* input_tensor = node->GetInputTensor(0);
    const TShape& input_shape = input_tensor->GetShape();
    int input_chan = input_shape.GetC();
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();
    float* input_org = ( float* )get_tensor_mem(input_tensor);

    float* kernel_interleaved = any_cast<float*>(node->GetAttr("kernel_interleaved"));

    Tensor* output_tensor = node->GetOutputTensor(0);
    int output_chan = output_tensor->GetShape().GetC();
    float* output_org = ( float* )get_tensor_mem(output_tensor);
    float* biases = NULL;
    bool have_biases = (node->GetInputNum() > 2);

    if(have_biases)
    {
        biases = ( float* )get_tensor_mem(node->GetInputTensor(2));
    }
    if(output_chan % 4 != 0)
        return false;

    if(cpu_number == 1)
    {
        int chan_12 = output_chan / 12;
        for(int i = 0; i < chan_12; i++)
        {
            float* kernel = kernel_interleaved + input_chan * 9 * i * 12;
            float* output = output_org + input_w * input_h * i * 12;
            float* bias = nullptr;
            if(biases)
                bias = biases + i * 12;

            direct_k3s1p1_4x12(bias, input_org, kernel, output, input_chan, input_w, input_h, activation,
                               direct_run_4x12);
        }
        int chan_12_less = chan_12 * 12;
        for(int i = chan_12_less; i < output_chan; i += 4)
        {
            float* kernel = kernel_interleaved + input_chan * 9 * i;
            float* output = output_org + input_w * input_h * i;
            float* bias = nullptr;
            if(biases)
                bias = biases + i;
            direct_k3s1p1_4x4(bias, input_org, kernel, output, input_chan, input_w, input_h, activation,
                              direct_run_4x4);
        }
    }
    else
    {
        int chan_12 = output_chan / 12;
        int max_task_num = ((output_chan % 12) ? 1 : 0) + chan_12;
        std::vector<sub_op_task> task_list;
        std::vector<direct_3x3_param> param_list;

        auto f = std::bind(&ConvFast::direct_3x3_aider, this, std::placeholders::_1, std::placeholders::_2,
                           std::placeholders::_3);
        // printf("task_num=  %d \n",max_task_num);
        param_list.resize(max_task_num);

        for(int i = 0; i < max_task_num; i++)
        {
            sub_op_task tmp_task;
            direct_3x3_param* param = &param_list[task_list.size()];
            sub_op_task* task = &tmp_task;

            task->exec_func = f;
            task->seq = i;
            task->data = param;

            param->bias = biases ? (biases + i * 12) : nullptr;
            param->input = input_org;
            param->kernel = kernel_interleaved + input_chan * 9 * i * 12;
            param->output = output_org + input_w * input_h * i * 12;
            param->input_c = input_chan;
            param->input_h = input_h;
            param->input_w = input_w;
            param->activation = activation;
            param->kernel_func = direct_run_4x12;
            param->step = 12;
            if(i == max_task_num - 1)
                param->step = output_chan - (max_task_num - 1) * 12;

            task_list.emplace_back(tmp_task);
        }

        task_dispatch(task_list, -1);
        wait_done();
    }

    return true;
}

bool ConvFast::Run(Node* node)
{
    bool ret = RealRun(node);

    // if(ret && activation>0)
    //{
    //    Tensor * output_tensor=node->GetOutputTensor(0);
    //    float * output_ptr = (float *)get_tensor_mem(output_tensor);

    //    int elem_num=output_tensor->GetTotalSize()/sizeof(float);

    //    for(int i=0;i<elem_num;i++)
    //    {
    //         output_ptr[i]=std::min(output_ptr[i],6.0f);
    //    }
    //}

    return ret;
}

bool ConvFast::RealRun(Node* node)
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
    float* input_org = ( float* )get_tensor_mem(input_tensor);
    float* col = any_cast<float*>(node->GetAttr("col_buf"));

    /* output */
    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& output_shape = output_tensor->GetShape();
    float* output_org = ( float* )get_tensor_mem(output_tensor);
    int output_y = output_shape.GetH();
    int output_x = output_shape.GetW();
    int output_xy = output_x * output_y;
    int output_chan = output_shape.GetC() / group;
    int output_n = output_shape.GetN();

    /* kernel */
    int kernel_x = param->kernel_w;
    int kernel_y = param->kernel_h;
    int kernel_size = input_chan * kernel_x * kernel_y;

    float* kernel_interleaved = any_cast<float*>(node->GetAttr("kernel_interleaved"));

    /* biases */
    float* biases = NULL;
    bool have_biases = (node->GetInputNum() > 2);

    if(have_biases)
    {
        biases = ( float* )get_tensor_mem(node->GetInputTensor(2));
    }

    // if(pad_w0 == 1 && pad_h0 == 1 && kernel_x == 3 && kernel_y == 3 && stride_x == 1 && stride_y == 1 &&
    //    dilation_x == 1 && dilation_y == 1 && group == 1 && input_h >= 6 && input_w >= 6 && (output_chan % 4 == 0))
    // {
    //     if(cpu_number == 1 || (cpu_number > 1 && input_chan < output_chan && output_chan > 120))
    //         return direct_3x3_run(node);
    // }

    /* block size split parameter */
    int L2_CACHE_SIZE = 1024 * 1024;
    int kernel_size_l1 = kernel_size;
    int col_cnt_l2 = L2_CACHE_SIZE / 4 / kernel_size_l1 * 7 / 8;
    col_cnt_l2 = col_cnt_l2 > 4 ? (col_cnt_l2 & -4) : 4;
    int chan_12_num = output_chan / 12;
    int output_chan12 = chan_12_num * 12;

    int block_4 = (output_xy + 15) / 16;
    int new_multi = 0;
    if((block_4 > 16) & (output_chan < 12*cpu_number))
        new_multi = 1;
    /* one image per time */
    for(int i = 0; i < output_n; i++)
    {
        float* input = input_org + i * input_size * group;
        float* output = output_org + i * output_xy * output_chan * group;

        if(use_direct_k1s1p0)
        {
            if(cpu_number == 1)
            {
                direct_k1s1p0_4x12(input, output, kernel_interleaved, biases, input_chan, output_xy, 0, output_chan12,
                                   have_biases, activation);
                if(output_chan12 != output_chan)
                {
                    direct_k1s1p0_4x4(input, output, kernel_interleaved, biases, input_chan, output_xy, output_chan12,
                                      output_chan, have_biases, activation);
                }
            }
            else
            {
                std::vector<sub_op_task> task_list;
                std::vector<direct_k1s1p0_param> param_list;

                int max_task_num = chan_12_num + ((output_chan12 != output_chan) ? 1 : 0);
                param_list.resize(max_task_num);

                auto f = std::bind(&ConvFast::direct_k1s1p0_4x12_aider, this, std::placeholders::_1,
                                   std::placeholders::_2, std::placeholders::_3);
                for(int i = 0; i < chan_12_num; i++)
                {
                    sub_op_task tmp_task;
                    direct_k1s1p0_param* param = &param_list[task_list.size()];
                    sub_op_task* task = &tmp_task;
                    task->exec_func = f;
                    task->seq = i;
                    task->data = param;
                    param->inp = input;
                    param->ker = kernel_interleaved;
                    param->bias = biases;
                    param->bias_term = have_biases;
                    param->out = output;
                    param->hw = output_xy;
                    param->k_start = i * 12;
                    param->k_end = param->k_start + 12;
                    param->c_in = input_chan;
                    task_list.emplace_back(tmp_task);
                }

                if(output_chan != output_chan12)
                {
                    auto f = std::bind(&ConvFast::direct_k1s1p0_4x4_aider, this, std::placeholders::_1,
                                       std::placeholders::_2, std::placeholders::_3);
                    sub_op_task tmp_task;

                    direct_k1s1p0_param* param = &param_list[task_list.size()];
                    sub_op_task* task = &tmp_task;
                    task->exec_func = f;
                    task->seq = 0;
                    task->data = param;
                    param->inp = input;
                    param->ker = kernel_interleaved;
                    param->bias = biases;
                    param->bias_term = have_biases;
                    param->out = output;
                    param->hw = output_xy;
                    param->k_start = output_chan12;
                    param->k_end = output_chan;
                    param->c_in = input_chan;

                    task_list.emplace_back(tmp_task);
                }
                task_dispatch(task_list, -1);
                wait_done();
            }
        }
        else
        {
            for(int g = 0; g < group; g++)
            {
                float* input_g = input + g * input_size;
                float* kernel_g = kernel_interleaved + g * (kernel_size * ((output_chan + 3) & -4));
                float* output_g = output + g * output_xy * output_chan;
                float* bias_g = biases + g * output_chan;

                if(cpu_number == 1)
                {
                    long im2col_start = get_cur_time();

                    im2col(input_g, col, input_chan, input_w, input_h, kernel_x, kernel_y, stride_x, stride_y,
                           dilation_x, dilation_y, pad_w0, pad_w1, pad_h0, pad_h1, output_x, output_y, 0, output_xy);

                    long im2col_end = get_cur_time();
                    // for input block of L2 cache size
                    for(int col_i = 0; col_i < output_xy; col_i += col_cnt_l2)
                    {
                        int col_start = col_i;
                        int col_end = col_i + col_cnt_l2;
                        col_end = col_end > output_xy ? output_xy : col_end;

                        sgemm4x12(col, kernel_g, bias_g, have_biases, output_g, kernel_size, col_start, col_end, 0,
                                  output_chan12, output_xy, activation, kernel_run_4x12);
                        if(output_chan != output_chan12)
                            sgemm4x4(col, kernel_g, bias_g, have_biases, output_g, kernel_size, col_start, col_end,
                                     output_chan12, output_chan, output_xy, activation, kernel_run_4x4);
                    }
                    long gemm_end = get_cur_time();
                    time_log.add_log(im2col_end - im2col_start, gemm_end - im2col_end);
                }
                else if(new_multi == 1)
                {
                    int task_num = (block_4 < cpu_number) ? block_4: cpu_number;
                    MULTI_THREAD_START(task_num, task_num, tid, param_step)
                    im2col_gemm_kernel(0, tid, param_step, block_4, col, input_g, kernel_g, bias_g, output_g, output_xy,
                                       input_chan, output_chan12, output_chan, input_w, input_h, kernel_x, kernel_y, stride_x,
                                       stride_y, dilation_x, dilation_y, pad_w0, pad_w1, pad_h0, pad_h1, output_x,
                                       output_y, have_biases, kernel_size, activation);
                    MULTI_THREAD_END();
                }
                else
                {
                    long im2col_start = get_cur_time();
                    int total_num = output_xy * input_chan * kernel_x * kernel_y;
                    if(total_num < 100 * 1000)
                    {
                        im2col(input_g, col, input_chan, input_w, input_h, kernel_x, kernel_y, stride_x, stride_y,
                               dilation_x, dilation_y, pad_w0, pad_w1, pad_h0, pad_h1, output_x, output_y, 0,
                               output_xy);
                    }
                    else
                    {
                        std::vector<sub_op_task> task_list;
                        std::vector<im2col_param> param_list;

                        task_list.resize(cpu_number);
                        param_list.resize(cpu_number);

                        auto f = std::bind(&ConvFast::im2col_aider, this, std::placeholders::_1, std::placeholders::_2,
                                           std::placeholders::_3);

                        int steps = (output_xy / cpu_number + 3) & (~0x3);
                        int i = 0;

                        while(1)
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

                            if((param->col_end < output_xy) && (i < cpu_number - 1))
                                i++;
                            else
                                break;
                        }

                        param_list[i].col_end = output_xy;

                        task_list.resize(i + 1);

                        task_dispatch(task_list, -1);
                        wait_done();
                    }
                    long im2col_end = get_cur_time();

                    std::vector<sub_op_task> task_list;
                    std::vector<sgemm_param> param_list;

                    int l2_loop = (output_xy - 1) / col_cnt_l2 + 1;
                    int max_task_num = l2_loop * (chan_12_num + ((output_chan12 != output_chan) ? 1 : 0));

                    param_list.resize(max_task_num);

                    // for input block of L2 cache size
                    for(int col_i = 0; col_i < output_xy; col_i += col_cnt_l2)
                    {
                        int col_start = col_i;
                        int col_end = col_i + col_cnt_l2;
                        col_end = col_end > output_xy ? output_xy : col_end;

                        auto f = std::bind(&ConvFast::sgemm_aider, this, std::placeholders::_1, std::placeholders::_2,
                                           std::placeholders::_3);
                        for(int i = 0; i < chan_12_num; i++)
                        {
                            sub_op_task tmp_task;

                            sgemm_param* param = &param_list[task_list.size()];
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
                            param->kernel_start = i * 12;
                            param->kernel_end = param->kernel_start + 12;
                            param->output_xy = output_xy;
                            param->kernel_func = kernel_run_4x12;
                            task_list.emplace_back(tmp_task);
                        }

                        if(output_chan != output_chan12)
                        {
                            auto f = std::bind(&ConvFast::sgemm4x4_aider, this, std::placeholders::_1,
                                               std::placeholders::_2, std::placeholders::_3);
                            sub_op_task tmp_task;

                            sgemm_param* param = &param_list[task_list.size()];
                            sub_op_task* task = &tmp_task;
                            task->exec_func = f;
                            task->seq = 0;
                            task->data = param;

                            param->col = col;
                            param->kernel = kernel_g;
                            param->biases = bias_g;
                            param->bias_term = have_biases;
                            param->output = output_g;
                            param->kernel_size = kernel_size;
                            param->col_start = col_start;
                            param->col_end = col_end;
                            param->kernel_start = output_chan12;
                            param->kernel_end = output_chan;
                            param->output_xy = output_xy;
                            param->kernel_func = kernel_run_4x4;

                            task_list.emplace_back(tmp_task);
                        }
                    }

                    task_dispatch(task_list, -1);
                    wait_done();

                    long gemm_end = get_cur_time();
                    time_log.add_log(im2col_end - im2col_start, gemm_end - im2col_end);
                }
            }
        }    // end else
    }

    return true;
}

bool ConvFast::Postrun(Node* node)
{
    float* addr;
    if(use_direct_k1s1p0)
        use_direct_k1s1p0 = false;

    if(node->ExistAttr("kernel_interleaved"))
    {
        addr = any_cast<float*>(node->GetAttr("kernel_interleaved"));
        mem_free(addr);

        node->RemoveAttr("kernel_interleaved");
    }

    if(node->ExistAttr("col_buf_allocated"))
    {
        addr = any_cast<float*>(node->GetAttr("col_buf"));
        mem_free(addr);

        node->RemoveAttr("col_buf_allocated");
    }

    if(node->ExistAttr("col_buf"))
        node->RemoveAttr("col_buf");

    return true;
}

bool ConvFast::GetSharedMemorySize(Node* node, unsigned int& mem_size)
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

    mem_size = (sizeof(float) * (kernel_size * ((output_xy + 3) & -4)) + 128);

    return true;
}

bool ConvFast::SetSharedMemoryAddr(Node* node, void* mem_addr, int mem_size)
{
    (*node)["shared_col_buf"] = mem_addr;
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));

#ifdef CONFIG_AUTH_DEVICE
    if(!get_auth_float_enabled())
        return nullptr;

    bool int8_enabled = get_auth_int8_enabled();

    if(int8_enabled && exec_attr->kernel_mode != EXEC_KERNEL_FP32)
        return nullptr;
#else
    if(exec_attr->kernel_mode != EXEC_KERNEL_FP32)
        return nullptr;
#endif

    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;
    ConvFast* ops = new ConvFast();

    ops->use_direct_k1s1p0 = false;

    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();

    ops->activation = param->activation;
    int master_cpu = cpu_info->GetMasterCPU();
    ops->cpu_type = cpu_info->GetCPUModel(master_cpu);
    ops->cpu_number = cpu_info->GetCPUNumber();
    return ops;
}

}    // namespace conv_fast

void RegisterConv2dFast(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("arm32", "Convolution", conv_fast::SelectFunc,
                                                      conv_fast::default_prio))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio[" << conv_fast::default_prio << "]\n";
}

}    // namespace TEngine
