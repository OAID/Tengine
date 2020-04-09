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
 * Copyright (c) 2017, Open AI Lab
 * Author: xiaowei@openailab.com
 */
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <math.h>
#include <arm_neon.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"

#include "graph.hpp"
#include "operator/convolution.hpp"
#include "op_utils.hpp"

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif


extern "C" void im2col_fp32_1x1(float* input, long input_xy, float* col, long col_cnt, long input_chan);
extern "C" void im2col_fp32_3x3(float* input, long input_x, long input_y, long input_chan, float* col, long stride);

#include "conv_2d_fast_kernel/A72.inl"                                    

namespace TEngine {

namespace conv_fast {

#define TYPE_A53 0
#define TYPE_A72 1
const char* conv_name = "CONV_FAST";
const int default_prio = 500;

static void direct_k3s1p1_4x16(float* biases, float* input, float* kernel, float* output, int input_chan, int input_w,
                               int input_h, int activatioin, int cpu_type)
{
    {
        direct_k3s1p1_4x16_a72(biases, input, kernel, output, input_chan, input_w, input_h, activatioin);
    }
}

static void direct_k1s1p0_4x16(float* input_data, float* output, float* kernel, float* bias, const int c_in,
                               const int hw, int k_start, int k_end, int bias_term, int activation, int cpu_type)
{
    // only support hw%4==0  cout%16==0
    float* bias_ptr = NULL;
    int block_hw = hw >> 2;

    // [c_out: block 16]

    {
        for(int i = k_start; i < k_end; i += 16)
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
                direct_k1s1p0_4x16_a72(bias_ptr, inp_ptr, ker_ptr, out_ptr, hw, c_in, activation);
                out_ptr += 4;
                inp_ptr += 4;
            }
        }
    }

}

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
    int col_i, col_j, kch, ky, kx, i;
    int col_end3 = col_end & 0x3;

    if(is_1x1)
    {
#if 0
    // equivlent C code
    for(col_i = (col_start & -4); col_i < (col_end & -4) ; col_i+=4 ){
      for(col_j = 0; col_j < kernel_size ; col_j++ ){
        for(i = 0; i < 4; i++)
         * cur_col++ = *(im + input_xy * col_j + col_i + i);
      }
    }
#else
        {
            int col_cnt = (col_end & -4) - (col_start & -4);
            im2col_fp32_1x1(( float* )im + col_start, input_xy, cur_col, col_cnt, input_chan);
            cur_col += col_cnt * kernel_size;
            col_i = col_end & -4;
        }
#endif
        // final 4 input
        if(col_end3)
        {
            for(col_j = 0; col_j < kernel_size; col_j++)
                for(i = 0; i < 4; i++)
                {
                    if(i < col_end3)
                        *cur_col++ = *(im + input_xy * col_j + col_i + i);
                    else
                        *cur_col++ = 0.0;
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
                float* l0 = im + (imy0 * stride_y - pad_y) * input_x + (imx0 * stride_x - pad_x);
#if 0
        // equalavant C code
        int stride_x2 = stride_x * 2;
        int stride_x3 = stride_x * 3;
        float * l1 = l0 + input_x;
        float * l2 = l0 + input_x * 2;
        for(i = 0; i < input_chan; i++){
          for(int j=0 ; j < 3; j++){
             cur_col[j*4+0]  = l0[j];
             cur_col[j*4+1]  = l0[j+stride_x];
             cur_col[j*4+2]  = l0[j+stride_x2];
             cur_col[j*4+3]  = l0[j+stride_x3];
             cur_col[j*4+12] = l1[j];
             cur_col[j*4+13] = l1[j+stride_x];
             cur_col[j*4+14] = l1[j+stride_x2];
             cur_col[j*4+15] = l1[j+stride_x3];
             cur_col[j*4+24] = l2[j];
             cur_col[j*4+25] = l2[j+stride_x];
             cur_col[j*4+26] = l2[j+stride_x2];
             cur_col[j*4+27] = l2[j+stride_x3];
          }
          cur_col += 36;
          l0 += input_xy;
          l1 += input_xy;
          l2 += input_xy;
        }
#else
                {
                    im2col_fp32_3x3(l0, input_x, input_y, input_chan, cur_col, stride_x);
                    cur_col += 4 * kernel_size;
                }
#endif
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

// interleave 0 ~ (output_chan & -16) kernels with 16 in form of k[0-15][0],k[0-15][1],k[0-15][2]..
// interleave (output_chan & -16) ~ ((output_chan + 3) & -4) tail kernls with 4 in form of
// k[0-3][0],k[0-3][1],k[0-3][2]..
void interleave_kernel(float* kernel, float* kernel_interleaved, int kernel_chan, int kernel_size)
{
    int i, j;
    float *cur_kernel0, *cur_kernel1, *cur_kernel2, *cur_kernel3, *cur_kernel4, *cur_kernel5, *cur_kernel6,
        *cur_kernel7;
    float *cur_kernel8, *cur_kernel9, *cur_kernel10, *cur_kernel11, *cur_kernel12, *cur_kernel13, *cur_kernel14,
        *cur_kernel15;
    float* cur_kernel_interleaved = kernel_interleaved;

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

static void sgemm4x16(float* col, float* kernel, float* biases, bool bias_term, float* output, int kernel_size,
                      int col_start, int col_end, int kernel_start, int kernel_end, int output_xy, int activation,
                      int cpu_type)
{
    float result[64];
    float* cur_biases = nullptr;
    float *cur_col, *cur_kernel, *cur_output;
    int col_line, kernel_num;
    int i, j;
    int col_end3 = col_end & 0x3;

    for(kernel_num = (kernel_start & -16); kernel_num < (kernel_end & -16); kernel_num += 16)
    {
        if(bias_term)
            cur_biases = ( float* )(biases + kernel_num);
        cur_kernel = ( float* )(kernel + kernel_num * kernel_size);
        cur_output = ( float* )(output + kernel_num * output_xy);
        for(col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x16_a72(cur_biases, cur_col, cur_kernel, kernel_size, cur_output + col_line, output_xy,
                               activation, 0);

        }
        if(col_end3)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x16_a72(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
            for(i = 0; i < 16; i++)
                for(j = 0; j < (col_end3); j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
        }
    }

    return;
}

static void sgemm4x4(float* col, float* kernel, float* biases, bool bias_term, float* output, int kernel_size,
                     int col_start, int col_end, int kernel_start, int kernel_end, int output_xy, int activation,
                     int cpu_type)
{
    float result[16];
    float* cur_biases = nullptr;
    int col_line, kernel_num;
    float *cur_col, *cur_kernel, *cur_output;
    int i, j;
    int col_end3 = col_end & 0x3;
    int kernel_end3 = kernel_end & 0x3;

    for(kernel_num = kernel_start & -4; kernel_num < (kernel_end & -4); kernel_num += 4)
    {
        if(bias_term)
            cur_biases = ( float* )(biases + kernel_num);
        cur_kernel = ( float* )(kernel + kernel_num * kernel_size);
        cur_output = ( float* )(output + kernel_num * output_xy);
        for(col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, cur_output + col_line, output_xy,
                              activation, 0);
        }
        if(col_end3)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
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
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
            for(i = 0; i < kernel_end3; i++)
                for(j = 0; j < 4; j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
        }
        if(col_end3)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
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
};

struct direct_k1s1p0_4x16_param
{
    float* input;
    float* output;
    float* kernel;
    float* biases;
    int input_chan;
    int output_xy;
    int k_start;
    int k_end;
    bool have_biases;
    int activation;
    int cpu_type;
};

struct direct_param
{
    float* input;
    float* kernel;
    float* bias;
    float* output;
    int input_c;
    int input_w;
    int input_h;
    int activation;
    int cpu_type;
};

struct ConvFast : public MTNodeOps
{
    ConvFast()
    {
        name_ = "arm_gemm_conv_fp32";
    }

    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Reshape(Node* node) override;
    bool Postrun(Node* node) override;
    bool GetSharedMemorySize(Node*, unsigned int& mem_size) override;
    bool SetSharedMemoryAddr(Node*, void* mem_addr, int mem_size) override;

    bool float_mode;
    bool im2col_aider(int cpu, int seq, void* data /* im2col_param * param */);
    bool sgemm_aider(int cpu, int seq, void* data /* sgemm_param * param */);
    bool sgemm4x4_aider(int cpu, int seq, void* data /* sgemm_param * param */);
    bool direct_k1s1p0_4x16_aider(int cpu, int seq, void* data /* conv1x1s1_param * param */);
    bool direct_k3s1p1_4x16_aider(int cpu, int seq, void* data /* direct_param * param */);
    bool direct_run(Node* node);

    int activation;

    bool use_direct_k1s1p0;

    bool dynamic_shape;
};
void conv3x3s1_neon(float* bottom_blob_transform, float* top_blob, float* kernel_interleave, float* bias_data, int in_w,
                    int in_h, int in_c, int out_w, int out_h, int out_c, int activation);

bool ConvFast::direct_k1s1p0_4x16_aider(int cpu, int seq, void* data)
{
    direct_k1s1p0_4x16_param* param = ( direct_k1s1p0_4x16_param* )(data);
    direct_k1s1p0_4x16(param->input, param->output, param->kernel, param->biases, param->input_chan, param->output_xy,
                       param->k_start, param->k_end, param->have_biases, param->activation, param->cpu_type);
    return true;
}
bool ConvFast::direct_k3s1p1_4x16_aider(int cpu, int seq, void* data)
{
    direct_param* param = ( direct_param* )(data);
    direct_k3s1p1_4x16(param->bias, param->input, param->kernel, param->output, param->input_c, param->input_w,
                       param->input_h, param->activation, param->cpu_type);

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

bool ConvFast::sgemm4x4_aider(int cpu, int seq, void* data)
{
    int cpu_type = -1;
    sgemm_param* param = ( sgemm_param* )(data);

    if(cpu_info->GetCPUModel(cpu) == CPU_A72)
        cpu_type = TYPE_A72;
    else
        cpu_type = TYPE_A53;

    sgemm4x4(param->col, param->kernel, param->biases, param->bias_term, param->output, param->kernel_size,
             param->col_start, param->col_end, param->kernel_start, param->kernel_end, param->output_xy, activation,
             cpu_type);

    return true;
}

bool ConvFast::sgemm_aider(int cpu, int seq, void* data)
{
    int cpu_type = -1;
    sgemm_param* param = ( sgemm_param* )(data);

    if(cpu_info->GetCPUModel(cpu) == CPU_A72)
        cpu_type = TYPE_A72;
    else
        cpu_type = TYPE_A53;

    sgemm4x16(param->col, param->kernel, param->biases, param->bias_term, param->output, param->kernel_size,
              param->col_start, param->col_end, param->kernel_start, param->kernel_end, param->output_xy, activation,
              cpu_type);

    return true;
}

bool ConvFast::Prerun(Node* node)
{
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();
    int group = param->group;

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

    if(!dynamic_shape)
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

    float* kernel_interleaved = NULL;
    float* kernel_org = ( float* )get_tensor_mem(kernel_tensor);
    use_direct_k1s1p0 = false;

    if(0 && param->kernel_h == 1 && param->kernel_w == 1 && param->stride_h == 1 && param->stride_w == 1 &&
       param->pad_h0 == 0 && param->pad_w0 == 0 && param->group == 1 && output_xy % 4 == 0 && output_chan % 16 == 0 &&
       output_chan <= 48 && input_chan % 4 == 0)
    {
        use_direct_k1s1p0 = true;
        kernel_interleaved = ( float* )mem_alloc(sizeof(float) * (kernel_size * output_chan));
        interleave_kernel(kernel_org, kernel_interleaved, output_chan, input_chan);
        // printf("node: %s %d %d %d %d 16x4\n",node->GetName().c_str(), input_chan,output_x,output_y,output_chan);
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

    /* check if bias will read over */
    if(node->GetInputNum() > 2)
    {
        Tensor* bias_tensor = node->GetInputTensor(2);
        int bias_size = bias_tensor->GetTotalSize() / sizeof(float);
        int mem_size = get_tensor_mem_size(bias_tensor) / sizeof(float);

        if((bias_size % 16) && (bias_size == mem_size))
        {
            int new_size = bias_size + 32;
            void* new_addr = mem_alloc(sizeof(float) * new_size);
            void* orig_addr = get_tensor_mem(bias_tensor);
            memcpy(new_addr, orig_addr, sizeof(float) * bias_size);
            set_tensor_mem(bias_tensor, new_addr, new_size, mem_free);
        }
    }

    return true;
}

#include <sys/time.h>

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

            printf("count [%d] total time: %lu im2col: %lu (%.2f) gemm: %lu (%.2f)\n", count, total, im2col_used,
                   1.0 * im2col_used / total, gemm_used, 1.0 * gemm_used / total);

            printf("per run time: im2col: %.2f us gemm: %.2f us\n", 1.0 * im2col_used / count, 1.0 * gemm_used / count);
        }
    }
};

TimeLogger time_log;

bool ConvFast::direct_run(Node* node)
{
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
    if(output_chan % 16 != 0)
        return false;

    int cpu_number = cpu_info->GetCPUNumber();
    int cpu_type;

    if(cpu_info->GetCPUModel(cpu_info->GetMasterCPU()) == CPU_A72)
        cpu_type = TYPE_A72;
    else
        cpu_type = TYPE_A53;

    if(cpu_number == 1)
    {
        for(int i = 0; i < output_chan; i += 16)
        {
            float* kernel = kernel_interleaved + input_chan * 9 * i;
            float* output = output_org + input_w * input_h * i;
            float* bias = nullptr;
            if(biases)
                bias = biases + i;

            direct_k3s1p1_4x16(bias, input_org, kernel, output, input_chan, input_w, input_h, activation, cpu_type);
        }
    }
    else
    {
        std::vector<sub_op_task> task_list;
        std::vector<direct_param> param_list;

        auto f = std::bind(&ConvFast::direct_k3s1p1_4x16_aider, this, std::placeholders::_1, std::placeholders::_2,
                           std::placeholders::_3);

        int task_num = output_chan / 16;

        task_list.resize(task_num);
        param_list.resize(task_num);

        for(int i = 0; i < task_num; i++)
        {
            direct_param* param = &param_list[i];
            sub_op_task* task = &task_list[i];

            task->exec_func = f;
            task->seq = i;
            task->data = param;

            param->bias = biases ? (biases + i * 16) : nullptr;
            param->input = input_org;
            param->kernel = kernel_interleaved + input_chan * 9 * i * 16;
            param->output = output_org + input_w * input_h * i * 16;
            param->input_c = input_chan;
            param->input_h = input_h;
            param->input_w = input_w;
            param->activation = activation;
            param->cpu_type = cpu_type;
        }

        task_dispatch(task_list, -1);
        wait_done();
    }

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
    node->SetAttr("col_buf", col_buf);
    node->SetAttr("col_buf_allocated", new_col_size);
    return true;
}

bool ConvFast::Run(Node* node)
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

    int cpu_number = cpu_info->GetCPUNumber();

    if(pad_w0 == 1 && pad_h0 == 1 && kernel_x == 3 && kernel_y == 3 && stride_x == 1 && stride_y == 1 &&
       dilation_x == 1 && dilation_y == 1 && group == 1 && input_chan < output_chan && output_chan <= 160 &&
       (output_chan % 16 == 0))
    {
        if(direct_run(node))
            return true;
    }
    /* biases */

    float* biases = NULL;
    bool have_biases = (node->GetInputNum() > 2);

    if(have_biases)
    {
        biases = ( float* )get_tensor_mem(node->GetInputTensor(2));
    }

    int cpu_type;

    if(cpu_info->GetCPUModel(cpu_info->GetMasterCPU()) == CPU_A72)
        cpu_type = TYPE_A72;
    else
        cpu_type = TYPE_A53;

    /* block size split parameter */
    int L2_CACHE_SIZE = (cpu_type == TYPE_A53) ? 512 * 1024 : 1024 * 1024;
    int kernel_size_l1 = kernel_size;
    int col_cnt_l2 = L2_CACHE_SIZE / 4 / kernel_size_l1 * 7 / 8;
    col_cnt_l2 = col_cnt_l2 > 4 ? (col_cnt_l2 & -4) : 4;
    int chan_16_num = output_chan / 16;

    long im2col_time = 0;
    long gemm_time = 0;
    /* one image per time */
    for(int i = 0; i < output_n; i++)
    {
        float* input = input_org + i * input_size * group;
        float* output = output_org + i * output_xy * output_chan * group;
#if 0
        if(use_direct_k1s1p0)
        {
            if(cpu_number == 1)
            {
                direct_k1s1p0_4x16(input, output, kernel_interleaved, biases, input_chan, output_xy, 0, output_chan,
                                   have_biases, activation, cpu_type);
            }
            else
            {
                // divide tasks by channel, divide 16
                std::vector<sub_op_task> task_list;
                std::vector<direct_k1s1p0_4x16_param> param_list;

                param_list.resize(chan_16_num);

                auto f = std::bind(&ConvFast::direct_k1s1p0_4x16_aider, this, std::placeholders::_1,
                                   std::placeholders::_2, std::placeholders::_3);
                for(int i = 0; i < chan_16_num; i++)
                {
                    sub_op_task tmp_task;
                    direct_k1s1p0_4x16_param* param = &param_list[task_list.size()];
                    sub_op_task* task = &tmp_task;
                    task->exec_func = f;
                    task->seq = i;
                    task->data = param;

                    param->input = input;
                    param->output = output;
                    param->kernel = kernel_interleaved;
                    param->biases = biases;
                    param->input_chan = input_chan;
                    param->output_xy = output_xy;
                    param->k_start = i * 16;
                    param->k_end = param->k_start + 16;
                    param->have_biases = have_biases;
                    param->activation = activation;
                    param->cpu_type = cpu_type;
                    task_list.emplace_back(tmp_task);
                }
                task_dispatch(task_list, -1);
                wait_done();
            }
        }
        else
#endif
        {
            for(int g = 0; g < group; g++)
            {
                float* input_g = input + g * input_size;

                {
                    long im2col_start = get_cur_time();

                    int total_num = output_xy * input_chan * kernel_x * kernel_y;

                    if(cpu_number == 1 || total_num < 100 * 1000)
                        im2col(input_g, col, input_chan, input_w, input_h, kernel_x, kernel_y, stride_x, stride_y,
                               dilation_x, dilation_y, pad_w0, pad_w1, pad_h0, pad_h1, output_x, output_y, 0,
                               output_xy);
                    else
                    {
                        std::vector<sub_op_task> task_list;
                        std::vector<im2col_param> param_list;

                        auto f = std::bind(&ConvFast::im2col_aider, this, std::placeholders::_1, std::placeholders::_2,
                                           std::placeholders::_3);

                        int steps = output_xy / cpu_number;

                        steps = (steps + 3) & (~0x3);

                        task_list.resize(cpu_number);
                        param_list.resize(cpu_number);

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

                    float* kernel_g = kernel_interleaved + g * (kernel_size * ((output_chan + 3) & -4));
                    float* output_g = output + g * output_xy * output_chan;
                    float* bias_g = biases + g * output_chan;

                    std::vector<sub_op_task> task_list;
                    std::vector<sgemm_param> param_list;

                    int chan_4_num = (output_chan & 0xf) ? 1 : 0;
                    int l2_loop = (output_xy - 1) / col_cnt_l2 + 1;
                    int max_task_num = l2_loop * (chan_16_num + chan_4_num);

                    if(cpu_number > 1)
                        param_list.resize(max_task_num);

                    // for input block of L2 cache size
                    for(int col_i = 0; col_i < output_xy; col_i += col_cnt_l2)
                    {
                        int col_start = col_i;
                        int col_end = col_i + col_cnt_l2;
                        col_end = col_end > output_xy ? output_xy : col_end;

                        if(cpu_number == 1)
                        {
                            sgemm4x16(col, kernel_g, bias_g, have_biases, output_g, kernel_size, col_start, col_end, 0,
                                      output_chan & -16, output_xy, activation, cpu_type);
                            if(output_chan & 0xf)
                                sgemm4x4(col, kernel_g, bias_g, have_biases, output_g, kernel_size, col_start, col_end,
                                         output_chan & -16, output_chan, output_xy, activation, cpu_type);
                        }
                        else
                        {
                            auto f = std::bind(&ConvFast::sgemm_aider, this, std::placeholders::_1,
                                               std::placeholders::_2, std::placeholders::_3);

                            for(int i = 0; i < chan_16_num; i++)
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
                                param->kernel_start = i * 16;
                                param->kernel_end = param->kernel_start + 16;
                                param->output_xy = output_xy;

                                task_list.emplace_back(tmp_task);
                            }

                            if(output_chan & 0xf)
                            {
                                auto f = std::bind(&ConvFast::sgemm4x4_aider, this, std::placeholders::_1,
                                                   std::placeholders::_2, std::placeholders::_3);
                                sub_op_task tmp_task;
                                sgemm_param* param = &param_list[task_list.size()];
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

                    im2col_time += im2col_end - im2col_start;
                    gemm_time += gemm_end - im2col_end;
                }
            }
        }    // end else
    }
    time_log.add_log(im2col_time, gemm_time);

    return true;
}

bool ConvFast::Postrun(Node* node)
{
    if(node->ExistAttr("kernel_interleaved"))
    {
        float* addr;
        addr = any_cast<float*>(node->GetAttr("kernel_interleaved"));
        mem_free(addr);

        node->RemoveAttr("kernel_interleaved");
    }

    if(node->ExistAttr("col_buf_allocated"))
    {
        float* addr = any_cast<float*>(node->GetAttr("col_buf"));
        mem_free(addr);
        node->RemoveAttr("col_buf_allocated");
    }

    if(node->ExistAttr("col_buf"))
        node->RemoveAttr("col_buf");

    activation = false;
    use_direct_k1s1p0 = false;
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
    bool float_enabled = get_auth_float_enabled();

    if(!float_enabled)
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

}    // namespace conv_fast

void RegisterConv2dFast(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Convolution", conv_fast::SelectFunc,
                                                      conv_fast::default_prio))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio[" << conv_fast::default_prio << "]\n";
}

}    // namespace TEngine
