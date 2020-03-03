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
 * Author: haoluo@openailab.com
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
#include "op_utils.hpp"
#include <math.h>

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif


extern "C" void im2col_fp32_1x1(float* input, long input_xy, float* col, long col_cnt, long input_chan);
extern "C" void im2col_fp32_3x3(float* input, long input_x, long input_y, long input_chan, float* col, long stride);

#include "conv_2d_fp32_nhwc_kernel/A72.inl"

namespace TEngine {

namespace conv_fast_nhwc {

#define TYPE_A53 0
#define TYPE_A72 1
const char* conv_name = "CONV_FAST_NHWC";
const int default_prio = 5500;

void im2col(float* im, float* col, int input_chan, int input_x, int input_y, int kernel_x, int kernel_y, int stride_x,
            int stride_y, int dilation_x, int dilation_y, int pad_w0, int pad_w1, int pad_h0, int pad_h1, int output_x,
            int output_y, int col_start, int col_end)
{
    int kernel_size = kernel_x * kernel_y * input_chan;
    // int input_xy = input_x * input_y;
    int pad_x = pad_w0;
    int pad_y = pad_h0;
    float* cur_col = col + col_start * kernel_size;
    bool is_1x1 = (kernel_x == 1) && (kernel_y == 1) && (stride_x == 1) && (stride_y == 1);
    // bool is_dilation = (dilation_x != 1) || (dilation_y != 1);
    // bool is_3x3 = (kernel_x == 3) && (kernel_y == 3) && (!is_dilation);
    int col_i, col_j, kch, ky, kx, i;
    int col_end3 = col_end & 0x3;

    if(is_1x1)
    {
#if 1
        // equivlent C code
        for(col_i = (col_start & -4); col_i < (col_end & -4); col_i += 4)
        {
            for(col_j = 0; col_j < input_chan; col_j++)
            {
                for(i = 0; i < 4; i++)
                    *cur_col++ = *(im + (i + col_i) * input_chan + col_j);
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
                        *cur_col++ = *(im + (col_i + i) * input_chan + col_j);
                    else
                        *cur_col++ = 0.0;
                }
        }
    } /*
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
                                 int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] +
     kx};
                                 int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] +
     ky};
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
                 int cnt_y[4] = {col_i / output_x, (col_i + 1) / output_x, (col_i + 2) / output_x, (col_i + 3) /
     output_x};
                 int cnt_x[4] = {col_i - cnt_y[0] * output_x, col_i - cnt_y[1] * output_x + 1,
                                 col_i - cnt_y[2] * output_x + 2, col_i - cnt_y[3] * output_x + 3};
                 int imx_start[4] = {cnt_x[0] * stride_x - pad_x, cnt_x[1] * stride_x - pad_x, cnt_x[2] * stride_x -
     pad_x,
                                     cnt_x[3] * stride_x - pad_x};
                 int imy_start[4] = {cnt_y[0] * stride_y - pad_y, cnt_y[1] * stride_y - pad_y, cnt_y[2] * stride_y -
     pad_y,
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
         }*/
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
            for(ky = 0; ky < (kernel_y * dilation_y); ky += dilation_y)
                for(kx = 0; kx < (kernel_x * dilation_x); kx += dilation_x)
                    for(kch = 0; kch < input_chan; kch++)
                    {
                        int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                        int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                        for(i = 0; i < 4; i++)
                        {
                            if(imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + imy[i] * input_x * input_chan + imx[i] * input_chan + kch);
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
            for(ky = 0; ky < (kernel_y * dilation_y); ky += dilation_y)
                for(kx = 0; kx < (kernel_x * dilation_x); kx += dilation_x)
                    for(kch = 0; kch < input_chan; kch++)
                    {
                        int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                        int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                        for(i = 0; i < 4; i++)
                        {
                            if((col_i + i) < col_end && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 &&
                               imy[i] < input_y)
                                *cur_col++ = *(im + imy[i] * input_x * input_chan + imx[i] * input_chan + kch);
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
    int i, j, k;
    float* cur_kernel[16];
    float* cur_kernel_interleaved = kernel_interleaved;

    // interleave 16 kernels
    for(i = 0; i < (kernel_chan & -16); i += 16)
    {
        for(k = 0; k < 16; k++)
            cur_kernel[k] = kernel + kernel_size * (i + k);
        for(j = 0; j < kernel_size; j++)
        {
            for(k = 0; k < 16; k++)
                *(cur_kernel_interleaved++) = cur_kernel[k][j];
        }
    }

    for(i = (kernel_chan & -16); i < (kernel_chan & -4); i += 4)
    {
        for(k = 0; k < 4; k++)
            cur_kernel[k] = kernel + kernel_size * (i + k);
        for(j = 0; j < kernel_size; j++)
        {
            for(k = 0; k < 4; k++)
                *(cur_kernel_interleaved++) = cur_kernel[k][j];
        }
    }
    // last 4 kernel
    for(k = 0; k < 3; k++)
        cur_kernel[k] = kernel + kernel_size * (i + k);
    if((kernel_chan & 0x3) == 3)
    {
        for(j = 0; j < kernel_size; j++)
        {
            for(k = 0; k < 3; k++)
                *(cur_kernel_interleaved++) = cur_kernel[k][j];
            *(cur_kernel_interleaved++) = 0.0;
        }
    }
    else if((kernel_chan & 0x3) == 2)
    {
        for(j = 0; j < kernel_size; j++)
        {
            for(k = 0; k < 2; k++)
                *(cur_kernel_interleaved++) = cur_kernel[k][j];
            *(cur_kernel_interleaved++) = 0.0;
            *(cur_kernel_interleaved++) = 0.0;
        }
    }
    else if((kernel_chan & 0x3) == 1)
    {
        for(j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel[0][j];
            *(cur_kernel_interleaved++) = 0.0;
            *(cur_kernel_interleaved++) = 0.0;
            *(cur_kernel_interleaved++) = 0.0;
        }
    }

    return;
}

static void sgemm4x16(float* col, float* kernel, float* biases, bool bias_term, float* output, int kernel_size,
                      int col_start, int col_end, int kernel_start, int kernel_end, int output_chan, int activation,
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
        cur_output = ( float* )(output + kernel_num);
        for(col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            float* cur_output_16 = cur_output + col_line * output_chan;
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x16_a72(cur_biases, cur_col, cur_kernel, kernel_size, cur_output_16, output_chan, activation, 1);
        }
        if(col_end3)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            float* cur_output_16 = cur_output + col_line * output_chan;
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x16_a72(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 1);
            for(i = 0; i < 16; i++)
                for(j = 0; j < (col_end3); j++)
                    *(cur_output_16 + j * output_chan + i) = result[i + j * 16];
        }
    }

    return;
}

static void sgemm4x4(float* col, float* kernel, float* biases, bool bias_term, float* output, int kernel_size,
                     int col_start, int col_end, int kernel_start, int kernel_end, int output_chan, int activation,
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
        cur_output = ( float* )(output + kernel_num);
        for(col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            float* cur_output_4 = cur_output + col_line * output_chan;
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, cur_output_4, output_chan, activation, 1);
        }
        if(col_end3)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            float* cur_output_4 = cur_output + col_line * output_chan;
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 1);

            for(i = 0; i < 4; i++)
            {
                for(j = 0; j < (col_end3); j++)
                    *(cur_output_4 + j * output_chan + i) = result[i + j * 4];
            }
        }
    }
    if(kernel_end3)
    {
        if(bias_term)
            cur_biases = ( float* )(biases + kernel_num);
        cur_kernel = ( float* )(kernel + kernel_num * kernel_size);
        cur_output = ( float* )(output + kernel_num);
        for(col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            float* cur_output_4 = cur_output + col_line * output_chan;
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 1);

            for(i = 0; i < kernel_end3; i++)
                for(j = 0; j < 4; j++)
                    *(cur_output_4 + j * output_chan + i) = result[i + j * 4];
        }
        if(col_end3)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            float* cur_output_4 = cur_output + col_line * output_chan;
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 1);
            for(i = 0; i < (kernel_end3); i++)
            {
                for(j = 0; j < (col_end3); j++)
                    *(cur_output_4 + j * output_chan + i) = result[i + j * 4];
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

struct ConvFastNHWC : public MTNodeOps
{
    ConvFastNHWC()
    {
        name_ = "arm_gemm_conv_fp32_nhwc";
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

    int activation;

    bool dynamic_shape;
};

bool ConvFastNHWC::im2col_aider(int cpu, int seq, void* data)
{
    im2col_param* param = ( im2col_param* )(data);
    im2col(param->im, param->col, param->input_chan, param->input_x, param->input_y, param->kernel_x, param->kernel_y,
           param->stride_x, param->stride_y, param->dilation_x, param->dilation_y, param->pad_w0, param->pad_w1,
           param->pad_h0, param->pad_h1, param->output_x, param->output_y, param->col_start, param->col_end);

    return true;
}

bool ConvFastNHWC::sgemm4x4_aider(int cpu, int seq, void* data)
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

bool ConvFastNHWC::sgemm_aider(int cpu, int seq, void* data)
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

bool ConvFastNHWC::Prerun(Node* node)
{
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();
    int group = param->group;

    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& output_shape = output_tensor->GetShape();
    int output_chan = output_shape.GetC() / group;
    // int output_y = output_shape.GetH();
    // int output_x = output_shape.GetW();

    /* pre-allocate col_buf */
    Tensor* input_tensor = node->GetInputTensor(0);
    TShape& input_shape = input_tensor->GetShape();

    int input_chan = input_shape.GetC() / group;
    int kernel_size = input_chan * param->kernel_h * param->kernel_w;

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

bool ConvFastNHWC::Reshape(Node* node)
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

bool ConvFastNHWC::Run(Node* node)
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

    /* one image per time */
    for(int i = 0; i < output_n; i++)
    {
        float* input = input_org + i * input_size * group;
        float* output = output_org + i * output_xy * output_chan * group;

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

                        auto f = std::bind(&ConvFastNHWC::im2col_aider, this, std::placeholders::_1,
                                           std::placeholders::_2, std::placeholders::_3);

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
                                      output_chan & -16, output_chan, activation, cpu_type);
                            if(output_chan & 0xf)
                                sgemm4x4(col, kernel_g, bias_g, have_biases, output_g, kernel_size, col_start, col_end,
                                         output_chan & -16, output_chan, output_chan, activation, cpu_type);
                        }
                        else
                        {
                            auto f = std::bind(&ConvFastNHWC::sgemm_aider, this, std::placeholders::_1,
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
                                param->output_xy = output_chan;

                                task_list.emplace_back(tmp_task);
                            }

                            if(output_chan & 0xf)
                            {
                                auto f = std::bind(&ConvFastNHWC::sgemm4x4_aider, this, std::placeholders::_1,
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
                                param->output_xy = output_chan;

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
        }    // end else
    }

    return true;
}

bool ConvFastNHWC::Postrun(Node* node)
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
    return true;
}

bool ConvFastNHWC::GetSharedMemorySize(Node* node, unsigned int& mem_size)
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

bool ConvFastNHWC::SetSharedMemoryAddr(Node* node, void* mem_addr, int mem_size)
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
#endif

    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NHWC)
        return nullptr;

    ConvFastNHWC* ops = new ConvFastNHWC();

    if(node->IsDynamicShape())
        ops->dynamic_shape = true;
    else
        ops->dynamic_shape = false;

    Operator* op = node->GetOp();
    Convolution* conv_op = dynamic_cast<Convolution*>(op);
    ConvParam* param = conv_op->GetParam();
    if(param->group > 1)
        return nullptr;
    ops->activation = param->activation;

    return ops;
}

}    // namespace conv_fast_nhwc

void RegisterConv2dFp32NHWC(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Convolution", conv_fast_nhwc::SelectFunc,
                                                      conv_fast_nhwc::default_prio))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio[" << conv_fast_nhwc::default_prio << "]\n";
}

}    // namespace TEngine
