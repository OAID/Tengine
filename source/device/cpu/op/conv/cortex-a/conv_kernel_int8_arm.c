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
 * Author: qwang@openailab.com
 */

#include "conv_kernel_int8_arm.h"

#include "api/c_api.h"
#include "utility/sys_port.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>


#ifdef __aarch64__
void i8gemm_4x16_a72_int8(int* biases, int8_t* input, int8_t* kernel, long kernel_size, int8_t* output,
                                int* multi, long output_xy, int* shift, int activation_min, int activation_max);
void i8gemm_4x4_a72_int8(int* biases, int8_t* input, int8_t* kernel, long kernel_size, int8_t* output,
                                int* multi, long output_xy, int* shift, int activation_min, int activation_max);
void im2col_int8_1x1(int8_t* input, long input_xy, int8_t* col, long col_cnt, long input_chan);
void im2col_int8_3x3(int8_t* input, long input_x, long input_y, long input_chan, int8_t* col, long stride);
// col_start and col_end need to be 16 aligned
// kernel_start need to be 4 aligned
static void i8gemm4x16(int8_t* col, int8_t* kernel, bool bias_term, int* biases, int8_t* output, int* multi,
                       int kernel_size, int output_xy, int col_start, int col_end, int kernel_start, int kernel_end,
                       int activation_min, int activation_max, int* q_shift, int num_thread, int cpu_affinity)
{
    int col_end3 = col_end & 3;
    int kernel_size_aligned2 = (kernel_size + 1) & -2;

#pragma omp parallel for num_threads(num_thread)
    for(int kernel_num = (kernel_start & -16); kernel_num < (kernel_end & -16); kernel_num += 16)
    {
        int* cur_biases = NULL;
        if(bias_term)
        {
            cur_biases = biases + kernel_num;
        }

        int result[64] = {0};
        int8_t* output_line[4];

        int* pmulti = multi + kernel_num;
        int* pq_shift = q_shift + kernel_num;

        int8_t* cur_kernel = kernel + kernel_num * kernel_size_aligned2;
        int8_t* output_result = output + kernel_num * output_xy;

        for(int col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            int8_t* cur_col = col + col_line * kernel_size_aligned2;
            
            i8gemm_4x16_a72_int8(cur_biases, cur_col, cur_kernel, kernel_size_aligned2, output_result + col_line, pmulti,
                              output_xy, pq_shift, activation_min, activation_max);
        }

        if(col_end3)
        {
            int col_line = col_end & -4;
            int8_t* cur_col = col + col_line * kernel_size_aligned2;

            i8gemm_4x16_a72_int8(cur_biases, cur_col, cur_kernel, kernel_size_aligned2, (int8_t*)result, pmulti, 0, pq_shift, activation_min, activation_max);

            for(int i = 0; i < 4; i++)
            {
                for(int j = 0; j < 4; j++)
                {
                    output_line[j] = output + (kernel_num + i * 4 + j) * output_xy + col_line;
                }

                *(output_line[0] + 0) = result[i * 16 + 0];
                *(output_line[1] + 0) = result[i * 16 + 5];
                *(output_line[2] + 0) = result[i * 16 + 10];
                *(output_line[3] + 0) = result[i * 16 + 15];

                if((col_end3) >= 2)
                {
                    *(output_line[0] + 1) = result[i * 16 + 4];
                    *(output_line[1] + 1) = result[i * 16 + 1];
                    *(output_line[2] + 1) = result[i * 16 + 14];
                    *(output_line[3] + 1) = result[i * 16 + 11];
                }
                if((col_end3) == 3)
                {
                    *(output_line[0] + 2) = result[i * 16 + 8];
                    *(output_line[1] + 2) = result[i * 16 + 13];
                    *(output_line[2] + 2) = result[i * 16 + 2];
                    *(output_line[3] + 2) = result[i * 16 + 7];
                }
            }
        }
    }
    return;
}
// col_start and kernel_start need to be 4 aligned
static void i8gemm4x4(int8_t* col, int8_t* kernel, bool bias_term, int* biases, int8_t* output, int* multi,
                      int kernel_size, int output_xy, int col_start, int col_end, int kernel_start, int kernel_end,
                      int activation_min, int activation_max, int* q_shift, int num_thread, int cpu_affinity)
{
    int col_end3 = col_end & 3;
    int kernel_end3 = kernel_end & 3;
    int kernel_size_aligned2 = (kernel_size + 1) & -2;

#pragma omp parallel for num_threads(num_thread)
    for(int kernel_num = kernel_start & -4; kernel_num < (kernel_end & -4); kernel_num += 4)
    {
        int* cur_biases = NULL;
        if(bias_term)
        {
            cur_biases = biases + kernel_num;
        }

        int result[16] = {0};
        int8_t* output_line[4];

        int* pmulti = multi + kernel_num;
        int* pq_shift = q_shift + kernel_num;

        int8_t* cur_kernel = kernel + kernel_num * kernel_size_aligned2;
        int8_t* output_result = output + kernel_num * output_xy;

        for(int col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            int8_t* cur_col = col + col_line * kernel_size_aligned2;

            i8gemm_4x4_a72_int8(cur_biases, cur_col, cur_kernel, kernel_size_aligned2, output_result + col_line, pmulti,
                              output_xy, pq_shift, activation_min, activation_max);
        }
        if(col_end3)
        {
            int col_line = col_end & -4;
            int8_t* cur_col = col + col_line * kernel_size_aligned2;

            i8gemm_4x4_a72_int8(cur_biases, cur_col, cur_kernel, kernel_size_aligned2, (int8_t*)result, pmulti, 0, pq_shift, activation_min, activation_max);

            for(int j = 0; j < 4; j++)
            {
                output_line[j] = output + (kernel_num + j) * output_xy + col_line;
            }

            *(output_line[0] + 0) = result[0];
            *(output_line[1] + 0) = result[5];
            *(output_line[2] + 0) = result[10];
            *(output_line[3] + 0) = result[15];

            if(col_end3 >= 2)
            {
                *(output_line[0] + 1) = result[4];
                *(output_line[1] + 1) = result[1];
                *(output_line[2] + 1) = result[14];
                *(output_line[3] + 1) = result[11];
            }
            if(col_end3 == 3)
            {
                *(output_line[0] + 2) = result[8];
                *(output_line[1] + 2) = result[13];
                *(output_line[2] + 2) = result[2];
                *(output_line[3] + 2) = result[7];
            }
        }
    }
    if(kernel_end3)
    {
        int kernel_num = kernel_end & -4;
        int* cur_biases = NULL;
        if(bias_term)
        {
            cur_biases = biases + kernel_num;
        }

        int result[16] = {0};
        int8_t* output_line[4];

        int* pmulti = multi + kernel_num;
        int* pq_shift = q_shift + kernel_num;
        int8_t* cur_kernel = kernel + kernel_num * kernel_size_aligned2;

        for(int col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            int8_t* cur_col = col + col_line * kernel_size_aligned2;

            i8gemm_4x4_a72_int8(cur_biases, cur_col, cur_kernel, kernel_size_aligned2, (int8_t*)result, pmulti, 0, pq_shift, activation_min, activation_max);

            for(int j = 0; j < 4; j++)
            {
                output_line[j] = output + (kernel_num + j) * output_xy + col_line;
            }

            *(output_line[0] + 0) = result[0];
            *(output_line[0] + 1) = result[4];
            *(output_line[0] + 2) = result[8];
            *(output_line[0] + 3) = result[12];

            if(kernel_end3 >= 2)
            {
                *(output_line[1] + 0) = result[5];
                *(output_line[1] + 1) = result[1];
                *(output_line[1] + 2) = result[13];
                *(output_line[1] + 3) = result[9];
            }
            if(kernel_end3 == 3)
            {
                *(output_line[2] + 0) = result[10];
                *(output_line[2] + 1) = result[14];
                *(output_line[2] + 2) = result[2];
                *(output_line[2] + 3) = result[6];
            }
        }
        if(col_end3)
        {
            int col_line = col_end & -4;
            int8_t* cur_col = col + col_line * kernel_size_aligned2;

            i8gemm_4x4_a72_int8(cur_biases, cur_col, cur_kernel, kernel_size_aligned2, (int8_t*)result, pmulti, 0, pq_shift, activation_min, activation_max);

            for(int j = 0; j < 4; j++)
            {
                output_line[j] = output + (kernel_num + j) * output_xy + col_line;
            }

            *(output_line[0] + 0) = result[0];
            if(col_end3 >= 2)
                *(output_line[0] + 1) = result[4];
            if(col_end3 == 3)
                *(output_line[0] + 2) = result[8];
            if(kernel_end3 >= 2)
            {
                *(output_line[1] + 0) = result[5];
                if(col_end3 >= 2)
                    *(output_line[1] + 1) = result[1];
                if(col_end3 == 3)
                    *(output_line[1] + 2) = result[13];
            }
            if(kernel_end3 == 3)
            {
                *(output_line[2] + 0) = result[10];
                if(col_end3 >= 2)
                    *(output_line[2] + 1) = result[14];
                if(col_end3 == 3)
                    *(output_line[2] + 2) = result[2];
            }
        }
    }
    return;
}
#else
void i8gemm_4x4_a17_int8(int* biases, int8_t* input, int8_t* kernel, int kernel_size, int8_t* output,
                         int* multi, int output_xy, int* shift, int activation_min, int activation_max);
void i8gemm_4x8_a17_int8(int* biases, int8_t* input, int8_t* kernel, int kernel_size, int8_t* output,
                         int* multi, int output_xy, int* shift, int activation_min, int activation_max);

// col_start and col_end need to be 8 aligned kernel_start need to be 4 aligned
static void i8gemm4x8(int8_t* col, int8_t* kernel, bool bias_term, int* biases, int8_t* output, int* multi,
                      int kernel_size, int output_xy, int col_start, int col_end, int kernel_start, int kernel_end,
                      int activation_min, int activation_max, int* q_shift, int num_thread, int cpu_affinity)
{
    int col_end3 = col_end & 3;
    int kernel_size_aligned2 = (kernel_size + 1) & -2;

#pragma omp parallel for num_threads(num_thread)
    for(int kernel_num = (kernel_start & -8); kernel_num < (kernel_end & -8); kernel_num += 8)
    {
        int* cur_biases = NULL;
        if(bias_term)
        {
            cur_biases = biases + kernel_num;
        }

        int result[32] = {0};
        int8_t* output_line[4];

        int* pmulti = multi + kernel_num;
        int* pq_shift = q_shift + kernel_num;

        int8_t* cur_kernel = kernel + kernel_num * kernel_size_aligned2;
        int8_t* output_result = output + kernel_num * output_xy;

        for(int col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            int8_t* cur_col = col + col_line * kernel_size_aligned2;

            i8gemm_4x8_a17_int8(cur_biases, cur_col, cur_kernel, kernel_size_aligned2, output_result + col_line, pmulti,
                               output_xy, pq_shift, activation_min, activation_max);
        }
        if(col_end3)
        {
            int col_line = col_end & -4;
            int8_t* cur_col = col + col_line * kernel_size_aligned2;

            i8gemm_4x8_a17_int8(cur_biases, cur_col, cur_kernel, kernel_size_aligned2, (int8_t*)result, pmulti, 0, pq_shift, activation_min, activation_max);

            for(int i = 0; i < 2; i++)
            {
                for(int j = 0; j < 4; j++)
                {
                    output_line[j] = output + (kernel_num + i * 4 + j) * output_xy + col_line;
                }

                *(output_line[0] + 0) = result[i * 16 + 0];
                *(output_line[1] + 0) = result[i * 16 + 5];
                *(output_line[2] + 0) = result[i * 16 + 10];
                *(output_line[3] + 0) = result[i * 16 + 15];

                if(col_end3 >= 2)
                {
                    *(output_line[0] + 1) = result[i * 16 + 4];
                    *(output_line[1] + 1) = result[i * 16 + 1];
                    *(output_line[2] + 1) = result[i * 16 + 14];
                    *(output_line[3] + 1) = result[i * 16 + 11];
                }
                if(col_end3 == 3)
                {
                    *(output_line[0] + 2) = result[i * 16 + 8];
                    *(output_line[1] + 2) = result[i * 16 + 13];
                    *(output_line[2] + 2) = result[i * 16 + 2];
                    *(output_line[3] + 2) = result[i * 16 + 7];
                }
            }
        }
    }
    return;
}

// col_start and kernel_start need to be 4 aligned
static void i8gemm4x4(int8_t* col, int8_t* kernel, bool bias_term, int* biases, int8_t* output, int* multi,
                      int kernel_size, int output_xy, int col_start, int col_end, int kernel_start, int kernel_end,
                      int activation_min, int activation_max, int* q_shift, int num_thread, int cpu_affinity)
{
    int col_end3 = col_end & 3;
    int kernel_end3 = kernel_end & 3;
    int kernel_size_aligned2 = (kernel_size + 1) & -2;

#pragma omp parallel for num_threads(num_thread)
    for(int kernel_num = (kernel_start & -4); kernel_num < (kernel_end & -4); kernel_num += 4)
    {
        int* cur_biases = NULL;
        if(bias_term)
        {
            cur_biases = biases + kernel_num;
        }

        int result[16] = {0};
        int8_t* output_line[4];

        int* pmulti = multi + kernel_num;
	    int* pq_shift = q_shift + kernel_num;

        int8_t* cur_kernel = kernel + kernel_num * kernel_size_aligned2;
        int8_t* output_result = output + kernel_num * output_xy;

        for(int col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            int8_t* cur_col = col + col_line * kernel_size_aligned2;

            i8gemm_4x4_a17_int8(cur_biases, cur_col, cur_kernel, kernel_size_aligned2, output_result + col_line, pmulti,
                               output_xy, pq_shift, activation_min, activation_max);
        }

        if(col_end3)
        {
            int col_line = col_end & -4;
            int8_t* cur_col = col + col_line * kernel_size_aligned2;

            i8gemm_4x4_a17_int8(cur_biases, cur_col, cur_kernel, kernel_size_aligned2, (int8_t*)result, pmulti, 0, pq_shift, activation_min, activation_max);

            for(int j = 0; j < 4; j++)
            {
                output_line[j] = output + (kernel_num + j) * output_xy + col_line;
            }

            *(output_line[0] + 0) = result[0];
            *(output_line[1] + 0) = result[5];
            *(output_line[2] + 0) = result[10];
            *(output_line[3] + 0) = result[15];

            if(col_end3 >= 2)
            {
                *(output_line[0] + 1) = result[4];
                *(output_line[1] + 1) = result[1];
                *(output_line[2] + 1) = result[14];
                *(output_line[3] + 1) = result[11];
            }
            if(col_end3 == 3)
            {
                *(output_line[0] + 2) = result[8];
                *(output_line[1] + 2) = result[13];
                *(output_line[2] + 2) = result[2];
                *(output_line[3] + 2) = result[7];
            }
        }
    }
    if(kernel_end3)
    {
        int kernel_num = kernel_end & -4;
        int* cur_biases = NULL;
        if(bias_term)
        {
            cur_biases = biases + kernel_num;
        }

        int result[16] = {0};
        int8_t* output_line[4];

        int* pmulti = multi + kernel_num;
        int* pq_shift = q_shift + kernel_num;
        int8_t* cur_kernel = kernel + kernel_num * kernel_size_aligned2;

        for(int col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            int8_t* cur_col = col + col_line * kernel_size_aligned2;

            i8gemm_4x4_a17_int8(cur_biases, cur_col, cur_kernel, kernel_size_aligned2, (int8_t*)result, pmulti, 0, pq_shift, activation_min, activation_max);

            for(int j = 0; j < 4; j++)
            {
                output_line[j] = output + (kernel_num + j) * output_xy + col_line;
            }

            *(output_line[0] + 0) = result[0];
            *(output_line[0] + 1) = result[4];
            *(output_line[0] + 2) = result[8];
            *(output_line[0] + 3) = result[12];

            if(kernel_end3 >= 2)
            {
                *(output_line[1] + 0) = result[5];
                *(output_line[1] + 1) = result[1];
                *(output_line[1] + 2) = result[13];
                *(output_line[1] + 3) = result[9];
            }
            if(kernel_end3 == 3)
            {
                *(output_line[2] + 0) = result[10];
                *(output_line[2] + 1) = result[14];
                *(output_line[2] + 2) = result[2];
                *(output_line[2] + 3) = result[6];
            }
        }
        if(col_end3)
        {
            int col_line = col_end & -4;
            int8_t* cur_col = col + col_line * kernel_size_aligned2;

            i8gemm_4x4_a17_int8(cur_biases, cur_col, cur_kernel, kernel_size_aligned2, (int8_t*)result, pmulti, 0, pq_shift, activation_min, activation_max);

            for(int j = 0; j < 4; j++)
            {
                output_line[j] = output + (kernel_num + j) * output_xy + col_line;
            }

            *(output_line[0] + 0) = result[0];
            if(col_end3 >= 2)
                *(output_line[0] + 1) = result[4];
            if(col_end3 == 3)
                *(output_line[0] + 2) = result[8];
            if(kernel_end3 >= 2)
            {
                *(output_line[1] + 0) = result[5];
                if(col_end3 >= 2)
                    *(output_line[1] + 1) = result[1];
                if(col_end3 == 3)
                    *(output_line[1] + 2) = result[13];
            }
            if(kernel_end3 == 3)
            {
                *(output_line[2] + 0) = result[10];
                if(col_end3 >= 2)
                    *(output_line[2] + 1) = result[14];
                if(col_end3 == 3)
                    *(output_line[2] + 2) = result[2];
            }
        }
    }
    return;
}
#endif
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
int int8_conv_hcl_set_shared_mem(struct conv_priv_info* priv_info, void* mem, int mem_size)
{
    priv_info->external_im2col_mem = 1;
    priv_info->im2col_buffer = mem;
    priv_info->im2col_buffer_size = mem_size;

    return 0;
}

int int8_conv_hcl_set_shared_pack4_mem(struct conv_priv_info* priv_info, void* mem, int mem_size)
{
    priv_info->external_im2col_pack4_mem = 0;
    priv_info->im2col_buffer_pack4 = NULL;
    priv_info->im2col_buffer_pack4_size = 0;

    return 0;
}
int int8_conv_hcl_get_shared_mem_size(struct tensor* input, struct tensor* output, struct conv_param* param)
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
    
    int kernel_size_aligned2 = (kernel_size + 1) & -2;
    int mem_size = elem_size * kernel_size_aligned2 * out_cstep + 128;

    return mem_size;
}

void interleave_kernel_int8(int8_t* kernel, int8_t* kernel_int8, int kernel_chan, int kernel_size)
{
#ifdef __aarch64__
    int8_t* cur_kernel[16];
    int8_t* cur_kernel_int8 = kernel_int8;
    int i, j, k;

    // interleave 16 kernels
    for(i = 0; i < (kernel_chan & -16); i += 16)
    {
        for(j = 0; j < 16; j++)
            cur_kernel[j] = kernel + kernel_size * (i + j);
        for(j = 0; j < (kernel_size & -2); j += 2)
            for(k = 0; k < 16; k++)
            {
                *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                *(cur_kernel_int8++) = *(cur_kernel[k] + j + 1);
            }
        if(kernel_size & 0x1)
            for(k = 0; k < 16; k++)
            {
                *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                *(cur_kernel_int8++) = 0;
            }
    }

    // interleave 4 kernels
    for(i = (kernel_chan & -16); i < (kernel_chan & -4); i += 4)
    {
        for(j = 0; j < 4; j++)
            cur_kernel[j] = kernel + kernel_size * (i + j);
        for(j = 0; j < (kernel_size & -2); j += 2)
            for(k = 0; k < 4; k++)
            {
                *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                *(cur_kernel_int8++) = *(cur_kernel[k] + j + 1);
            }
        if(kernel_size & 0x1)
            for(k = 0; k < 4; k++)
            {
                *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                *(cur_kernel_int8++) = 0;
            }
    }
    // last 4 kernels
    if((kernel_chan & 0x3) != 0)
    {
        for(j = 0; j < 3; j++)
            cur_kernel[j] = kernel + kernel_size * (i + j);
        if((kernel_chan & 0x3) == 3)
        {
            for(j = 0; j < (kernel_size & -2); j += 2)
            {
                for(k = 0; k < 3; k++)
                {
                    *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                    *(cur_kernel_int8++) = *(cur_kernel[k] + j + 1);
                }
                for(k = 0; k < 2; k++)
                    *(cur_kernel_int8++) = 0;
            }
            if(kernel_size & 0x1)
            {
                for(k = 0; k < 3; k++)
                {
                    *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                    *(cur_kernel_int8++) = 0;
                }
                for(k = 0; k < 2; k++)
                    *(cur_kernel_int8++) = 0;
            }
        }
        else if((kernel_chan & 0x3) == 2)
        {
            for(j = 0; j < (kernel_size & -2); j += 2)
            {
                for(k = 0; k < 2; k++)
                {
                    *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                    *(cur_kernel_int8++) = *(cur_kernel[k] + j + 1);
                }
                for(k = 0; k < 4; k++)
                    *(cur_kernel_int8++) = 0;
            }
            if(kernel_size & 0x1)
            {
                for(k = 0; k < 2; k++)
                {
                    *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                    *(cur_kernel_int8++) = 0;
                }
                for(k = 0; k < 4; k++)
                    *(cur_kernel_int8++) = 0;
            }
        }
        else if((kernel_chan & 0x3) == 1)
        {
            for(j = 0; j < (kernel_size & -2); j += 2)
            {
                *(cur_kernel_int8++) = *(cur_kernel[0] + j);
                *(cur_kernel_int8++) = *(cur_kernel[0] + j + 1);
                for(k = 0; k < 6; k++)
                    *(cur_kernel_int8++) = 0;
            }
            if(kernel_size & 0x1)
            {
                *(cur_kernel_int8++) = *(cur_kernel[0] + j);
                for(k = 0; k < 7; k++)
                    *(cur_kernel_int8++) = 0;
            }
        }
    }
#else
    int8_t* cur_kernel[8];
    int8_t* cur_kernel_int8 = kernel_int8;
    int i, j, k;
    int kernel_chan3 = kernel_chan & 0x3;
    int kernel_size1 = kernel_size & 0x1;

    // interleave 8 kernels
    for(i = 0; i < (kernel_chan & -8); i += 8)
    {
        for(j = 0; j < 8; j++)
            cur_kernel[j] = kernel + kernel_size * (i + j);
        for(j = 0; j < (kernel_size & -2); j += 2)
            for(k = 0; k < 8; k++)
            {
                *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                *(cur_kernel_int8++) = *(cur_kernel[k] + j + 1);
            }
        if(kernel_size1)
            for(k = 0; k < 8; k++)
            {
                *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                *(cur_kernel_int8++) = 0;
            }
    }

    // interleave 4 kernels
    for(; i < (kernel_chan & -4); i += 4)
    {
        for(j = 0; j < 4; j++)
            cur_kernel[j] = kernel + kernel_size * (i + j);
        for(j = 0; j < (kernel_size & -2); j += 2)
            for(k = 0; k < 4; k++)
            {
                *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                *(cur_kernel_int8++) = *(cur_kernel[k] + j + 1);
            }
        if(kernel_size1)
            for(k = 0; k < 4; k++)
            {
                *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                *(cur_kernel_int8++) = 0;
            }
    }
    // last 4 kernels
    if(kernel_chan3)
    {
        for(j = 0; j < 3; j++)
            cur_kernel[j] = kernel + kernel_size * (i + j);
        if((kernel_chan3) == 3)
        {
            for(j = 0; j < (kernel_size & -2); j += 2)
            {
                for(k = 0; k < 3; k++)
                {
                    *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                    *(cur_kernel_int8++) = *(cur_kernel[k] + j + 1);
                }
                for(k = 0; k < 2; k++)
                    *(cur_kernel_int8++) = 0;
            }
            if(kernel_size1)
            {
                for(k = 0; k < 3; k++)
                {
                    *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                    *(cur_kernel_int8++) = 0;
                }
                for(k = 0; k < 2; k++)
                    *(cur_kernel_int8++) = 0;
            }
        }
        else if((kernel_chan3) == 2)
        {
            for(j = 0; j < (kernel_size & -2); j += 2)
            {
                for(k = 0; k < 2; k++)
                {
                    *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                    *(cur_kernel_int8++) = *(cur_kernel[k] + j + 1);
                }
                for(k = 0; k < 4; k++)
                    *(cur_kernel_int8++) = 0;
            }
            if(kernel_size1)
            {
                for(k = 0; k < 2; k++)
                {
                    *(cur_kernel_int8++) = *(cur_kernel[k] + j);
                    *(cur_kernel_int8++) = 0;
                }
                for(k = 0; k < 4; k++)
                    *(cur_kernel_int8++) = 0;
            }
        }
        else
        {    // kernel_chan & 0x3 == 1
            for(j = 0; j < (kernel_size & -2); j += 2)
            {
                *(cur_kernel_int8++) = *(cur_kernel[0] + j);
                *(cur_kernel_int8++) = *(cur_kernel[0] + j + 1);
                for(k = 0; k < 6; k++)
                    *(cur_kernel_int8++) = 0;
            }
            if(kernel_size1)
            {
                *(cur_kernel_int8++) = *(cur_kernel[0] + j);
                for(k = 0; k < 7; k++)
                    *(cur_kernel_int8++) = 0;
            }
        }
    }
#endif
    return;
}

/* kernel interleave */
static void interleave_int8(struct tensor* filter, struct conv_priv_info* priv_info, struct conv_param* param)
{
    int group       = param->group;
    int kernel_size = filter->dims[1] * filter->dims[2] * filter->dims[3];
    int out_chan    = filter->dims[0] / group;
    int out_chan_align4 = (out_chan + 3) / 4 * 4;

    int kernel_size_algin = kernel_size * out_chan_align4;
    int kernel_size_group = kernel_size * out_chan;

    int8_t* kernel = filter->data;
    int8_t* interleave_buf = priv_info->interleave_buffer;
    for (int g = 0; g < group; g++)
    {
        int8_t* cur_kernel     = kernel + g * kernel_size_group;
        int8_t* cur_interleave = interleave_buf + g * kernel_size_algin;
        interleave_kernel_int8(cur_kernel, cur_interleave, out_chan, kernel_size);
    }
}


static void im2col_int8(int8_t* im, int8_t* col, int input_chan, int input_x, int input_y, int kernel_x, int kernel_y, int stride_x, int stride_y, int dilation_x,
                   int dilation_y, int pad_x0, int pad_x1, int pad_y0, int pad_y1, int output_x, int output_y, int num_thread)
{
    int col_start = 0;
    int col_end = output_x * output_y;
    int kernel_xy = kernel_x * kernel_y;
    int kernel_size = kernel_xy * input_chan;
    int kernel_size_aligned2 = (kernel_size + 1) & -2;
    int input_xy = input_x * input_y;

    int col_end3 = col_end & 0x3;
    int kernel_size1 = kernel_size & 0x1;
    int is_1x1 = (kernel_x == 1) && (kernel_y == 1) && (stride_x == 1) && (stride_y == 1);
    int is_3x3 = (kernel_x == 3) && (kernel_y == 3) && (dilation_x == 1) && (dilation_y == 1);
    bool is_pad0 = (pad_x0 == 0) && (pad_y0 == 0) && (pad_x1 == 0) && (pad_y1 == 0);

#ifdef __aarch64__
    // is 1x1
    if(is_1x1)
    {
        int8_t* cur_col = col + col_start * kernel_size_aligned2;
        int col_cnt = (col_end & -4) - (col_start & -4);
        im2col_int8_1x1(( int8_t* )im + col_start, input_xy, cur_col, col_cnt, kernel_size);
        cur_col += col_cnt * kernel_size_aligned2;
        int col_i = col_end & -4;
        // final 4 input
        if(col_end3)
        {
            for(int kch = 0; kch < (kernel_size & -2); kch += 2)
            {
                for(int i = 0; i < 4; i++)
                {
                    if((col_i + i) < col_end)
                    {
                        *cur_col++ = *(im + input_xy * (kch + 0) + col_i + i);
                        *cur_col++ = *(im + input_xy * (kch + 1) + col_i + i);
                    }
                    else
                    {
                        *cur_col++ = 0;
                        *cur_col++ = 0;
                    }
                }
            }
            int kch = kernel_size & -2;
            if(kernel_size1)
            {
                for(int i = 0; i < 4; i++)
                {
                    if((col_i + i) < col_end)
                    {
                        *cur_col++ = *(im + input_xy * (kch + 0) + col_i + i);
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
    }
    // 3x3 non dilation
    else if(is_3x3)
    {
#pragma omp parallel for num_threads(num_thread)
        for(int col_i = (col_start & -4); col_i < (col_end & -4); col_i += 4)
        {
            int imx[4] = {0};
            int imy[4] = {0};
            int cnt_x[4] = {0};
            int cnt_y[4] = {0};
            int imx_start[4] = {0};
            int imy_start[4] = {0};
            int8_t* cur_col = col + col_i * kernel_size_aligned2;

            for(int i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_x0;
                imy_start[i] = cnt_y[i] * stride_y - pad_y0;
            }
            if((cnt_y[0] == cnt_y[3]) &&
               (is_pad0 || (cnt_y[0] > 0 && cnt_x[0] > 0 && cnt_y[0] < (output_y - 1) && cnt_x[3] < (output_x - 1))))
            {
                int8_t* input_start = ( int8_t* )(im + imy_start[0] * input_x + imx_start[0]);
                im2col_int8_3x3(input_start, input_x, input_y, input_chan, cur_col, stride_x);
                cur_col += 4 * kernel_size_aligned2;
            }
            else
            {
                bool odd_line = false;
                int kchp = 0;
                int kyp = 0;
                for(int kch = 0; kch < input_chan; kch++)
                {
                    for(int ky = 0; ky < 3; ky++)
                    {
                        if(odd_line)
                        {
                            for(int i = 0; i < 4; i++)
                            {
                                imy[i] = imy_start[i] + kyp;
                                imx[i] = imx_start[i] + 2;
                                if(imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                                else
                                    *cur_col++ = 0;
                                imy[i] = imy_start[i] + ky;
                                if(imx_start[i] >= 0 && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx_start[i]);
                                else
                                    *cur_col++ = 0;
                            }
                            for(int i = 0; i < 4; i++)
                            {
                                for(int k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + 1 + k;
                                    if(imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                        *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                    else
                                        *cur_col++ = 0;
                                }
                            }
                            odd_line = false;
                        }
                        // even line  2n
                        else
                        {
                            for(int i = 0; i < 4; i++)
                                imy[i] = imy_start[i] + ky;
                            for(int i = 0; i < 4; i++)
                            {
                                for(int k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + k;
                                    if(imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                        *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                    else
                                        *cur_col++ = 0;
                                }
                            }
                            kchp = kch;
                            kyp = ky;
                            odd_line = true;
                        }
                    }
                }
                if(kernel_size1)
                {
                    for(int i = 0; i < 4; i++)
                    {
                        imy[i] = imy_start[i] + kyp;
                        imx[i] = imx_start[i] + 2;
                        if(imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                            *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                        else
                            *cur_col++ = 0;
                        *cur_col++ = 0;
                    }
                }
            }
        }
        int col_i = col_end & -4;
        if(col_end3)
        {
            int imx[4] = {0};
            int imy[4] = {0};
            int cnt_x[4] = {0};
            int cnt_y[4] = {0};
            int imx_start[4] = {0};
            int imy_start[4] = {0};
            int8_t* cur_col = col + col_i * kernel_size_aligned2;
            for(int i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_x0;
                imy_start[i] = cnt_y[i] * stride_y - pad_y0;
            }
            bool odd_line = false;
            int kchp = 0;
            int kyp = 0;
            for(int kch = 0; kch < input_chan; kch++)
            {
                for(int ky = 0; ky < 3; ky++)
                {
                    // odd line 1 + 2n
                    if(odd_line)
                    {
                        for(int i = 0; i < 4; i++)
                        {
                            imy[i] = imy_start[i] + kyp;
                            imx[i] = imx_start[i] + 2;
                            if((i < col_end3) && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0;
                            imy[i] = imy_start[i] + ky;
                            if((i < col_end3) && imx_start[i] >= 0 && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx_start[i]);
                            else
                                *cur_col++ = 0;
                        }
                        for(int i = 0; i < 4; i++)
                        {
                            for(int k = 0; k < 2; k++)
                            {
                                imx[i] = imx_start[i] + (1 + k);
                                if((i < col_end3) && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                else
                                    *cur_col++ = 0;
                            }
                        }
                        odd_line = false;
                    }
                    // even line  2n + 1
                    else
                    {
                        for(int i = 0; i < 4; i++)
                            imy[i] = imy_start[i] + ky;
                        for(int i = 0; i < 4; i++)
                        {
                            for(int k = 0; k < 2; k++)
                            {
                                imx[i] = imx_start[i] + k;
                                if(i < col_end3 && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                else
                                    *cur_col++ = 0;
                            }
                        }
                        kchp = kch;
                        kyp = ky;
                        odd_line = true;
                    }
                }
            }
            if(kernel_size1)
            {
                for(int i = 0; i < 4; i++)
                {
                    imy[i] = imy_start[i] + kyp;
                    imx[i] = imx_start[i] + 2;
                    if((i < col_end3) && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                        *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                    else
                        *cur_col++ = 0;
                    *cur_col++ = 0;
                }
            }
        }
    }
    // general case for kernel size <=3
    else if((kernel_x) < 4 && (kernel_y < 4))
    {
        int kch[2], kx[2], ky[2], imx[4][2], imy[4][2];
        int8_t* cur_col = col + col_start * kernel_size_aligned2;
        for(int col_i = (col_start & -4); col_i < (col_end & -4); col_i += 4)
        {
            int cnt_x[4] = {0};
            int cnt_y[4] = {0};
            int imx_start[4] = {0};
            int imy_start[4] = {0};
            for(int i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_x0;
                imy_start[i] = cnt_y[i] * stride_y - pad_y0;
            }
            for(int col_j = 0; col_j < (kernel_size & -2); col_j += 2)
            {
                for(int k = 0; k < 2; k++)
                {
                    kch[k] = (col_j + k) / kernel_xy;
                    ky[k] = (col_j + k - kch[k] * kernel_xy) / kernel_x;
                    kx[k] = (col_j + k - kch[k] * kernel_xy) - ky[k] * kernel_x;
                    ky[k] = ky[k] * dilation_y;
                    kx[k] = kx[k] * dilation_x;
                    for(int i = 0; i < 4; i++)
                    {
                        imx[i][k] = imx_start[i] + kx[k];
                        imy[i][k] = imy_start[i] + ky[k];
                    }
                }
                for(int i = 0; i < 4; i++)
                {
                    for(int k = 0; k < 2; k++)
                    {
                        if(imx[i][k] >= 0 && imx[i][k] < input_x && imy[i][k] >= 0 && imy[i][k] < input_y)
                            *cur_col++ = *(im + input_xy * kch[k] + input_x * imy[i][k] + imx[i][k]);
                        else
                            *cur_col++ = 0;
                    }
                }
            }
            int col_j = kernel_size & -2;
            if(kernel_size1)
            {
                kch[0] = col_j / kernel_xy;
                ky[0] = (col_j - kch[0] * kernel_xy) / kernel_x;
                kx[0] = col_j - kch[0] * kernel_xy - ky[0] * kernel_x;
                ky[0] = ky[0] * dilation_y;
                kx[0] = kx[0] * dilation_x;
                for(int i = 0; i < 4; i++)
                {
                    imx[i][0] = imx_start[i] + kx[0];
                    imy[i][0] = imy_start[i] + ky[0];
                    if(imx[i][0] >= 0 && imx[i][0] < input_x && imy[i][0] >= 0 && imy[i][0] < input_y)
                        *cur_col++ = *(im + input_xy * kch[0] + input_x * imy[i][0] + imx[i][0]);
                    else
                        *cur_col++ = 0;
                    *cur_col++ = 0;
                }
            }
        }
        int col_i = col_end & -4;
        // final 4 input
        if(col_end3)
        {
            int cnt_x[4] = {0};
            int cnt_y[4] = {0};
            int imx_start[4] = {0};
            int imy_start[4] = {0};
            for(int i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_x0;
                imy_start[i] = cnt_y[i] * stride_y - pad_y0;
            }
            for(int col_j = 0; col_j < (kernel_size & -2); col_j += 2)
            {
                for(int k = 0; k < 2; k++)
                {
                    kch[k] = (col_j + k) / kernel_xy;
                    ky[k] = (col_j + k - kch[k] * kernel_xy) / kernel_x;
                    kx[k] = (col_j + k - kch[k] * kernel_xy) - ky[k] * kernel_x;
                    ky[k] = ky[k] * dilation_y;
                    kx[k] = kx[k] * dilation_x;
                    for(int i = 0; i < 4; i++)
                    {
                        imx[i][k] = imx_start[i] + kx[k];
                        imy[i][k] = imy_start[i] + ky[k];
                    }
                }
                for(int i = 0; i < 4; i++)
                {
                    for(int k = 0; k < 2; k++)
                    {
                        if((col_i + i) < col_end && imx[i][k] >= 0 && imx[i][k] < input_x && imy[i][k] >= 0 &&
                           imy[i][k] < input_y)
                            *cur_col++ = *(im + input_xy * kch[k] + input_x * imy[i][k] + imx[i][k]);
                        else
                            *cur_col++ = 0;
                    }
                }
            }
            int col_j = kernel_size & -2;
            if(kernel_size1)
            {
                kch[0] = col_j / kernel_xy;
                ky[0] = (col_j - kch[0] * kernel_xy) / kernel_x;
                kx[0] = col_j - kch[0] * kernel_xy - ky[0] * kernel_x;
                ky[0] = ky[0] * dilation_y;
                kx[0] = kx[0] * dilation_x;
                for(int i = 0; i < 4; i++)
                {
                    imx[i][0] = imx_start[i] + kx[0];
                    imy[i][0] = imy_start[i] + ky[0];
                    if((col_i + i) < col_end && imx[i][0] >= 0 && imx[i][0] < input_x && imy[i][0] >= 0 && imy[i][0] < input_y)
                        *cur_col++ = *(im + input_xy * kch[0] + input_x * imy[i][0] + imx[i][0]);
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
        int kch, kx, ky, kchp, kyp, imx[4], imy[4] = {0};
        int kernel_x1 = kernel_x & 0x1;
        int8_t* cur_col = col + col_start * kernel_size_aligned2;
        for(int col_i = (col_start & -4); col_i < (col_end & -4); col_i += 4)
        {
            int cnt_x[4] = {0};
            int cnt_y[4] = {0};
            int imx_start[4] = {0};
            int imy_start[4] = {0};
            for(int i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_x0;
                imy_start[i] = cnt_y[i] * stride_y - pad_y0;
            }
            bool odd_line = false;
            kchp = 0;
            kyp = 0;
            for(int kch = 0; kch < input_chan; kch++)
            {
                for(ky = 0; ky < kernel_y; ky++)
                {
                    // odd line 2 + 2n
                    if(odd_line)
                    {
                        for(int i = 0; i < 4; i++)
                        {
                            imy[i] = imy_start[i] + kyp * dilation_y;
                            imx[i] = imx_start[i] + (kernel_x - 1) * dilation_x;
                            if(imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0;
                            imy[i] = imy_start[i] + ky * dilation_y;
                            if(imx_start[i] >= 0 && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx_start[i]);
                            else
                                *cur_col++ = 0;
                        }
                        for(kx = 1; kx < kernel_x; kx += 2)
                        {
                            for(int i = 0; i < 4; i++)
                            {
                                for(int k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + (kx + k) * dilation_x;
                                    if(imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                        *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                    else
                                        *cur_col++ = 0;
                                }
                            }
                        }
                        odd_line = false;
                    }
                    // even line  2n
                    else
                    {
                        for(int i = 0; i < 4; i++)
                            imy[i] = imy_start[i] + ky * dilation_y;
                        for(kx = 0; kx < (kernel_x - 1); kx += 2)
                        {
                            for(int i = 0; i < 4; i++)
                            {
                                for(int k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + (kx + k) * dilation_x;
                                    if(imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                        *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                    else
                                        *cur_col++ = 0;
                                }
                            }
                        }
                        kchp = kch;
                        kyp = ky;
                        odd_line = kernel_x1 ? true : false;
                    }
                }
            }
            if(kernel_size1)
            {
                for(int i = 0; i < 4; i++)
                {
                    imy[i] = imy_start[i] + kyp * dilation_y;
                    imx[i] = imx_start[i] + (kernel_x - 1) * dilation_x;
                    if(imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                        *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                    else
                        *cur_col++ = 0;
                    *cur_col++ = 0;
                }
            }
        }
        int col_i = col_end & -4;
        // final 4 input
        if(col_end3)
        {
            int cnt_x[4] = {0};
            int cnt_y[4] = {0};
            int imx_start[4] = {0};
            int imy_start[4] = {0};
            for(int i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_x0;
                imy_start[i] = cnt_y[i] * stride_y - pad_y0;
            }
            bool odd_line = false;
            kchp = 0;
            kyp = 0;
            for(int kch = 0; kch < input_chan; kch++)
            {
                for(ky = 0; ky < kernel_y; ky++)
                {
                    // odd line 1 + 2n
                    if(odd_line)
                    {
                        for(int i = 0; i < 4; i++)
                        {
                            imy[i] = imy_start[i] + kyp * dilation_y;
                            imx[i] = imx_start[i] + (kernel_x - 1) * dilation_x;
                            if((i < col_end3) && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0;
                            imy[i] = imy_start[i] + ky * dilation_y;
                            if((i < col_end3) && imx_start[i] >= 0 && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx_start[i]);
                            else
                                *cur_col++ = 0;
                        }
                        for(kx = 1; kx < kernel_x; kx += 2)
                        {
                            for(int i = 0; i < 4; i++)
                            {
                                for(int k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + (kx + k) * dilation_x;
                                    if((i < col_end3) && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 &&
                                       imy[i] < input_y)
                                        *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                    else
                                        *cur_col++ = 0;
                                }
                            }
                        }
                        odd_line = false;
                    }
                    // even line  2n + 1
                    else
                    {
                        for(int i = 0; i < 4; i++)
                            imy[i] = imy_start[i] + ky * dilation_y;
                        for(kx = 0; kx < (kernel_x - 1); kx += 2)
                        {
                            for(int i = 0; i < 4; i++)
                            {
                                for(int k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + (kx + k) * dilation_x;
                                    if(i < col_end3 && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 &&
                                       imy[i] < input_y)
                                        *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                    else
                                        *cur_col++ = 0;
                                }
                            }
                        }
                        kchp = kch;
                        kyp = ky;
                        odd_line = kernel_x1 ? true : false;
                    }
                }
            }
            if(kernel_size1)
            {
                for(int i = 0; i < 4; i++)
                {
                    imy[i] = imy_start[i] + kyp * dilation_y;
                    imx[i] = imx_start[i] + (kernel_x - 1) * dilation_x;
                    if((i < col_end3) && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                        *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                    else
                        *cur_col++ = 0;
                    *cur_col++ = 0;
                }
            }
        }
    }
#else
    if(is_3x3)
    {
        int stride_x2 = stride_x * 2;
        int stride_x3 = stride_x * 3;
// #pragma omp parallel for num_threads(num_thread)
        for(int col_i = (col_start & -4); col_i < (col_end & -4); col_i += 4)
        {
            int imx[4] = {0};
            int imy[4] = {0};
            int cnt_x[4] = {0};
            int cnt_y[4] = {0};
            int imx_start[4] = {0};
            int imy_start[4] = {0};
            int8_t* cur_col = col + col_i * kernel_size_aligned2;
            for(int i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_x0;
                imy_start[i] = cnt_y[i] * stride_y - pad_y0;
            }
            if((cnt_y[0] == cnt_y[3]) &&
               (is_pad0 || (cnt_y[0] > 0 && cnt_x[0] > 0 && cnt_y[0] < (output_y - 1) && cnt_x[3] < (output_x - 1))))
            {
                int8_t* l00 = ( int8_t* )(im + imy_start[0] * input_x + imx_start[0]);
                int8_t* l01 = l00 + input_x;
                int8_t* l02 = l00 + input_x * 2;
                int8_t* l10 = l00 + input_xy;
                int8_t* l11 = l10 + input_x;
                int8_t* l12 = l10 + input_x * 2;
                for(int kch = 0; kch < (input_chan & -2); kch += 2)
                {
                    cur_col[0] = l00[0];
                    cur_col[1] = l00[1];
                    cur_col[2] = l00[0 + stride_x];
                    cur_col[3] = l00[1 + stride_x];
                    cur_col[4] = l00[0 + stride_x2];
                    cur_col[5] = l00[1 + stride_x2];
                    cur_col[6] = l00[0 + stride_x3];
                    cur_col[7] = l00[1 + stride_x3];
                    cur_col[8] = l00[2];
                    cur_col[9] = l01[0];
                    cur_col[10] = l00[2 + stride_x];
                    cur_col[11] = l01[0 + stride_x];
                    cur_col[12] = l00[2 + stride_x2];
                    cur_col[13] = l01[0 + stride_x2];
                    cur_col[14] = l00[2 + stride_x3];
                    cur_col[15] = l01[0 + stride_x3];
                    cur_col[16] = l01[1];
                    cur_col[17] = l01[2];
                    cur_col[18] = l01[1 + stride_x];
                    cur_col[19] = l01[2 + stride_x];
                    cur_col[20] = l01[1 + stride_x2];
                    cur_col[21] = l01[2 + stride_x2];
                    cur_col[22] = l01[1 + stride_x3];
                    cur_col[23] = l01[2 + stride_x3];
                    cur_col[24] = l02[0];
                    cur_col[25] = l02[1];
                    cur_col[26] = l02[0 + stride_x];
                    cur_col[27] = l02[1 + stride_x];
                    cur_col[28] = l02[0 + stride_x2];
                    cur_col[29] = l02[1 + stride_x2];
                    cur_col[30] = l02[0 + stride_x3];
                    cur_col[31] = l02[1 + stride_x3];
                    cur_col[32] = l02[2];
                    cur_col[33] = l10[0];
                    cur_col[34] = l02[2 + stride_x];
                    cur_col[35] = l10[0 + stride_x];
                    cur_col[36] = l02[2 + stride_x2];
                    cur_col[37] = l10[0 + stride_x2];
                    cur_col[38] = l02[2 + stride_x3];
                    cur_col[39] = l10[0 + stride_x3];
                    cur_col[40] = l10[1];
                    cur_col[41] = l10[2];
                    cur_col[42] = l10[1 + stride_x];
                    cur_col[43] = l10[2 + stride_x];
                    cur_col[44] = l10[1 + stride_x2];
                    cur_col[45] = l10[2 + stride_x2];
                    cur_col[46] = l10[1 + stride_x3];
                    cur_col[47] = l10[2 + stride_x3];
                    cur_col[48] = l11[0];
                    cur_col[49] = l11[1];
                    cur_col[50] = l11[0 + stride_x];
                    cur_col[51] = l11[1 + stride_x];
                    cur_col[52] = l11[0 + stride_x2];
                    cur_col[53] = l11[1 + stride_x2];
                    cur_col[54] = l11[0 + stride_x3];
                    cur_col[55] = l11[1 + stride_x3];
                    cur_col[56] = l11[2];
                    cur_col[57] = l12[0];
                    cur_col[58] = l11[2 + stride_x];
                    cur_col[59] = l12[0 + stride_x];
                    cur_col[60] = l11[2 + stride_x2];
                    cur_col[61] = l12[0 + stride_x2];
                    cur_col[62] = l11[2 + stride_x3];
                    cur_col[63] = l12[0 + stride_x3];
                    cur_col[64] = l12[1];
                    cur_col[65] = l12[2];
                    cur_col[66] = l12[1 + stride_x];
                    cur_col[67] = l12[2 + stride_x];
                    cur_col[68] = l12[1 + stride_x2];
                    cur_col[69] = l12[2 + stride_x2];
                    cur_col[70] = l12[1 + stride_x3];
                    cur_col[71] = l12[2 + stride_x3];
                    cur_col += 72;
                    l00 += input_xy * 2;
                    l01 += input_xy * 2;
                    l02 += input_xy * 2;
                    l10 += input_xy * 2;
                    l11 += input_xy * 2;
                    l12 += input_xy * 2;
                }
                if(input_chan & 0x1)
                {
                    cur_col[0] = l00[0];
                    cur_col[1] = l00[1];
                    cur_col[2] = l00[0 + stride_x];
                    cur_col[3] = l00[1 + stride_x];
                    cur_col[4] = l00[0 + stride_x2];
                    cur_col[5] = l00[1 + stride_x2];
                    cur_col[6] = l00[0 + stride_x3];
                    cur_col[7] = l00[1 + stride_x3];
                    cur_col[8] = l00[2];
                    cur_col[9] = l01[0];
                    cur_col[10] = l00[2 + stride_x];
                    cur_col[11] = l01[0 + stride_x];
                    cur_col[12] = l00[2 + stride_x2];
                    cur_col[13] = l01[0 + stride_x2];
                    cur_col[14] = l00[2 + stride_x3];
                    cur_col[15] = l01[0 + stride_x3];
                    cur_col[16] = l01[1];
                    cur_col[17] = l01[2];
                    cur_col[18] = l01[1 + stride_x];
                    cur_col[19] = l01[2 + stride_x];
                    cur_col[20] = l01[1 + stride_x2];
                    cur_col[21] = l01[2 + stride_x2];
                    cur_col[22] = l01[1 + stride_x3];
                    cur_col[23] = l01[2 + stride_x3];
                    cur_col[24] = l02[0];
                    cur_col[25] = l02[1];
                    cur_col[26] = l02[0 + stride_x];
                    cur_col[27] = l02[1 + stride_x];
                    cur_col[28] = l02[0 + stride_x2];
                    cur_col[29] = l02[1 + stride_x2];
                    cur_col[30] = l02[0 + stride_x3];
                    cur_col[31] = l02[1 + stride_x3];
                    cur_col[32] = l02[2];
                    cur_col[33] = 0;
                    cur_col[34] = l02[2 + stride_x];
                    cur_col[35] = 0;
                    cur_col[36] = l02[2 + stride_x2];
                    cur_col[37] = 0;
                    cur_col[38] = l02[2 + stride_x3];
                    cur_col[39] = 0;
                }
            }
            else
            {
                bool odd_line = false;
                int kchp = 0;
                int kyp = 0;
                for(int kch = 0; kch < input_chan; kch++)
                {
                    for(int ky = 0; ky < 3; ky++)
                    {
                        if(odd_line)
                        {
                            for(int i = 0; i < 4; i++)
                            {
                                imy[i] = imy_start[i] + kyp;
                                imx[i] = imx_start[i] + 2;
                                if(imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                                else
                                    *cur_col++ = 0;
                                imy[i] = imy_start[i] + ky;
                                if(imx_start[i] >= 0 && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx_start[i]);
                                else
                                    *cur_col++ = 0;
                            }
                            for(int i = 0; i < 4; i++)
                            {
                                for(int k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + 1 + k;
                                    if(imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                        *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                    else
                                        *cur_col++ = 0;
                                }
                            }
                            odd_line = false;
                        }
                        // even line  2n
                        else
                        {
                            for(int i = 0; i < 4; i++)
                                imy[i] = imy_start[i] + ky;
                            for(int i = 0; i < 4; i++)
                            {
                                for(int k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + k;
                                    if(imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                        *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                    else
                                        *cur_col++ = 0;
                                }
                            }
                            kchp = kch;
                            kyp = ky;
                            odd_line = true;
                        }
                    }
                }
                if(kernel_size1)
                {
                    for(int i = 0; i < 4; i++)
                    {
                        imy[i] = imy_start[i] + kyp;
                        imx[i] = imx_start[i] + 2;
                        if(imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                            *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                        else
                            *cur_col++ = 0;
                        *cur_col++ = 0;
                    }
                }
            }
        }
        int col_i = col_end & -4;
        if(col_end3)
        {
            int imx[4] = {0};
            int imy[4] = {0};
            int cnt_x[4] = {0};
            int cnt_y[4] = {0};
            int imx_start[4] = {0};
            int imy_start[4] = {0};
            int8_t* cur_col = col + col_i * kernel_size_aligned2;
            for(int i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_x0;
                imy_start[i] = cnt_y[i] * stride_y - pad_y0;
            }
            bool odd_line = false;
            int kchp = 0;
            int kyp = 0;
            for(int kch = 0; kch < input_chan; kch++)
            {
                for(int ky = 0; ky < 3; ky++)
                {
                    // odd line 1 + 2n
                    if(odd_line)
                    {
                        for(int i = 0; i < 4; i++)
                        {
                            imy[i] = imy_start[i] + kyp;
                            imx[i] = imx_start[i] + 2;
                            if((i < col_end3) && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0;
                            imy[i] = imy_start[i] + ky;
                            if((i < col_end3) && imx_start[i] >= 0 && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx_start[i]);
                            else
                                *cur_col++ = 0;
                        }
                        for(int i = 0; i < 4; i++)
                        {
                            for(int k = 0; k < 2; k++)
                            {
                                imx[i] = imx_start[i] + (1 + k);
                                if((i < col_end3) && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                else
                                    *cur_col++ = 0;
                            }
                        }
                        odd_line = false;
                    }
                    // even line  2n + 1
                    else
                    {
                        for(int i = 0; i < 4; i++)
                            imy[i] = imy_start[i] + ky;
                        for(int i = 0; i < 4; i++)
                        {
                            for(int k = 0; k < 2; k++)
                            {
                                imx[i] = imx_start[i] + k;
                                if(i < col_end3 && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                else
                                    *cur_col++ = 0;
                            }
                        }
                        kchp = kch;
                        kyp = ky;
                        odd_line = true;
                    }
                }
            }
            if(kernel_size1)
            {
                for(int i = 0; i < 4; i++)
                {
                    imy[i] = imy_start[i] + kyp;
                    imx[i] = imx_start[i] + 2;
                    if((i < col_end3) && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                        *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                    else
                        *cur_col++ = 0;
                    *cur_col++ = 0;
                }
            }
        }
    }
    // general case for kernel size <=3
    else if((kernel_x) < 4 && (kernel_y < 4))
    {
        int kch[2], kx[2], ky[2], imx[4][2], imy[4][2];
        for(int col_i = (col_start & -4); col_i < (col_end & -4); col_i += 4)
        {
            int cnt_x[4] = {0};
            int cnt_y[4] = {0};
            int imx_start[4] = {0};
            int imy_start[4] = {0};
            int8_t* cur_col = col + col_i * kernel_size_aligned2;
            for(int i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_x0;
                imy_start[i] = cnt_y[i] * stride_y - pad_y0;
            }
            for(int col_j = 0; col_j < (kernel_size & -2); col_j += 2)
            {
                for(int k = 0; k < 2; k++)
                {
                    kch[k] = (col_j + k) / kernel_xy;
                    ky[k] = (col_j + k - kch[k] * kernel_xy) / kernel_x;
                    kx[k] = (col_j + k - kch[k] * kernel_xy) - ky[k] * kernel_x;
                    ky[k] = ky[k] * dilation_y;
                    kx[k] = kx[k] * dilation_x;
                    for(int i = 0; i < 4; i++)
                    {
                        imx[i][k] = imx_start[i] + kx[k];
                        imy[i][k] = imy_start[i] + ky[k];
                    }
                }
                for(int i = 0; i < 4; i++)
                {
                    for(int k = 0; k < 2; k++)
                    {
                        if(imx[i][k] >= 0 && imx[i][k] < input_x && imy[i][k] >= 0 && imy[i][k] < input_y)
                            *cur_col++ = *(im + input_xy * kch[k] + input_x * imy[i][k] + imx[i][k]);
                        else
                            *cur_col++ = 0;
                    }
                }
            }
            int col_j = kernel_size & -2;
            if(kernel_size1)
            {
                kch[0] = col_j / kernel_xy;
                ky[0] = (col_j - kch[0] * kernel_xy) / kernel_x;
                kx[0] = col_j - kch[0] * kernel_xy - ky[0] * kernel_x;
                ky[0] = ky[0] * dilation_y;
                kx[0] = kx[0] * dilation_x;
                for(int i = 0; i < 4; i++)
                {
                    imx[i][0] = imx_start[i] + kx[0];
                    imy[i][0] = imy_start[i] + ky[0];
                    if(imx[i][0] >= 0 && imx[i][0] < input_x && imy[i][0] >= 0 && imy[i][0] < input_y)
                        *cur_col++ = *(im + input_xy * kch[0] + input_x * imy[i][0] + imx[i][0]);
                    else
                        *cur_col++ = 0;
                    *cur_col++ = 0;
                }
            }
        }
        int col_i = col_end & -4;
        // final 4 input
        if(col_end3)
        {
            int cnt_x[4] = {0};
            int cnt_y[4] = {0};
            int imx_start[4] = {0};
            int imy_start[4] = {0};
            int8_t* cur_col = col + col_i * kernel_size_aligned2;
            for(int i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_x0;
                imy_start[i] = cnt_y[i] * stride_y - pad_y0;
            }
            for(int col_j = 0; col_j < (kernel_size & -2); col_j += 2)
            {
                for(int k = 0; k < 2; k++)
                {
                    kch[k] = (col_j + k) / kernel_xy;
                    ky[k] = (col_j + k - kch[k] * kernel_xy) / kernel_x;
                    kx[k] = (col_j + k - kch[k] * kernel_xy) - ky[k] * kernel_x;
                    ky[k] = ky[k] * dilation_y;
                    kx[k] = kx[k] * dilation_x;
                    for(int i = 0; i < 4; i++)
                    {
                        imx[i][k] = imx_start[i] + kx[k];
                        imy[i][k] = imy_start[i] + ky[k];
                    }
                }
                for(int i = 0; i < 4; i++)
                {
                    for(int k = 0; k < 2; k++)
                    {
                        if((col_i + i) < col_end && imx[i][k] >= 0 && imx[i][k] < input_x && imy[i][k] >= 0 &&
                           imy[i][k] < input_y)
                            *cur_col++ = *(im + input_xy * kch[k] + input_x * imy[i][k] + imx[i][k]);
                        else
                            *cur_col++ = 0;
                    }
                }
            }
            int col_j = kernel_size & -2;
            if(kernel_size1)
            {
                kch[0] = col_j / kernel_xy;
                ky[0] = (col_j - kch[0] * kernel_xy) / kernel_x;
                kx[0] = col_j - kch[0] * kernel_xy - ky[0] * kernel_x;
                ky[0] = ky[0] * dilation_y;
                kx[0] = kx[0] * dilation_x;
                for(int i = 0; i < 4; i++)
                {
                    imx[i][0] = imx_start[i] + kx[0];
                    imy[i][0] = imy_start[i] + ky[0];
                    if((col_i + i) < col_end && imx[i][0] >= 0 && imx[i][0] < input_x && imy[i][0] >= 0 &&
                       imy[i][0] < input_y)
                        *cur_col++ = *(im + input_xy * kch[0] + input_x * imy[i][0] + imx[i][0]);
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
        int kernel_x1 = kernel_x & 0x1;
        int8_t* cur_col = col + col_start * kernel_size_aligned2;
        for(int col_i = (col_start & -4); col_i < (col_end & -4); col_i += 4)
        {
            int cnt_x[4] = {0};
            int cnt_y[4] = {0};
            int imx_start[4] = {0};
            int imy_start[4] = {0};
            for(int i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_x0;
                imy_start[i] = cnt_y[i] * stride_y - pad_y0;
            }
            bool odd_line = false;
            kchp = 0;
            kyp = 0;
            for(int kch = 0; kch < input_chan; kch++)
            {
                for(int ky = 0; ky < kernel_y; ky++)
                {
                    // odd line 2 + 2n
                    if(odd_line)
                    {
                        for(int i = 0; i < 4; i++)
                        {
                            imy[i] = imy_start[i] + kyp * dilation_y;
                            imx[i] = imx_start[i] + (kernel_x - 1) * dilation_x;
                            if(imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0;
                            imy[i] = imy_start[i] + ky * dilation_y;
                            if(imx_start[i] >= 0 && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx_start[i]);
                            else
                                *cur_col++ = 0;
                        }
                        for(int kx = 1; kx < kernel_x; kx += 2)
                        {
                            for(int i = 0; i < 4; i++)
                            {
                                for(int k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + (kx + k) * dilation_x;
                                    if(imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                        *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                    else
                                        *cur_col++ = 0;
                                }
                            }
                        }
                        odd_line = false;
                    }
                    // even line  2n
                    else
                    {
                        for(int i = 0; i < 4; i++)
                            imy[i] = imy_start[i] + ky * dilation_y;
                        for(int kx = 0; kx < (kernel_x - 1); kx += 2)
                        {
                            for(int i = 0; i < 4; i++)
                            {
                                for(int k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + (kx + k) * dilation_x;
                                    if(imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                        *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                    else
                                        *cur_col++ = 0;
                                }
                            }
                        }
                        kchp = kch;
                        kyp = ky;
                        odd_line = kernel_x1 ? true : false;
                    }
                }
            }
            if(kernel_size1)
            {
                for(int i = 0; i < 4; i++)
                {
                    imy[i] = imy_start[i] + kyp * dilation_y;
                    imx[i] = imx_start[i] + (kernel_x - 1) * dilation_x;
                    if(imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                        *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                    else
                        *cur_col++ = 0;
                    *cur_col++ = 0;
                }
            }
        }
        int col_i = col_end & -4;
        // final 4 input
        if(col_end3)
        {
            int cnt_x[4] = {0};
            int cnt_y[4] = {0};
            int imx_start[4] = {0};
            int imy_start[4] = {0};
            for(int i = 0; i < 4; i++)
            {
                cnt_y[i] = (col_i + i) / output_x;
                cnt_x[i] = col_i + i - cnt_y[i] * output_x;
                imx_start[i] = cnt_x[i] * stride_x - pad_x0;
                imy_start[i] = cnt_y[i] * stride_y - pad_y0;
            }
            bool odd_line = false;
            kchp = 0;
            kyp = 0;
            for(int kch = 0; kch < input_chan; kch++)
            {
                for(int ky = 0; ky < kernel_y; ky++)
                {
                    // odd line 1 + 2n
                    if(odd_line)
                    {
                        for(int i = 0; i < 4; i++)
                        {
                            imy[i] = imy_start[i] + kyp * dilation_y;
                            imx[i] = imx_start[i] + (kernel_x - 1) * dilation_x;
                            if((i < col_end3) && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0;
                            imy[i] = imy_start[i] + ky * dilation_y;
                            if((i < col_end3) && imx_start[i] >= 0 && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx_start[i]);
                            else
                                *cur_col++ = 0;
                        }
                        for(int kx = 1; kx < kernel_x; kx += 2)
                        {
                            for(int i = 0; i < 4; i++)
                            {
                                for(int k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + (kx + k) * dilation_x;
                                    if((i < col_end3) && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 &&
                                       imy[i] < input_y)
                                        *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                    else
                                        *cur_col++ = 0;
                                }
                            }
                        }
                        odd_line = false;
                    }
                    // even line  2n + 1
                    else
                    {
                        for(int i = 0; i < 4; i++)
                        {
                            imy[i] = imy_start[i] + ky * dilation_y;
                        }
                        for(int kx = 0; kx < (kernel_x - 1); kx += 2)
                        {
                            for(int i = 0; i < 4; i++)
                            {
                                for(int k = 0; k < 2; k++)
                                {
                                    imx[i] = imx_start[i] + (kx + k) * dilation_x;
                                    if(i < col_end3 && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 &&
                                       imy[i] < input_y)
                                        *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                    else
                                        *cur_col++ = 0;
                                }
                            }
                        }
                        kchp = kch;
                        kyp = ky;
                        odd_line = kernel_x1 ? true : false;
                    }
                }
            }
            if(kernel_size1)
            {
                for(int i = 0; i < 4; i++)
                {
                    imy[i] = imy_start[i] + kyp * dilation_y;
                    imx[i] = imx_start[i] + (kernel_x - 1) * dilation_x;
                    if((i < col_end3) && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                        *cur_col++ = *(im + input_xy * kchp + input_x * imy[i] + imx[i]);
                    else
                        *cur_col++ = 0;
                    *cur_col++ = 0;
                }
            }
        }
    }
#endif
    return;
}


int int8_conv_hcl_prerun(struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* output_tensor,
                    struct conv_priv_info* priv_info, struct conv_param* param)
{
    int in_c = input_tensor->dims[1];
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];

    int out_c = output_tensor->dims[1];
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    /* alloc mem of im2col  */
    if (!priv_info->external_im2col_mem)
    {
        int mem_size = int8_conv_hcl_get_shared_mem_size(input_tensor, output_tensor, param);
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
    interleave_int8(filter_tensor, priv_info, param);

    priv_info->multi = (int*)sys_malloc(out_c * sizeof(int));
    priv_info->q_shift = (int*)sys_malloc(out_c * sizeof(int));

    float input_scale = input_tensor->scale;
    float* kernel_scales = filter_tensor->scale_list;
    float output_scale = output_tensor->scale;

    priv_info->activation_min = -127;
    priv_info->activation_max = 127;
    /*  set activation   */
    if(param->activation >= 0)
    {
        priv_info->activation_min = 0;
        if(param->activation == 1)
            priv_info->activation_max = round(1.0 / output_scale);
        if(param->activation == 6)
            priv_info->activation_max = round(6.0 / output_scale);

        if(priv_info->activation_max > 127)
            priv_info->activation_max = 127;
    }

    for(int i=0; i<out_c; i++)
    {
        float kernel_scale = kernel_scales[i];
        float scale = input_scale * kernel_scale / output_scale;

        int shift;
        float q = frexp(scale, &shift);
        int fix_q = round(q * (1ll << 31));
        // TLOG_ERR("prerun: %f,%lld,%d,%d, %lld\n",q, fix_q, multi, q_shift, 1ll<<31);
        if(fix_q == (1l << 31))
        {
            fix_q /= 2;
            shift++;
        }

        priv_info->multi[i] = (int)fix_q;
        priv_info->q_shift[i] = (int)shift;
    }
    return 0;
}

int int8_conv_hcl_postrun(struct conv_priv_info* priv_info)
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
    if (priv_info->multi)
    {
        sys_free(priv_info->multi);
        priv_info->multi = NULL;
    }
    if (priv_info->q_shift)
    {
        sys_free(priv_info->q_shift);
        priv_info->q_shift = NULL;
    }

    return 0;
}

int int8_conv_hcl_run(struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* bias_tensor,
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

    int out_c = output_tensor->dims[1] / group;
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int out_hw = out_h * out_w;
    int output_size = out_c * out_h * out_w;
    int out_c_align = ((out_c + 3) & -4);
    int output_image_size = output_tensor->dims[1] * output_tensor->dims[2] * output_tensor->dims[3];

    int activation_min = priv_info->activation_min;
    int activation_max = priv_info->activation_max;

    /* buffer addr */
    int8_t* input_buf = ( int8_t* )input_tensor->data;
    int8_t* output_buf = ( int8_t* )output_tensor->data;
    int32_t* biases_buf = NULL;
    bool have_biases = false;
    if (bias_tensor != NULL)
    {
        biases_buf = (int32_t*)bias_tensor->data;
        have_biases = true;
    }

    int8_t* col_buf = ( int8_t* )priv_info->im2col_buffer;
    int8_t* interleave_buf = ( int8_t* )priv_info->interleave_buffer;

    /* block size split parameter */
    int L2_CACHE_SIZE = (cpu_affinity == TENGINE_CLUSTER_LITTLE)? 512 * 1024 : 1024 * 1024;
    int kernel_size_l1 = kernel_size;
#ifdef __aarch64__
    int col_cnt_l2 = L2_CACHE_SIZE * 3 / kernel_size_l1 / 4;
#else
    int col_cnt_l2 = L2_CACHE_SIZE / 4 / kernel_size_l1 * 3 / 4;
#endif
    col_cnt_l2 = col_cnt_l2 > 4 ? (col_cnt_l2 & -4) : 4;

    for (int n = 0; n < batch; n++)    // batch size
    {
        int8_t* input = input_buf + n * input_size * group;
        int8_t* output = output_buf + n * output_size * group;
        for (int g = 0; g < group; g++)
        {
            int8_t* cur_input = input + g * input_size;

            im2col_int8(cur_input, col_buf, in_c, in_w, in_h, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h,
                   pad_w0, pad_w1, pad_h0, pad_h1, out_w, out_h, num_thread);

            int kernel_size_aligned2 = (kernel_size + 1) & -2;
            int output_chan_aligned4 = (out_c + 3) & -4;

            int8_t* kernel_g = interleave_buf + g * kernel_size_aligned2 * output_chan_aligned4;
            int8_t* output_g = output + g * output_size;
            int* bias_g = have_biases ? (biases_buf + g * out_c) : NULL;
            int* multi_g = priv_info->multi + g * out_c;
            int* q_shift_g = priv_info->q_shift + g * out_c;

            // for input block of L2 cache size
            for(int col_i = 0; col_i < out_hw; col_i += col_cnt_l2)
            {
                int col_start = col_i;
                int col_end = col_i + col_cnt_l2;
                col_end = col_end > out_hw ? out_hw : col_end;
#ifdef __aarch64__
                i8gemm4x16(col_buf, kernel_g, have_biases, bias_g, output_g, multi_g, kernel_size, out_hw,
                            col_start, col_end, 0, out_c & -16, activation_min, activation_max, q_shift_g, num_thread, cpu_affinity);
                if(out_c & 0xf)
                    i8gemm4x4(col_buf, kernel_g, have_biases, bias_g, output_g, multi_g, kernel_size, out_hw,
                                col_start, col_end, out_c & -16, out_c, activation_min, activation_max, q_shift_g, num_thread, cpu_affinity);
#else
                i8gemm4x8(col_buf, kernel_g, have_biases, bias_g, output_g, multi_g, kernel_size, out_hw,
                            col_start, col_end, 0, out_c & -8, activation_min, activation_max, q_shift_g, num_thread, cpu_affinity);
                if(out_c & 0x7)
                    i8gemm4x4(col_buf, kernel_g, have_biases, bias_g, output_g, multi_g, kernel_size, out_hw,
                                col_start, col_end, out_c & -8, out_c, activation_min, activation_max, q_shift_g, num_thread, cpu_affinity);
#endif
            }    // col_cont
        }
    }
    return 0;
}
