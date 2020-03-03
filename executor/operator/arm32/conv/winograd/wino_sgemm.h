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
 * Copyright (c) 2019, Open AI Lab
 * Author: chunyinglv@openailab.com
 */

#ifndef __WINO_SGEMM_H__
#define __WINO_SGEMM_H__

#include "wino_config.h"
#include "wino_trans_out.h"
#ifdef __cplusplus
extern "C" {
#endif    //__cplusplus

extern void wino_sgemm_4x12_A17(float* output, float* input, float* kernel, long cin);
extern void wino_sgemm_1x12_A17(float* output, float* input, float* kernel, long cin);
extern void wino_sgemm_4x4_A17(float* output, float* input, float* kernel, long cin);

typedef void (*sgemm_kernel_t)(float* output, float* input, float* kernel, long cin);

// pour debug
static inline void wino_sgemm_4x12_cpu(float* output, float* input, float* kernel, long cin)
{
    for(int i = 0; i < KER_COUT_UNIT; i++)
    {
        for(int j = 0; j < BLOCK_HW_UNIT; j++)
        {
            float sum = 0;
            for(int k = 0; k < cin; k++)
            {
                sum += input[k * BLOCK_HW_UNIT + j] * kernel[k * KER_COUT_UNIT + i];
            }
            output[i * BLOCK_HW_UNIT + j] = sum;
        }
    }
}
static inline void wino_sgemm_4x4_cpu(float* output, float* input, float* kernel, long cin)
{
    for(int i = 0; i < KER_COUT_UNIT4; i++)
    {
        for(int j = 0; j < BLOCK_HW_UNIT; j++)
        {
            float sum = 0;
            for(int k = 0; k < cin; k++)
            {
                sum += input[k * BLOCK_HW_UNIT + j] * kernel[k * KER_COUT_UNIT4 + i];
            }
            output[i * BLOCK_HW_UNIT + j] = sum;
        }
    }
}
static inline void wino_sgemm_1x12_cpu(float* output, float* input, float* kernel, long cin)
{
    for(int i = 0; i < KER_COUT_UNIT; i++)
    {
        float sum = 0;
        for(int k = 0; k < cin; k++)
        {
            sum += input[k] * kernel[k * KER_COUT_UNIT + i];
        }
        output[i] = sum;
    }
}
static inline void wino_sgemm_1x4_cpu(float* output, float* input, float* kernel, long cin)
{
    for(int i = 0; i < KER_COUT_UNIT4; i++)
    {
        float sum = 0;
        for(int k = 0; k < cin; k++)
        {
            sum += input[k] * kernel[k * KER_COUT_UNIT4 + i];
        }
        output[i] = sum;
    }
}

static inline void wino_sgemm_4x12(float* ker, float* inp, float* output, const float* bias, int bias_term, int cin,
                                   sgemm_kernel_t kernel_func,sgemm_kernel_t kernel_func1, int cout_start, int cout_end, int block_start, int block_end,
                                   int block_h, int block_w, int out_hw, int out_w, int resi_h, int resi_w,
                                   int activation)
{
    int p, i;

    float* ker_ptr;
    float* inp_ptr;

    for(p = cout_start; p < cout_end; p += KER_COUT_UNIT)
    {
        ker_ptr = ker + p * ELEM_SIZE * cin;

        for(i = (block_start & -4); i < (block_end & -4); i += 4)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            float out_buffer[KER_COUT_UNIT * BLOCK_HW_UNIT * ELEM_SIZE];
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                kernel_func(out_buffer + s * BLOCK_HW_UNIT * KER_COUT_UNIT,
                                       inp_ptr + s * BLOCK_HW_UNIT * cin, ker_ptr + s * KER_COUT_UNIT * cin, cin);
            }

            // interleave
            float buffer[KER_COUT_UNIT * BLOCK_HW_UNIT * ELEM_SIZE];
            float* buffer_ptr0 = buffer;
            for(int pp = 0; pp < KER_COUT_UNIT; pp++)
            {
                for(int t = 0; t < BLOCK_HW_UNIT; t++)
                {
                    for(int ss = 0; ss < ELEM_SIZE; ss++)
                    {
                        *buffer_ptr0 = out_buffer[ss * BLOCK_HW_UNIT * KER_COUT_UNIT + pp * BLOCK_HW_UNIT + t];
                        buffer_ptr0++;
                    }
                }
            }
            // end interleave
            transform_output_f43_4tile(buffer, output, p, i, block_h, block_w, out_hw, out_w, resi_h, resi_w,
                                       KER_COUT_UNIT, bias, bias_term, activation);
            // end transform
        }

        for(; i < block_end; i++)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            float out_buffer[KER_COUT_UNIT * ELEM_SIZE];
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                kernel_func1(out_buffer + s * KER_COUT_UNIT, inp_ptr + s * cin,
                                       ker_ptr + s * KER_COUT_UNIT * cin, cin);
    
            }
            // interleave
            float buffer[KER_COUT_UNIT * ELEM_SIZE];
            float* buffer_ptr0 = buffer;

            for(int pp = 0; pp < KER_COUT_UNIT; pp++)
            {
                for(int ss = 0; ss < ELEM_SIZE; ss++)
                {
                    *buffer_ptr0 = out_buffer[ss * KER_COUT_UNIT + pp];
                    buffer_ptr0++;
                }
            }
            // end interleave
            transform_output_f43_1tile(buffer, output, p, i, block_h, block_w, out_hw, out_w, resi_h, resi_w,
                                       KER_COUT_UNIT, bias, bias_term, activation);
            // end transform
        }
    }
}
static inline void wino_sgemm_4x4(float* ker, float* inp, float* output, const float* bias, int bias_term, int cin,
                                  sgemm_kernel_t kernel_func, int cout_start, int cout_end, int block_start, int block_end,
                                  int block_h, int block_w, int out_hw, int out_w, int resi_h, int resi_w,
                                  int activation)
{
    int p, i;

    float* ker_ptr;
    float* inp_ptr;
    for(p = (cout_start & -KER_COUT_UNIT4); p < (cout_end & -KER_COUT_UNIT4); p += KER_COUT_UNIT4)
    {
        ker_ptr = ker + p * ELEM_SIZE * cin;

        for(i = (block_start & -4); i < (block_end & -4); i += 4)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            float out_buffer[KER_COUT_UNIT4 * BLOCK_HW_UNIT * ELEM_SIZE];
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                kernel_func(out_buffer + s * BLOCK_HW_UNIT * KER_COUT_UNIT4,
                                      inp_ptr + s * BLOCK_HW_UNIT * cin, ker_ptr + s * KER_COUT_UNIT4 * cin, cin);
            }

            // interleave
            float buffer[KER_COUT_UNIT4 * BLOCK_HW_UNIT * ELEM_SIZE];
            float* buffer_ptr0 = buffer;
            for(int pp = 0; pp < KER_COUT_UNIT4; pp++)
            {
                for(int t = 0; t < BLOCK_HW_UNIT; t++)
                {
                    for(int ss = 0; ss < ELEM_SIZE; ss++)
                    {
                        *buffer_ptr0 = out_buffer[ss * BLOCK_HW_UNIT * KER_COUT_UNIT4 + pp * BLOCK_HW_UNIT + t];
                        buffer_ptr0++;
                    }
                }
            }
            // end interleave
            transform_output_f43_4tile(buffer, output, p, i, block_h, block_w, out_hw, out_w, resi_h, resi_w,
                                       KER_COUT_UNIT4, bias, bias_term, activation);
            // end transform
        }

        for(; i < block_end; i++)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            float out_buffer[KER_COUT_UNIT4 * ELEM_SIZE];
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                wino_sgemm_1x4_cpu(out_buffer + s * KER_COUT_UNIT4, inp_ptr + s * cin,
                                   ker_ptr + s * KER_COUT_UNIT4 * cin, cin);
            }
            // interleave
            float buffer[KER_COUT_UNIT4 * ELEM_SIZE];
            float* buffer_ptr0 = buffer;

            for(int pp = 0; pp < KER_COUT_UNIT4; pp++)
            {
                for(int ss = 0; ss < ELEM_SIZE; ss++)
                {
                    *buffer_ptr0 = out_buffer[ss * KER_COUT_UNIT4 + pp];
                    buffer_ptr0++;
                }
            }
            // end interleave
            transform_output_f43_1tile(buffer, output, p, i, block_h, block_w, out_hw, out_w, resi_h, resi_w,
                                       KER_COUT_UNIT4, bias, bias_term, activation);
            // end transform
        }
    }
    for(p = (cout_end & -KER_COUT_UNIT4); p < cout_end; p ++)
    {
        ker_ptr = ker + p * ELEM_SIZE * cin;
        for(i = (block_start & -4); i < (block_end & -4); i += 4)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;
            float buffer[BLOCK_HW_UNIT * ELEM_SIZE];
            int idx_h[4];
            int idx_w[4];
            idx_h[0] = (i) / block_w;
            idx_h[1] = (i + 1) / block_w;
            idx_h[2] = (i + 2) / block_w;
            idx_h[3] = (i + 3) / block_w;

            idx_w[0] = (i) % block_w;
            idx_w[1] = (i + 1) % block_w;
            idx_w[2] = (i + 2) % block_w;
            idx_w[3] = (i + 3) % block_w;

            //gemm+interleave buffer[4][36]
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                float* inp_ = (float*)(inp_ptr + s * BLOCK_HW_UNIT * cin);
                float* ker_ = (float*)(ker_ptr + s * cin);

                float sum0 = 0;
                float sum1 = 0;
                float sum2 = 0;
                float sum3 = 0;
                for(int k = 0; k < cin; k++)
                {
                    sum0 += inp_[k * 4    ] * ker_[k];
                    sum1 += inp_[k * 4 + 1] * ker_[k];
                    sum2 += inp_[k * 4 + 2] * ker_[k];
                    sum3 += inp_[k * 4 + 3] * ker_[k];
                }
                buffer[      s] = sum0;
                buffer[ 36 + s] = sum1;
                buffer[ 72 + s] = sum2;
                buffer[108 + s] = sum3;
            }
            //trans_out buffer[4][36]
            float tmp_buffer[TILE * TILE];
            float* bias_ptr = NULL;     
            float* out_ptr = output + p * out_hw;
            if(bias_term)
            {
                bias_ptr = (float*)(bias + p);
            }
            for(int ii = 0; ii < BLOCK_HW_UNIT; ii++)
            {
                int i_h = idx_h[ii];
                int j_w = idx_w[ii];
                if((resi_h == 0 && resi_w == 0) || (resi_h == 0 && (j_w < block_w - 1)) ||
                    (resi_w == 0 && (i_h < block_h - 1)) || ((j_w < block_w - 1) && (i_h < block_h - 1)))
                {
                    trans_output_f43(buffer + ii * ELEM_SIZE,
                                        out_ptr + (i_h * TILE * out_w + j_w * TILE), out_w,
                                        bias_ptr, activation);
                }    // direct use_out_ptr
                else
                {
                    int ret_h = TILE - resi_h;
                    if(i_h < block_h - 1)
                        ret_h = TILE;
                    int ret_w = TILE - resi_w;
                    if(j_w < block_w - 1)
                        ret_w = TILE;
                    // tmp_buffer
                    trans_output_f43_ordinary(buffer + ii * ELEM_SIZE, tmp_buffer,
                                                bias_ptr);
                    float* out_pointer = out_ptr + (i_h * TILE * out_w + j_w * TILE);
                    for(int hh = 0; hh < ret_h; hh++)
                    {
                        for(int ww = 0; ww < ret_w; ww++)
                        {
                            out_pointer[hh * out_w + ww] =
                                do_activation(tmp_buffer[hh * 4 + ww], activation);
                        }
                    }

                }    // end else, tmp_buff
            }// end transform
        }

        for(; i < block_end; i++)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            float buffer[ELEM_SIZE];
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                float* inp_ = (float*)(inp_ptr + s * cin);
                float* ker_ = (float*)(ker_ptr + s * cin);

                float sum = 0;
                for(int k = 0; k < cin; k++)
                {
                    sum += inp_[k] * ker_[k];
                }
                buffer[s] = sum;
            }
            // end interleave
            transform_output_f43_1tile(buffer, output, p, i, block_h, block_w, out_hw, out_w, resi_h,
                                       resi_w, 1, bias, bias_term, activation);
            // end transform
        }
    }
}
static inline void wino_sgemm_4x12_nhwc(float* ker, float* inp, float* output, const float* bias, int bias_term,
                                        int cin, sgemm_kernel_t kernel_func,  sgemm_kernel_t kernel_func1, 
                                        int cout_start, int cout_end, int block_start,
                                        int block_end, int block_h, int block_w, int out_c, int out_w, int resi_h,
                                        int resi_w, int activation)
{
    int p, i;

    float* ker_ptr;
    float* inp_ptr;

    for(p = cout_start; p < cout_end; p += KER_COUT_UNIT)
    {
        ker_ptr = ker + p * ELEM_SIZE * cin;

        for(i = (block_start & -4); i < (block_end & -4); i += 4)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            float out_buffer[KER_COUT_UNIT * BLOCK_HW_UNIT * ELEM_SIZE];
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                kernel_func(out_buffer + s * BLOCK_HW_UNIT * KER_COUT_UNIT,
                                       inp_ptr + s * BLOCK_HW_UNIT * cin, ker_ptr + s * KER_COUT_UNIT * cin, cin);
            }

            // interleave
            float buffer[KER_COUT_UNIT * BLOCK_HW_UNIT * ELEM_SIZE];
            float* buffer_ptr0 = buffer;
            for(int pp = 0; pp < KER_COUT_UNIT; pp++)
            {
                for(int t = 0; t < BLOCK_HW_UNIT; t++)
                {
                    for(int ss = 0; ss < ELEM_SIZE; ss++)
                    {
                        *buffer_ptr0 = out_buffer[ss * BLOCK_HW_UNIT * KER_COUT_UNIT + pp * BLOCK_HW_UNIT + t];
                        buffer_ptr0++;
                    }
                }
            }
            // end interleave
            transform_output_f43_4tile_nhwc(buffer, output, p, i, block_h, block_w, out_c, out_w, resi_h, resi_w,
                                            KER_COUT_UNIT, bias, bias_term, activation);
            // end transform
        }

        for(; i < block_end; i++)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            float out_buffer[KER_COUT_UNIT * ELEM_SIZE];
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                kernel_func1(out_buffer + s * KER_COUT_UNIT, inp_ptr + s * cin,
                                       ker_ptr + s * KER_COUT_UNIT * cin, cin);
            }
            // interleave
            float buffer[KER_COUT_UNIT * ELEM_SIZE];
            float* buffer_ptr0 = buffer;

            for(int pp = 0; pp < KER_COUT_UNIT; pp++)
            {
                for(int ss = 0; ss < ELEM_SIZE; ss++)
                {
                    *buffer_ptr0 = out_buffer[ss * KER_COUT_UNIT + pp];
                    buffer_ptr0++;
                }
            }
            // end interleave
            transform_output_f43_1tile_nhwc(buffer, output, p, i, block_h, block_w, out_c, out_w, resi_h, resi_w,
                                            KER_COUT_UNIT, bias, bias_term, activation);
            // end transform
        }
    }
}
static inline void wino_sgemm_4x4_nhwc(float* ker, float* inp, float* output, const float* bias, int bias_term, int cin,
                                       sgemm_kernel_t kernel_func, int cout_start, int cout_end, int block_start, int block_end,
                                       int block_h, int block_w, int out_c, int out_w, int resi_h, int resi_w,
                                       int activation)
{
    int p, i;

    float* ker_ptr;
    float* inp_ptr;

    for(p = cout_start; p < cout_end; p += KER_COUT_UNIT4)
    {
        ker_ptr = ker + p * ELEM_SIZE * cin;

        for(i = (block_start & -4); i < (block_end & -4); i += 4)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            float out_buffer[KER_COUT_UNIT4 * BLOCK_HW_UNIT * ELEM_SIZE];
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                kernel_func(out_buffer + s * BLOCK_HW_UNIT * KER_COUT_UNIT4,
                                      inp_ptr + s * BLOCK_HW_UNIT * cin, ker_ptr + s * KER_COUT_UNIT4 * cin, cin);
            }

            // interleave
            float buffer[KER_COUT_UNIT4 * BLOCK_HW_UNIT * ELEM_SIZE];
            float* buffer_ptr0 = buffer;
            for(int pp = 0; pp < KER_COUT_UNIT4; pp++)
            {
                for(int t = 0; t < BLOCK_HW_UNIT; t++)
                {
                    for(int ss = 0; ss < ELEM_SIZE; ss++)
                    {
                        *buffer_ptr0 = out_buffer[ss * BLOCK_HW_UNIT * KER_COUT_UNIT4 + pp * BLOCK_HW_UNIT + t];
                        buffer_ptr0++;
                    }
                }
            }
            // end interleave
            transform_output_f43_4tile_nhwc(buffer, output, p, i, block_h, block_w, out_c, out_w, resi_h, resi_w,
                                            KER_COUT_UNIT4, bias, bias_term, activation);
            // end transform
        }

        for(; i < block_end; i++)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            float out_buffer[KER_COUT_UNIT4 * ELEM_SIZE];
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                wino_sgemm_1x4_cpu(out_buffer + s * KER_COUT_UNIT4, inp_ptr + s * cin,
                                   ker_ptr + s * KER_COUT_UNIT4 * cin, cin);
            }
            // interleave
            float buffer[KER_COUT_UNIT4 * ELEM_SIZE];
            float* buffer_ptr0 = buffer;

            for(int pp = 0; pp < KER_COUT_UNIT4; pp++)
            {
                for(int ss = 0; ss < ELEM_SIZE; ss++)
                {
                    *buffer_ptr0 = out_buffer[ss * KER_COUT_UNIT4 + pp];
                    buffer_ptr0++;
                }
            }
            // end interleave
            transform_output_f43_1tile_nhwc(buffer, output, p, i, block_h, block_w, out_c, out_w, resi_h, resi_w,
                                            KER_COUT_UNIT4, bias, bias_term, activation);
            // end transform
        }
    }
}
#ifdef __cplusplus
}
#endif    //__cplusplus

#endif    // __WINO_SGENN_H__
