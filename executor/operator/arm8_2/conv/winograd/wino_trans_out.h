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

#ifndef __WINO_TRANS_OUT_H__
#define __WINO_TRANS_OUT_H__

#include "wino_config.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif    //__cplusplus

// extern void tran_out_4(__fp16* buffer,  __fp16* output,int out_w,__fp16*ker00, __fp16*bias,int activation);

// ------------------------------ OUTPUT ------------------------------------------
static inline void trans_output_f43(const __fp16* mid, __fp16* out, int outw, const __fp16* bias_ptr,int activation)
{
    /*
   __fp16 AT[24]={
     1.,  1.,  1.,  1.,  1.,  0.,
     0.,  1., -1.,  2., -2.,  0.,
     0.,  1.,  1.,  4.,  4.,  0.,
     0.,  1., -1.,  8., -8.,  1.
   };


   __fp16 A[24]={
     1.,  0.,  0.,  0.,
     1.,  1.,  1.,  1.,
     1., -1.,  1., -1.,
     1.,  2.,  4.,  8.,
     1., -2.,  4., -8.,
     0.,  0.,  0.,  1.
   };
   */
    __fp16 tmp[24] = {0};

    __fp16 r1_add_r2[6];
    __fp16 r1_minus_r2[6];
    __fp16 r3_add_r4[6];
    __fp16 r3_minus_r4_x2[6];

    for(int j = 0; j < 6; j++)
    {
        r1_add_r2[j] = mid[6 * 1 + j] + mid[6 * 2 + j];
        r1_minus_r2[j] = mid[6 * 1 + j] - mid[6 * 2 + j];
        r3_add_r4[j] = mid[6 * 3 + j] + mid[6 * 4 + j];
        r3_minus_r4_x2[j] = (mid[6 * 3 + j] - mid[6 * 4 + j]) * 2;
    }
    for(int j = 0; j < 6; j++)
    {
        tmp[j]      =0.125*(  mid[j] + r1_add_r2[j] + r3_add_r4[j]);
        tmp[6 + j]  =0.125*( r1_minus_r2[j] + r3_minus_r4_x2[j]   );
        tmp[12 + j] =0.125*( r1_add_r2[j] + 4 * r3_add_r4[j]      );
        tmp[18 + j] =0.125*( r1_minus_r2[j] + 4 * r3_minus_r4_x2[j] + mid[6 * 5 + j]);
    }

    __fp16* out0 = out;
    __fp16* out1 = out0 + outw;
    __fp16* out2 = out1 + outw;
    __fp16* out3 = out2 + outw;

    __fp16 _r1_add_r2[4];
    __fp16 _r1_minus_r2[4];
    __fp16 _r3_add_r4[4];
    __fp16 _r3_minus_r4_x2[4];
    int idx;
    for(int j = 0; j < 4; j++)
    {
        idx = 6 * j;
        _r1_add_r2[j] = tmp[idx + 1] + tmp[idx + 2];
        _r1_minus_r2[j] = tmp[idx + 1] - tmp[idx + 2];
        _r3_add_r4[j] = tmp[idx + 3] + tmp[idx + 4];
        _r3_minus_r4_x2[j] = (tmp[idx + 3] - tmp[idx + 4]) * 2;
    }
    if(bias_ptr)
    {
        __fp16 bias = bias_ptr[0];
        out0[0] = do_activation(32*(tmp[0 * 6] + _r1_add_r2[0] + _r3_add_r4[0]) + bias,activation);
        out1[0] = do_activation(32*(tmp[1 * 6] + _r1_add_r2[1] + _r3_add_r4[1]) + bias,activation);
        out2[0] = do_activation(32*(tmp[2 * 6] + _r1_add_r2[2] + _r3_add_r4[2]) + bias,activation);
        out3[0] = do_activation(32*(tmp[3 * 6] + _r1_add_r2[3] + _r3_add_r4[3]) + bias,activation);

        out0[1] = do_activation(32*(_r1_minus_r2[0] + _r3_minus_r4_x2[0]) + bias,activation);
        out1[1] = do_activation(32*(_r1_minus_r2[1] + _r3_minus_r4_x2[1]) + bias,activation);
        out2[1] = do_activation(32*(_r1_minus_r2[2] + _r3_minus_r4_x2[2]) + bias,activation);
        out3[1] = do_activation(32*(_r1_minus_r2[3] + _r3_minus_r4_x2[3]) + bias,activation);

        out0[2] = do_activation(32*(_r1_add_r2[0] + 4 * _r3_add_r4[0]) + bias,activation);
        out1[2] = do_activation(32*(_r1_add_r2[1] + 4 * _r3_add_r4[1]) + bias,activation);
        out2[2] = do_activation(32*(_r1_add_r2[2] + 4 * _r3_add_r4[2]) + bias,activation);
        out3[2] = do_activation(32*(_r1_add_r2[3] + 4 * _r3_add_r4[3]) + bias,activation);

        out0[3] = do_activation(32*(_r1_minus_r2[0] + 4 * _r3_minus_r4_x2[0] + tmp[0 * 6 + 5]) + bias,activation);
        out1[3] = do_activation(32*(_r1_minus_r2[1] + 4 * _r3_minus_r4_x2[1] + tmp[1 * 6 + 5]) + bias,activation);
        out2[3] = do_activation(32*(_r1_minus_r2[2] + 4 * _r3_minus_r4_x2[2] + tmp[2 * 6 + 5]) + bias,activation);
        out3[3] = do_activation(32*(_r1_minus_r2[3] + 4 * _r3_minus_r4_x2[3] + tmp[3 * 6 + 5]) + bias,activation);
    }
    else
    {
        out0[0] = do_activation(32*(tmp[0 * 6] + _r1_add_r2[0] + _r3_add_r4[0]),activation);
        out1[0] = do_activation(32*(tmp[1 * 6] + _r1_add_r2[1] + _r3_add_r4[1]),activation);
        out2[0] = do_activation(32*(tmp[2 * 6] + _r1_add_r2[2] + _r3_add_r4[2]),activation);
        out3[0] = do_activation(32*(tmp[3 * 6] + _r1_add_r2[3] + _r3_add_r4[3]),activation);

        out0[1] = do_activation(32*(_r1_minus_r2[0] + _r3_minus_r4_x2[0]), activation);
        out1[1] = do_activation(32*(_r1_minus_r2[1] + _r3_minus_r4_x2[1]), activation);
        out2[1] = do_activation(32*(_r1_minus_r2[2] + _r3_minus_r4_x2[2]), activation);
        out3[1] = do_activation(32*(_r1_minus_r2[3] + _r3_minus_r4_x2[3]), activation);

        out0[2] = do_activation(32*(_r1_add_r2[0] + 4 * _r3_add_r4[0]), activation);
        out1[2] = do_activation(32*(_r1_add_r2[1] + 4 * _r3_add_r4[1]), activation);
        out2[2] = do_activation(32*(_r1_add_r2[2] + 4 * _r3_add_r4[2]), activation);
        out3[2] = do_activation(32*(_r1_add_r2[3] + 4 * _r3_add_r4[3]), activation);

        out0[3] = do_activation(32*(_r1_minus_r2[0] + 4 * _r3_minus_r4_x2[0] + tmp[0 * 6 + 5]), activation);
        out1[3] = do_activation(32*(_r1_minus_r2[1] + 4 * _r3_minus_r4_x2[1] + tmp[1 * 6 + 5]), activation);
        out2[3] = do_activation(32*(_r1_minus_r2[2] + 4 * _r3_minus_r4_x2[2] + tmp[2 * 6 + 5]), activation);
        out3[3] = do_activation(32*(_r1_minus_r2[3] + 4 * _r3_minus_r4_x2[3] + tmp[3 * 6 + 5]), activation);
    }
}
static inline void trans_output_f43_ordinary(const __fp16* mid, __fp16* out, const __fp16* bias_ptr)
{
    /*
    __fp16 AT[24]={
      1.,  1.,  1.,  1.,  1.,  0.,
      0.,  1., -1.,  2., -2.,  0.,
      0.,  1.,  1.,  4.,  4.,  0.,
      0.,  1., -1.,  8., -8.,  1.
    };
    __fp16 A[24]={
      1.,  0.,  0.,  0.,
      1.,  1.,  1.,  1.,
      1., -1.,  1., -1.,
      1.,  2.,  4.,  8.,
      1., -2.,  4., -8.,
      0.,  0.,  0.,  1.
    };
    */

    __fp16 tmp[24] = {0};

    __fp16 r1_add_r2[6];
    __fp16 r1_minus_r2[6];
    __fp16 r3_add_r4[6];
    __fp16 r3_minus_r4_x2[6];

    for(int j = 0; j < 6; j++)
    {
        r1_add_r2[j] = mid[6 * 1 + j] + mid[6 * 2 + j];
        r1_minus_r2[j] = mid[6 * 1 + j] - mid[6 * 2 + j];
        r3_add_r4[j] = mid[6 * 3 + j] + mid[6 * 4 + j];
        r3_minus_r4_x2[j] = (mid[6 * 3 + j] - mid[6 * 4 + j]) * 2;
    }
    for(int j = 0; j < 6; j++)
    {
        tmp[j]      = 0.125*(mid[j] + r1_add_r2[j] + r3_add_r4[j]);
        tmp[6 + j]  = 0.125*(r1_minus_r2[j] + r3_minus_r4_x2[j]);
        tmp[12 + j] = 0.125*(r1_add_r2[j] + 4 * r3_add_r4[j]);
        tmp[18 + j] = 0.125*(r1_minus_r2[j] + 4 * r3_minus_r4_x2[j] + mid[6 * 5 + j]);
    }
    __fp16 _r1_add_r2[4];
    __fp16 _r1_minus_r2[4];
    __fp16 _r3_add_r4[4];
    __fp16 _r3_minus_r4_x2[4];
    int idx;
    for(int j = 0; j < 4; j++)
    {
        idx = 6 * j;
        _r1_add_r2[j] = tmp[idx + 1] + tmp[idx + 2];
        _r1_minus_r2[j] = tmp[idx + 1] - tmp[idx + 2];
        _r3_add_r4[j] = tmp[idx + 3] + tmp[idx + 4];
        _r3_minus_r4_x2[j] = (tmp[idx + 3] - tmp[idx + 4]) * 2;
    }
    if(bias_ptr)
    {
        __fp16 bias = bias_ptr[0];
        for(int j = 0; j < 4; j++)
        {
            idx = j * 4;
            out[idx] =     bias + 32*(tmp[j * 6] + _r1_add_r2[j] + _r3_add_r4[j]);
            out[idx + 1] = bias + 32*(_r1_minus_r2[j] + _r3_minus_r4_x2[j]);
            out[idx + 2] = bias + 32*(_r1_add_r2[j] + 4 * _r3_add_r4[j]);
            out[idx + 3] = bias + 32*(_r1_minus_r2[j] + 4 * _r3_minus_r4_x2[j] + tmp[j * 6 + 5]);
        }
    }
    else
    {
        for(int j = 0; j < 4; j++)
        {
            idx = j * 4;
            out[idx]     = 32*(tmp[j * 6] + _r1_add_r2[j] + _r3_add_r4[j]);
            out[idx + 1] = 32*(_r1_minus_r2[j] + _r3_minus_r4_x2[j]);
            out[idx + 2] = 32*(_r1_add_r2[j] + 4 * _r3_add_r4[j]);
            out[idx + 3] = 32*(_r1_minus_r2[j] + 4 * _r3_minus_r4_x2[j] + tmp[j * 6 + 5]);
        }
    }
}

// mid [ELEM_SIZE][KER_COUT_UNIT]
// interleave [KER_COUT_UNIT][ELEM_SIZE]
static inline void transform_output_f43_1tile(const __fp16* buffer_ptr, __fp16* out, int p_idx, int idx_blockhw, int block_h,
                                              int block_w, int out_hw, int outw, int resi_h, int resi_w,
                                              int KER_COUT_UNIT_, const __fp16* bias, int bias_term,int activation)
{
    __fp16 tmp_buffer[TILE * TILE];
    const __fp16* bias_ptr = NULL;
    for(int p = 0; p < KER_COUT_UNIT_; p++)
    {
        int cout_idx = p_idx + p;
        if(bias_term)
        {
            bias_ptr = (bias + cout_idx);
        }
        __fp16* out_ptr = out + cout_idx * out_hw;
        int i_h = idx_blockhw / block_w;
        int j_w = idx_blockhw % block_w;
        if((resi_h == 0 && resi_w == 0) || (resi_h == 0 && (j_w < block_w - 1)) ||
           (resi_w == 0 && (i_h < block_h - 1)) || ((j_w < block_w - 1) && (i_h < block_h - 1)))
        {
            trans_output_f43(buffer_ptr, out_ptr + (i_h * TILE * outw + j_w * TILE), outw, bias_ptr,activation);
        }
        else
        {
            int ret_h = TILE - resi_h;
            if(i_h < block_h - 1)
                ret_h = TILE;
            int ret_w = TILE - resi_w;
            if(j_w < block_w - 1)
                ret_w = TILE;

            // tmp_buffer
            trans_output_f43_ordinary(buffer_ptr, tmp_buffer, bias_ptr);
            __fp16* out_pointer = out_ptr + (i_h * TILE * outw + j_w * TILE);
            for(int hh = 0; hh < ret_h; hh++)
            {
                for(int ww = 0; ww < ret_w; ww++)
                {
                    out_pointer[hh * outw + ww] =do_activation(tmp_buffer[hh * TILE + ww],activation);
                }
            }
        }
        buffer_ptr += ELEM_SIZE;
    }
}

static inline void trans_output(__fp16* trans_out,__fp16* output,__fp16* bias,int bias_term,
    int block_h,int block_w,int cout_start,int cout_end,int out_hw,int out_w,int resi_h,int resi_w,int activation)
{
    // int flag_outw=1;
    // if(out_w<16)flag_outw=0;
    int block_hw=block_h*block_w;
    int nn_block=block_hw/BLOCK_HW_UNIT;
    int block_= nn_block*BLOCK_HW_UNIT;
    int i;
    for(int p = cout_start; p < cout_end; p += KER_COUT_UNIT)
    {
        for(i=0;i<block_;i+=BLOCK_HW_UNIT )
        {
            __fp16* buffer_ptr = trans_out + p* block_hw*ELEM_SIZE + i*KER_COUT_UNIT*ELEM_SIZE;

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
            // int wino_out_4_tiles=0;
            // if(flag_outw)
            // {
            //     if((idx_h[0]==idx_h[3]) &&  (idx_h[0] <(block_h - 1)) && (idx_w[3] < (block_w - 1)))
            //     {
            //         wino_out_4_tiles=1;
            //     }
            // }
            // if (wino_out_4_tiles==1)
            // {
            //     __fp16* bias_ptr = NULL;
            //     for(int pss = 0; pss < KER_COUT_UNIT; pss++)
            //     {
            //         int cout_idx = p + pss;
            //         __fp16* out_ptr = output + cout_idx * out_hw + idx_h[0] * TILE * out_w + idx_w[0] * TILE;
            //         if(bias_term)
            //         {
            //             bias_ptr =(__fp16*)( bias + cout_idx);
            //         }
            //         __fp16 ker00[4] = {2,4,8,0};
            //         tran_out_4(buffer_ptr+pss*ELEM_SIZE*BLOCK_HW_UNIT, 
            //                     out_ptr, 
            //                     out_w*sizeof(__fp16), ker00,bias_ptr,activation);
            //     }
            // }
            // else
            {
                __fp16 tmp_buffer[TILE * TILE];
                const __fp16* bias_ptr = NULL;
                for(int pss = 0; pss < KER_COUT_UNIT; pss++)
                {
                    int cout_idx = p + pss;
                    __fp16* out_ptr = output + cout_idx * out_hw;
                    if(bias_term)
                    {
                        bias_ptr = bias + cout_idx;
                    }
                    __fp16 buffer[ BLOCK_HW_UNIT * ELEM_SIZE];
                    __fp16* buffer_ptr0 = buffer;
                    __fp16* mid_ptr = buffer_ptr + pss;
                    //mid[36][4][16]
                    for(int t = 0; t < BLOCK_HW_UNIT; t++)
                    {
                        for(int ss = 0; ss < ELEM_SIZE; ss++)
                        {
                            *buffer_ptr0 = mid_ptr[(ss * BLOCK_HW_UNIT + t)*KER_COUT_UNIT];
                            buffer_ptr0++;
                        }
                    }
                    for(int ii = 0; ii < BLOCK_HW_UNIT; ii++)
                    {
                        int i_h = idx_h[ii];
                        int j_w = idx_w[ii];
                        if((resi_h == 0 && resi_w == 0) || (resi_h == 0 && (j_w < block_w - 1)) ||
                        (resi_w == 0 && (i_h < block_h - 1)) || ((j_w < block_w - 1) && (i_h < block_h - 1)))
                        {
                            trans_output_f43(buffer+ii*ELEM_SIZE, 
                                                out_ptr + (i_h * TILE * out_w + j_w * TILE), 
                                                out_w, (const __fp16*)bias_ptr,
                                                activation);
                        }
                        else
                        {
                            int ret_h = TILE - resi_h;
                            if(i_h < block_h - 1)
                                ret_h = TILE;
                            int ret_w = TILE - resi_w;
                            if(j_w < block_w - 1)
                                ret_w = TILE;
                            trans_output_f43_ordinary(buffer+ii*ELEM_SIZE, tmp_buffer,  (const __fp16*)bias_ptr);
                            __fp16* out_pointer = out_ptr + (i_h * TILE * out_w + j_w * TILE);
                            for(int hh = 0; hh < ret_h; hh++)
                            {
                                for(int ww = 0; ww < ret_w; ww++)
                                {
                                    out_pointer[hh * out_w + ww] = do_activation(tmp_buffer[hh * 4 + ww],activation);
                                }
                            }
                        }
                    }
                }
            }
        }
        for(;i<block_hw;i++)
        {
            __fp16* buffer_ptr = trans_out + p* block_hw*ELEM_SIZE + i*KER_COUT_UNIT*ELEM_SIZE;
            __fp16 resi_buffer[KER_COUT_UNIT*ELEM_SIZE];
            __fp16* buffer0=resi_buffer;            
            for(int pp = 0; pp < KER_COUT_UNIT; pp++)
            {
                for(int ss = 0; ss < ELEM_SIZE; ss++)
                {
                    *buffer0 = buffer_ptr[ss  * KER_COUT_UNIT + pp ];
                    buffer0++;
                }
            }
            transform_output_f43_1tile(resi_buffer, output,
                p, i,
                block_h, block_w,
                out_hw, out_w,resi_h,resi_w,
                KER_COUT_UNIT,bias,bias_term,activation);
        }
    }
}

#ifdef __cplusplus
}
#endif    //__cplusplus

#endif    // __WINO_TRANS_INP_OUT_H__