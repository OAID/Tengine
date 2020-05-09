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

#ifndef __WINO_TRANS_KER_H__
#define __WINO_TRANS_KER_H__

#include "wino_config.h"

#ifdef __cplusplus
extern "C" {
#endif    //__cplusplus

static inline void trans_kernel_f43(__fp16* ker, __fp16* trans_ker)
{
    /*
    __fp16 G[18]={
      1./4   ,     0.   ,     0.     ,
      -1./6  ,   -1./6  ,    -1./6   ,
      -1./6  ,    1./6  ,    -1./6   ,
      1./24  ,   1./12  ,    1./6    ,
      1./24  ,   -1./12 ,    1./6    ,
      0.     ,    0.    ,     1.
    };
    __fp16 GT[18]={
      1./4 ,  -1./6, -1./6 ,  1./24, 1./24 ,  0.,
      0.,     -1./6,  1./6 ,  1./12, -1./12 ,  0.,
      0.,     -1./6,  -1./6 ,  1./6, 1./6 ,  1.
    };
    */
    __fp16 tmp[18] = {0};

    __fp16 neg_r0_add_r2_x_1_6[6];    // (r0+r2)*1./6
    __fp16 r0_1_4_add_r2_x_1_6[6];    // (r0*1/4 + r2)*1./6
    __fp16 r1_1_6[6];    // r1*1/6
    __fp16 r1_1_12[6];    // r1*1/12
    __fp16 s_1_6 = 1. / 6.f;
    for(int j = 0; j < 3; j++)
    {
        neg_r0_add_r2_x_1_6[j] = -(ker[j] + ker[6 + j]) * s_1_6;
        r0_1_4_add_r2_x_1_6[j] = (ker[j] * 0.25 + ker[6 + j]) * s_1_6;
        r1_1_6[j] = ker[3 + j] * s_1_6;
        r1_1_12[j] = r1_1_6[j] * 0.5;
    }
    for(int j = 0; j < 3; j++)
    {
        tmp[j] =     8*( ker[j] * 0.25);
        tmp[3 + j] = 8*(-r1_1_6[j] + neg_r0_add_r2_x_1_6[j]);
        tmp[6 + j] = 8*(r1_1_6[j] + neg_r0_add_r2_x_1_6[j]);
        tmp[9 + j] = 8*(r1_1_12[j] + r0_1_4_add_r2_x_1_6[j]);
        tmp[12 + j] =8*( -r1_1_12[j] + r0_1_4_add_r2_x_1_6[j]);
        tmp[15 + j] =8*( ker[6 + j]);
    }
    // gemm(6,3,3,G,ker,tmp); done
    int idx;
    for(int j = 0; j < 6; j++)
    {
        idx = j * 3;
        neg_r0_add_r2_x_1_6[j] = -(tmp[idx] + tmp[idx + 2]) * s_1_6;
        r0_1_4_add_r2_x_1_6[j] = (tmp[idx] * 0.25 + tmp[idx + 2]) * s_1_6;
        r1_1_6[j] = tmp[idx + 1] * s_1_6;
        r1_1_12[j] = r1_1_6[j] * 0.5;
    }

    for(int j = 0; j < 6; j++)
    {
        idx = j * 6;
        trans_ker[idx] =    8*( tmp[j * 3] * 0.25);
        trans_ker[idx + 1] =8*( -r1_1_6[j] + neg_r0_add_r2_x_1_6[j]);
        trans_ker[idx + 2] =8*( r1_1_6[j] + neg_r0_add_r2_x_1_6[j] );
        trans_ker[idx + 3] =8*( r1_1_12[j] + r0_1_4_add_r2_x_1_6[j]);
        trans_ker[idx + 4] =8*( -r1_1_12[j] + r0_1_4_add_r2_x_1_6[j]);
        trans_ker[idx + 5] =8*( tmp[j * 3 + 2]);
    }
    // gemm(6,6,3,tmp,GT,trans_ker); done
}

static inline void transform_kernel_f43_tile(const __fp16* kernel, __fp16* trans_ker, int inc, int outc)
{
    __fp16* ker_ptr = trans_ker;

    for(int i = 0; i < outc; i++)
    {
        for(int j = 0; j < inc; j++)
        {
            trans_kernel_f43(( __fp16* )(kernel + 9 * (j + i * inc)), ker_ptr);
            ker_ptr += 36;
        }
    }
}
// ker0 [cout][cin][ELEM_SIZE]
// ker1 [cout//KER_COUT_UNIT][ELEM_SIZE][cin][KER_COUT_UNIT]
// ->   [ELEM_SIZE][cout//KER_COUT_UNIT][cin][KER_COUT_UNIT]
static inline void interleave_kernel_1(__fp16* ker0, __fp16* ker1, int cout, int cin)
{
    int nn_cout = cout / KER_COUT_UNIT;
    __fp16* ker1_ptr = ker1;
    for(int s = 0; s < ELEM_SIZE; s++)
    {
        for(int p = 0; p < nn_cout; p++)
        {
            for(int i = 0; i < cin; i++)
            {
                for(int j = 0; j < KER_COUT_UNIT; j++)
                {
                    int cout_idx = p * KER_COUT_UNIT + j;
                    *ker1_ptr = ker0[(cout_idx * cin + i) * ELEM_SIZE + s];
                    ker1_ptr++;
                }
            }
        }
    }
}
// ker0 [cout][cin][ELEM_SIZE]
// ker1 [cout//KER_COUT_UNIT][ELEM_SIZE][cin][KER_COUT_UNIT]
static inline void interleave_kernel(__fp16* ker0, __fp16* ker1, int cout, int cin)
{
    int nn_cout = cout / KER_COUT_UNIT;
    int resi_cout = nn_cout * KER_COUT_UNIT;
    __fp16* ker1_ptr = ker1;
    for(int p = 0; p < nn_cout; p++)
    {
        for(int s = 0; s < ELEM_SIZE; s++)
        {
            for(int i = 0; i < cin; i++)
            {
                for(int j = 0; j < KER_COUT_UNIT; j++)
                {
                    int cout_idx = p * KER_COUT_UNIT + j;
                    *ker1_ptr = ker0[(cout_idx * cin + i) * ELEM_SIZE + s];
                    ker1_ptr++;
                }
            }
        }
    }
    for(int p = resi_cout; p < cout; p += KER_COUT_UNIT4)
    {
        for(int s = 0; s < ELEM_SIZE; s++)
        {
            for(int i = 0; i < cin; i++)
            {
                for(int j = 0; j < KER_COUT_UNIT4; j++)
                {
                    int cout_idx = p + j;
                    *ker1_ptr = ker0[(cout_idx * cin + i) * ELEM_SIZE + s];
                    ker1_ptr++;
                }
            }
        }
    }
}

#ifdef __cplusplus
}
#endif    //__cplusplus

#endif    // __WINO_TRANS_KER_H__