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

#ifndef __WINO_TRANS_INP_H__
#define __WINO_TRANS_INP_H__

#include "wino_config.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif    //__cplusplus

extern void tran_inp_4(const float* inp, const float* out, float* ker, int inw, int inc4, int inhw);
// ------------------------------ INPUT ------------------------------------------
/*
    float BT[36]={
        4.,  0., -5.,  0.,  1.,  0.,
        0., -4., -4.,  1.,  1.,  0.,
        0.,  4., -4., -1.,  1.,  0.,
        0., -2., -1.,  2.,  1.,  0.,
        0.,  2., -1., -2.,  1.,  0.,
        0.,  4.,  0., -5.,  0.,  1.
    };

    float B[36]={
        4.,  0.,  0.,  0.,  0.,  0.,
        0., -4.,  4., -2.,  2.,  4.,
        -5.,-4., -4., -1., -1.,  0.,
        0.,  1., -1.,  2., -2., -5.,
        1.,  1.,1.,  1.,  1.,  0.,
        0.,  0.,  0.,  0.,  0.,  1.
    };
*/
static inline void trans_inp_1tile(float* input, float* inp_ptr, int ih, int jw, int c, int in_hw, int inw)
{
    float* inp = ( float* )input + c * in_hw + ih * 4 * inw + jw * 4;
    float* inp0 = inp;
    float* inp1 = inp0 + inw;
    float* inp2 = inp1 + inw;
    float* inp3 = inp2 + inw;
    float* inp4 = inp3 + inw;
    float* inp5 = inp4 + inw;
    float tmp[36] = {0};

    float r1_add_r2[6];
    float r3_add_r4[6];
    float r1_minus_r2[6];
    float r3_minus_r4[6];
    float r4_minus_r2[6];
    float r1_minus_r3[6];

    for(int j = 0; j < 6; j++)
    {
        r1_add_r2[j] = inp1[j] + inp2[j];
        r1_minus_r2[j] = inp1[j] - inp2[j];
        r3_add_r4[j] = inp3[j] + inp4[j];
        r3_minus_r4[j] = inp3[j] - inp4[j];
        r4_minus_r2[j] = inp4[j] - inp2[j];
        r1_minus_r3[j] = inp1[j] - inp3[j];
    }

    for(int j = 0; j < 6; j++)
    {
        tmp[j] = 4 * inp0[j] - 5 * inp2[j] + inp4[j];
        tmp[6 + j] = r3_add_r4[j] - 4 * r1_add_r2[j];
        tmp[12 + j] = 4 * r1_minus_r2[j] - r3_minus_r4[j];
        tmp[18 + j] = r4_minus_r2[j] - 2 * r1_minus_r3[j];
        tmp[24 + j] = r4_minus_r2[j] + 2 * r1_minus_r3[j];
        tmp[30 + j] = 4 * inp1[j] - 5 * inp3[j] + inp5[j];
    }
    float r1_4_minus_r3[6];
    float r4_minus_4_r2[6];
    float r4_minus_r2_[6];
    float r1_minus_r3_x2[6];
    for(int j = 0; j < 6; j++)
    {
        r4_minus_r2_[j] = tmp[j * 6 + 4] - tmp[j * 6 + 2];
        r1_4_minus_r3[j] = 4 * tmp[j * 6 + 1] - tmp[j * 6 + 3];
        r4_minus_4_r2[j] = tmp[j * 6 + 4] - 4 * tmp[j * 6 + 2];
        r1_minus_r3_x2[j] = 2 * (tmp[j * 6 + 1] - tmp[j * 6 + 3]);
    }
    for(int j = 0; j < 6; j++)
    {
        inp_ptr[j * 6] = 4 * tmp[j * 6] - 5 * tmp[j * 6 + 2] + tmp[j * 6 + 4];
        inp_ptr[1 + j * 6] = r4_minus_4_r2[j] - r1_4_minus_r3[j];
        inp_ptr[2 + j * 6] = r4_minus_4_r2[j] + r1_4_minus_r3[j];
        inp_ptr[3 + j * 6] = r4_minus_r2_[j] - r1_minus_r3_x2[j];
        inp_ptr[4 + j * 6] = r4_minus_r2_[j] + r1_minus_r3_x2[j];
        inp_ptr[5 + j * 6] = 4 * tmp[j * 6 + 1] - 5 * tmp[j * 6 + 3] + tmp[j * 6 + 5];
    }
}

static inline void trans_inp_4_cpu(float* inp, float* inp_ptr, int c, int inw, int inc)
{
    float* inp0 = inp;
    float* inp1 = inp0 + inw;
    float* inp2 = inp1 + inw;
    float* inp3 = inp2 + inw;
    float* inp4 = inp3 + inw;
    float* inp5 = inp4 + inw;

    float mid[36 * 4] = {0};
    {
        float r4_minus_r2[24];
        float r1_4_minus_r3[24];
        float r4_minus_4_r2[24];
        float r1_minus_r3_x2[24];
        for(int i = 0; i < 6; i++)
        {
            // 0
            mid[i * 4] = 4 * inp0[i] - 5 * inp2[i] + inp4[i];
            mid[(30 + i) * 4] = 4 * inp1[i] - 5 * inp3[i] + inp5[i];

            r1_minus_r3_x2[i * 4 + 0] = (inp1[i] - inp3[i]) * 2;
            r1_4_minus_r3[i * 4 + 0] = 4 * inp1[i] - inp3[i];
            r4_minus_4_r2[i * 4 + 0] = inp4[i] - 4 * inp2[i];
            r4_minus_r2[i * 4 + 0] = inp4[i] - inp2[i];

            // 1
            mid[i * 4 + 1] = 4 * inp0[i + 4] - 5 * inp2[i + 4] + inp4[i + 4];
            mid[(30 + i) * 4 + 1] = 4 * inp1[i + 4] - 5 * inp3[i + 4] + inp5[i + 4];

            r1_minus_r3_x2[i * 4 + 1] = (inp1[i + 4] - inp3[i + 4]) * 2;
            r1_4_minus_r3[i * 4 + 1] = 4 * inp1[i + 4] - inp3[i + 4];
            r4_minus_4_r2[i * 4 + 1] = inp4[i + 4] - 4 * inp2[i + 4];
            r4_minus_r2[i * 4 + 1] = inp4[i + 4] - inp2[i + 4];

            // 2
            mid[i * 4 + 2] = 4 * inp0[i + 8] - 5 * inp2[i + 8] + inp4[i + 8];
            mid[(30 + i) * 4 + 2] = 4 * inp1[i + 8] - 5 * inp3[i + 8] + inp5[i + 8];

            r1_minus_r3_x2[i * 4 + 2] = (inp1[i + 8] - inp3[i + 8]) * 2;
            r1_4_minus_r3[i * 4 + 2] = 4 * inp1[i + 8] - inp3[i + 8];
            r4_minus_4_r2[i * 4 + 2] = inp4[i + 8] - 4 * inp2[i + 8];
            r4_minus_r2[i * 4 + 2] = inp4[i + 8] - inp2[i + 8];

            // 3
            mid[i * 4 + 3] = 4 * inp0[i + 12] - 5 * inp2[i + 12] + inp4[i + 12];
            mid[(30 + i) * 4 + 3] = 4 * inp1[i + 12] - 5 * inp3[i + 12] + inp5[i + 12];

            r1_minus_r3_x2[i * 4 + 3] = (inp1[i + 12] - inp3[i + 12]) * 2;
            r1_4_minus_r3[i * 4 + 3] = 4 * inp1[i + 12] - inp3[i + 12];
            r4_minus_4_r2[i * 4 + 3] = inp4[i + 12] - 4 * inp2[i + 12];
            r4_minus_r2[i * 4 + 3] = inp4[i + 12] - inp2[i + 12];
        }

        for(int i = 0; i < 6; i++)
        {
            for(int k = 0; k < 4; k++)
            {
                mid[(6 + i) * 4 + k] = r4_minus_4_r2[i * 4 + k] - r1_4_minus_r3[i * 4 + k];
                mid[(12 + i) * 4 + k] = r4_minus_4_r2[i * 4 + k] + r1_4_minus_r3[i * 4 + k];
                mid[(18 + i) * 4 + k] = r4_minus_r2[i * 4 + k] - r1_minus_r3_x2[i * 4 + k];
                mid[(24 + i) * 4 + k] = r4_minus_r2[i * 4 + k] + r1_minus_r3_x2[i * 4 + k];
            }
        }
    }

    {
        float r4_minus_r2[24];
        float r1_4_minus_r3[24];
        float r4_minus_4_r2[24];
        float r1_minus_r3_x2[24];
        for(int i = 0; i < 6; i++)
        {
            for(int k = 0; k < 4; k++)
            {
                r4_minus_r2[i * 4 + k] = mid[(i * 6 + 4) * 4 + k] - mid[(i * 6 + 2) * 4 + k];
                r1_4_minus_r3[i * 4 + k] = 4 * mid[(i * 6 + 1) * 4 + k] - mid[(i * 6 + 3) * 4 + k];
                r4_minus_4_r2[i * 4 + k] = mid[(i * 6 + 4) * 4 + k] - 4 * mid[(i * 6 + 2) * 4 + k];
                r1_minus_r3_x2[i * 4 + k] = 2 * (mid[(i * 6 + 1) * 4 + k] - mid[(i * 6 + 3) * 4 + k]);
            }
        }
        int inc_4 = inc * 4;
        for(int i = 0; i < 6; i++)
        {
            for(int k = 0; k < 4; k++)
            {
                //[36][inc][4]   36_i * cin*4 + c*4+k
                inp_ptr[k + c * 4 + inc_4 * (i * 6)] =
                    4 * mid[(i * 6) * 4 + k] - 5 * mid[(i * 6 + 2) * 4 + k] + mid[(i * 6 + 4) * 4 + k];
                inp_ptr[k + c * 4 + inc_4 * (1 + i * 6)] = r4_minus_4_r2[i * 4 + k] - r1_4_minus_r3[i * 4 + k];
                inp_ptr[k + c * 4 + inc_4 * (2 + i * 6)] = r4_minus_4_r2[i * 4 + k] + r1_4_minus_r3[i * 4 + k];
                inp_ptr[k + c * 4 + inc_4 * (3 + i * 6)] = r4_minus_r2[i * 4 + k] - r1_minus_r3_x2[i * 4 + k];
                inp_ptr[k + c * 4 + inc_4 * (4 + i * 6)] = r4_minus_r2[i * 4 + k] + r1_minus_r3_x2[i * 4 + k];
                inp_ptr[k + c * 4 + inc_4 * (5 + i * 6)] =
                    4 * mid[(i * 6 + 1) * 4 + k] - 5 * mid[(i * 6 + 3) * 4 + k] + mid[(i * 6 + 5) * 4 + k];
            }
        }
    }
}
static inline void tran_input_4block_func(const float* input, float* trans_inp, int inc, int nn_block0,
                                     int block_w, int in_hw, int inw)
{
    int idxh[4];
    int idxw[4];

        int ib = nn_block0;
        float* inp_ptr_4tile = trans_inp;
        idxh[0] = (ib * 4) / block_w;
        idxh[1] = (ib * 4 + 1) / block_w;
        idxh[2] = (ib * 4 + 2) / block_w;
        idxh[3] = (ib * 4 + 3) / block_w;
        idxw[0] = (ib * 4) % block_w;
        idxw[1] = (ib * 4 + 1) % block_w;
        idxw[2] = (ib * 4 + 2) % block_w;
        idxw[3] = (ib * 4 + 3) % block_w;

        if(idxh[0] == idxh[3])
        {
            float* temp_inp_ptr = ( float* )(input + idxh[0] * 4 * inw + idxw[0] * 4);

            for(int c = 0; c < inc; c++)
            {
                // trans_inp_4_cpu(temp_inp_ptr,inp_ptr_4tile,c,inw,inc);
                float ker00[4] = {1, 2, 4, 5};
                tran_inp_4(( const float* )temp_inp_ptr, ( const float* )inp_ptr_4tile + c * 4, ker00, inw, inc * 16,
                           in_hw);
                temp_inp_ptr += in_hw;
            }
        }
        else
        {
            float buffer0[inc * ELEM_SIZE * BLOCK_HW_UNIT];
            float* buffer = buffer0;

            for(int c = 0; c < inc; c++)
            {
                trans_inp_1tile(( float* )input, buffer, idxh[0], idxw[0], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( float* )input, buffer, idxh[1], idxw[1], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( float* )input, buffer, idxh[2], idxw[2], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( float* )input, buffer, idxh[3], idxw[3], c, in_hw, inw);
                buffer += ELEM_SIZE;
            }
            // interleave
            float* tmp_inp = inp_ptr_4tile;
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                for(int i = 0; i < inc; i++)
                {
                    for(int j = 0; j < BLOCK_HW_UNIT; j++)
                    {
                        *tmp_inp = buffer0[i * ELEM_SIZE * BLOCK_HW_UNIT + j * ELEM_SIZE + s];
                        tmp_inp++;
                    }
                }
            }
            // end interleave
        }
    
}

static inline void tran_input_4block(const float* input, float* trans_inp, int inc, int nn_block0, int nn_block,
                                     int block_w, int in_hw, int inw)
{
    int idxh[4];
    int idxw[4];

    for(int ib = nn_block0; ib < nn_block; ib++)
    {
        float* inp_ptr_4tile = trans_inp + ib * BLOCK_HW_UNIT * ELEM_SIZE * inc;
        idxh[0] = (ib * 4) / block_w;
        idxh[1] = (ib * 4 + 1) / block_w;
        idxh[2] = (ib * 4 + 2) / block_w;
        idxh[3] = (ib * 4 + 3) / block_w;
        idxw[0] = (ib * 4) % block_w;
        idxw[1] = (ib * 4 + 1) % block_w;
        idxw[2] = (ib * 4 + 2) % block_w;
        idxw[3] = (ib * 4 + 3) % block_w;

        if(idxh[0] == idxh[3])
        {
            float* temp_inp_ptr = ( float* )(input + idxh[0] * 4 * inw + idxw[0] * 4);

            for(int c = 0; c < inc; c++)
            {
                // trans_inp_4_cpu(temp_inp_ptr,inp_ptr_4tile,c,inw,inc);
                float ker00[4] = {1, 2, 4, 5};
                tran_inp_4(( const float* )temp_inp_ptr, ( const float* )inp_ptr_4tile + c * 4, ker00, inw, inc * 16,
                           in_hw);
                temp_inp_ptr += in_hw;
            }
        }
        else
        {
            float buffer0[inc * ELEM_SIZE * BLOCK_HW_UNIT];
            float* buffer = buffer0;

            for(int c = 0; c < inc; c++)
            {
                trans_inp_1tile(( float* )input, buffer, idxh[0], idxw[0], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( float* )input, buffer, idxh[1], idxw[1], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( float* )input, buffer, idxh[2], idxw[2], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( float* )input, buffer, idxh[3], idxw[3], c, in_hw, inw);
                buffer += ELEM_SIZE;
            }
            // interleave
            float* tmp_inp = inp_ptr_4tile;
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                for(int i = 0; i < inc; i++)
                {
                    for(int j = 0; j < BLOCK_HW_UNIT; j++)
                    {
                        *tmp_inp = buffer0[i * ELEM_SIZE * BLOCK_HW_UNIT + j * ELEM_SIZE + s];
                        tmp_inp++;
                    }
                }
            }
            // end interleave
        }
    }
}

// tran_inp [block_hw/4][36][inc][4] -> [36][block_hw/4][inc][4]
static inline void tran_input_4block_1(const float* input, float* trans_inp, int inc, int nn_block0, int nn_block,
                                       int block_w, int in_hw, int inw, int block_hw)
{
    int idxh[4];
    int idxw[4];
    int s_size = block_hw * inc * sizeof(float);
    for(int ib = nn_block0; ib < nn_block; ib++)
    {
        int off_set0 = ib * BLOCK_HW_UNIT * inc;
        idxh[0] = (ib * 4) / block_w;
        idxh[1] = (ib * 4 + 1) / block_w;
        idxh[2] = (ib * 4 + 2) / block_w;
        idxh[3] = (ib * 4 + 3) / block_w;
        idxw[0] = (ib * 4) % block_w;
        idxw[1] = (ib * 4 + 1) % block_w;
        idxw[2] = (ib * 4 + 2) % block_w;
        idxw[3] = (ib * 4 + 3) % block_w;

        if(idxh[0] == idxh[3])
        {
            float* temp_inp_ptr = ( float* )(input + idxh[0] * 4 * inw + idxw[0] * 4);
            for(int c = 0; c < inc; c++)
            {
                float ker00[4] = {1, 2, 4, 5};
                tran_inp_4(( const float* )temp_inp_ptr, ( const float* )trans_inp + c * 4 + off_set0, ker00, inw,
                           s_size, in_hw);
                temp_inp_ptr += in_hw;
            }
        }
        else
        {
            float buffer0[inc * ELEM_SIZE * BLOCK_HW_UNIT];
            float* buffer = buffer0;

            for(int c = 0; c < inc; c++)
            {
                trans_inp_1tile(( float* )input, buffer, idxh[0], idxw[0], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( float* )input, buffer, idxh[1], idxw[1], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( float* )input, buffer, idxh[2], idxw[2], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( float* )input, buffer, idxh[3], idxw[3], c, in_hw, inw);
                buffer += ELEM_SIZE;
            }
            // interleave
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                float* tmp_inp = trans_inp + s * block_hw * inc + off_set0;
                for(int i = 0; i < inc; i++)
                {
                    for(int j = 0; j < BLOCK_HW_UNIT; j++)
                    {
                        *tmp_inp = buffer0[i * ELEM_SIZE * BLOCK_HW_UNIT + j * ELEM_SIZE + s];
                        tmp_inp++;
                    }
                }
            }
            // end interleave
        }
    }
}

// tran_inp [block_resi][36][inc] -> [36][block_resi][inc]
static inline void tran_input_resi_block_1(const float* input, float* trans_inp, int inc, int nn_block, int resi_block,
                                           int block_hw, int block_w, int in_hw, int inw)
{
    for(int ib = resi_block; ib < block_hw; ib++)
    {
        int off_set0 = ib * inc;

        float buffer0[ELEM_SIZE * inc];
        float* buffer = buffer0;
        for(int c = 0; c < inc; c++)
        {
            int ih = ib / block_w;
            int jw = ib % block_w;
            trans_inp_1tile(( float* )input, buffer, ih, jw, c, in_hw, inw);
            buffer += ELEM_SIZE;
        }
        // interleave
        for(int s = 0; s < ELEM_SIZE; s++)
        {
            float* tmp_inp = trans_inp + s * block_hw * inc + off_set0;
            for(int i = 0; i < inc; i++)
            {
                *tmp_inp = buffer0[i * ELEM_SIZE + s];
                tmp_inp++;
            }
        }
        // end interleave
    }
}

// tran_inp [block_hw/4][36][inc][4] -> [36][block_hw/4][inc][4]
static inline void tran_input_1(const float* input, float* trans_inp, int inc, int inc_start, int inc_end,
                                int block_w, int in_hw, int inw, int block_hw)
{
    int idxh[4];
    int idxw[4];
    int s_size = block_hw * inc * sizeof(float);
    int ib;
    int c_length = inc_end - inc_start;
    for(ib = 0; ib < (block_hw &-4); ib+=4)
    {
        int off_set0 = ib * inc;
        idxh[0] = (ib) / block_w;
        idxh[1] = (ib + 1) / block_w;
        idxh[2] = (ib + 2) / block_w;
        idxh[3] = (ib + 3) / block_w;
        idxw[0] = (ib) % block_w;
        idxw[1] = (ib + 1) % block_w;
        idxw[2] = (ib + 2) % block_w;
        idxw[3] = (ib + 3) % block_w;

        if(idxh[0] == idxh[3])
        {
            float* temp_inp_ptr = ( float* )(input + idxh[0] * 4 * inw + idxw[0] * 4);
            for(int c = inc_start; c < inc_end; c++)
            {
                float ker00[4] = {1, 2, 4, 5};
                tran_inp_4(( const float* )(temp_inp_ptr+ c*in_hw), ( const float* )trans_inp + c * 4 + off_set0, ker00, inw,
                           s_size, in_hw);
            }
        }
        else
        {
            float buffer0[c_length * ELEM_SIZE * BLOCK_HW_UNIT];
            float* buffer = buffer0;

            for(int c = inc_start; c < inc_end; c++)
            {
                trans_inp_1tile(( float* )input, buffer, idxh[0], idxw[0], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( float* )input, buffer, idxh[1], idxw[1], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( float* )input, buffer, idxh[2], idxw[2], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( float* )input, buffer, idxh[3], idxw[3], c, in_hw, inw);
                buffer += ELEM_SIZE;
            }
            // interleave //[c-c][4][36]->[36][c][4]
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                float* tmp_inp = trans_inp + s * block_hw * inc + off_set0 + inc_start*4;
                for(int i = 0; i < c_length; i++)
                {
                    for(int j = 0; j < BLOCK_HW_UNIT; j++)
                    {
                        *tmp_inp = buffer0[i * ELEM_SIZE * BLOCK_HW_UNIT + j * ELEM_SIZE + s];
                        tmp_inp++;
                    }
                }
            }
            // end interleave    [36][block_hw/4][inc][4]
        }
    }
    for(ib=(block_hw & -4); ib < block_hw; ib++)
    {
        int off_set0 = ib * inc;
        float buffer0[ELEM_SIZE * c_length];
        float* buffer = buffer0;
        //[c-c][36]
        for(int c = inc_start; c < inc_end; c++)
        {
            int ih = ib / block_w;
            int jw = ib % block_w;
            trans_inp_1tile(( float* )input, buffer, ih, jw, c, in_hw, inw);
            buffer += ELEM_SIZE;
        }
        // interleave
        for(int s = 0; s < ELEM_SIZE; s++)
        {
            float* tmp_inp = trans_inp + s * block_hw * inc + off_set0 + inc_start;
            for(int i = 0; i < c_length; i++)
            {
                *tmp_inp = buffer0[i * ELEM_SIZE + s];
                tmp_inp++;
            }
        }
        // end interleave
    }
}
static inline void tran_input_resi_block(const float* input, float* trans_inp, int inc, int nn_block, int resi_block,
                                         int block_hw, int block_w, int in_hw, int inw)
{
    float* inp_ptr = trans_inp + nn_block * BLOCK_HW_UNIT * ELEM_SIZE * inc;
    for(int ib = resi_block; ib < block_hw; ib++)
    {
        float buffer0[ELEM_SIZE * inc];
        float* buffer = buffer0;
        for(int c = 0; c < inc; c++)
        {
            int ih = ib / block_w;
            int jw = ib % block_w;
            trans_inp_1tile(( float* )input, buffer, ih, jw, c, in_hw, inw);
            buffer += ELEM_SIZE;
        }
        // interleave
        for(int s = 0; s < ELEM_SIZE; s++)
        {
            for(int i = 0; i < inc; i++)
            {
                *inp_ptr = buffer0[i * ELEM_SIZE + s];
                inp_ptr++;
            }
        }
        // end interleave
    }
}
static inline void tran_input_resi_block_func(const float* input, float* trans_inp, int inc, int resi_block,
                                         int block_hw, int block_w, int in_hw, int inw)
{
    float* inp_ptr = trans_inp;

    for(int ib = resi_block; ib < block_hw; ib++)
    {
        float buffer0[ELEM_SIZE * inc];
        float* buffer = buffer0;
        for(int c = 0; c < inc; c++)
        {
            int ih = ib / block_w;
            int jw = ib % block_w;
            trans_inp_1tile(( float* )input, buffer, ih, jw, c, in_hw, inw);
            buffer += ELEM_SIZE;
        }
        // interleave
        for(int s = 0; s < ELEM_SIZE; s++)
        {
            for(int i = 0; i < inc; i++)
            {
                *inp_ptr = buffer0[i * ELEM_SIZE + s];
                inp_ptr++;
            }
        }
        // end interleave
    }
}
// pad input into 4n+2 and p_h, p_w
static inline void pad_input1(const float* input, float* inp_padded, int inc, int inh, int inw, int padded_h,
                              int padded_w, int pad0, int pad1)
{
    int padded_hw = padded_h * padded_w;

    float* pad_ptr;
    float* inp_ptr = ( float* )input;
    int resi_h = padded_h - pad0 - inh;
    int resi_w = padded_w - pad1 - inw;
    for(int c = 0; c < inc; c++)
    {
        pad_ptr = inp_padded + c * padded_hw;
        // pad h_top
        memset(pad_ptr, 0, padded_w * pad0 * sizeof(float));
        pad_ptr += pad0 * padded_w;
        // pad h_mid
        for(int h = 0; h < inh; h++)
        {
            // pad w_left
            memset(pad_ptr, 0, pad1 * sizeof(float));
            // pad w_mid
            memcpy(pad_ptr + pad1, inp_ptr, inw * sizeof(float));
            // pad w_end
            if(resi_w)
                memset(pad_ptr + pad1 + inw, 0, resi_w * sizeof(float));

            inp_ptr += inw;
            pad_ptr += padded_w;
        }
        // pad h_bottom
        if(resi_h)
            memset(pad_ptr, 0, padded_w * resi_h * sizeof(float));
    }
}
// input [HWC]
static inline void convert_pad_input(const float* input, float* inp_padded, int inc, int inh, int inw, int padded_h,
                                     int padded_w, int pad0, int pad1)
{
    int padded_hw = padded_h * padded_w;
    float* pad_ptr;
    int resi_h = padded_h - pad0 - inh;
    int resi_w = padded_w - pad1 - inw;
    for(int c = 0; c < inc; c++)
    {
        pad_ptr = inp_padded + c * padded_hw;
        // pad h_top
        memset(pad_ptr, 0, padded_w * pad0 * sizeof(float));
        pad_ptr += pad0 * padded_w;
        // pad h_mid
        for(int h = 0; h < inh; h++)
        {
            // pad w_left
            memset(pad_ptr, 0, pad1 * sizeof(float));
            // pad w_mid [input HWC]
            for(int w = 0; w < inw; w++)
            {
                pad_ptr[pad1 + w] = input[(h * inw + w) * inc + c];
            }
            // memcpy(pad_ptr + pad1, inp_ptr, inw * sizeof(float));
            // pad w_end
            if(resi_w)
                memset(pad_ptr + pad1 + inw, 0, resi_w * sizeof(float));
            pad_ptr += padded_w;
        }
        // pad h_bottom
        if(resi_h)
            memset(pad_ptr, 0, padded_w * resi_h * sizeof(float));
    }
}
#ifdef __cplusplus
}
#endif    //__cplusplus

#endif    // __WINO_TRANS_INP_H__