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

extern void wino_trans_inp4_fp16(const __fp16* inp, const __fp16* out, __fp16* ker, int inw, int inc4,int inhw);
// ------------------------------ INPUT ------------------------------------------
/*
    // up to 10 times as before, make it 2**n, => reduce 1./16 = 0.0625
    // out* 16*16 =*256
    __fp16 BT[36]={
        4.,  0., -5.,  0.,  1.,  0.,
        0., -4., -4.,  1.,  1.,  0.,
        0.,  4., -4., -1.,  1.,  0.,
        0., -2., -1.,  2.,  1.,  0.,
        0.,  2., -1., -2.,  1.,  0.,
        0.,  4.,  0., -5.,  0.,  1.
    };

    __fp16 B[36]={
        4.,  0.,  0.,  0.,  0.,  0.,
        0., -4.,  4., -2.,  2.,  4.,
        -5.,-4., -4., -1., -1.,  0.,
        0.,  1., -1.,  2., -2., -5.,
        1.,  1.,1.,  1.,  1.,  0.,
        0.,  0.,  0.,  0.,  0.,  1.
    };
*/
static inline void trans_inp_1tile(__fp16* input, __fp16* inp_ptr, int ih, int jw, int c, int in_hw, int inw)
{
    __fp16* inp = ( __fp16* )input + c * in_hw + ih * 4 * inw + jw * 4;
    __fp16* inp0 = inp;
    __fp16* inp1 = inp0 + inw;
    __fp16* inp2 = inp1 + inw;
    __fp16* inp3 = inp2 + inw;
    __fp16* inp4 = inp3 + inw;
    __fp16* inp5 = inp4 + inw;
    __fp16 tmp[36] = {0};

    __fp16 r1_add_r2[6];
    __fp16 r3_add_r4[6];
    __fp16 r1_minus_r2[6];
    __fp16 r3_minus_r4[6];
    __fp16 r4_minus_r2[6];
    __fp16 r1_minus_r3[6];

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
        tmp[j]      =0.0625*(4 * inp0[j] - 5 * inp2[j] + inp4[j]);
        tmp[6 + j]  =0.0625*(r3_add_r4[j] - 4 * r1_add_r2[j]    );
        tmp[12 + j] =0.0625*(4 * r1_minus_r2[j] - r3_minus_r4[j]);
        tmp[18 + j] =0.0625*(r4_minus_r2[j] - 2 * r1_minus_r3[j]);
        tmp[24 + j] =0.0625*(r4_minus_r2[j] + 2 * r1_minus_r3[j]);
        tmp[30 + j] =0.0625*(4 * inp1[j] - 5 * inp3[j] + inp5[j]);
    }
    __fp16 r1_4_minus_r3[6];
    __fp16 r4_minus_4_r2[6];
    __fp16 r4_minus_r2_[6];
    __fp16 r1_minus_r3_x2[6];
    for(int j = 0; j < 6; j++)
    {
        r4_minus_r2_[j] = tmp[j * 6 + 4] - tmp[j * 6 + 2];
        r1_4_minus_r3[j] = 4 * tmp[j * 6 + 1] - tmp[j * 6 + 3];
        r4_minus_4_r2[j] = tmp[j * 6 + 4] - 4 * tmp[j * 6 + 2];
        r1_minus_r3_x2[j] = 2 * (tmp[j * 6 + 1] - tmp[j * 6 + 3]);
    }
    for(int j = 0; j < 6; j++)
    {
        inp_ptr[j * 6]     =0.0625*(4 * tmp[j * 6] - 5 * tmp[j * 6 + 2] + tmp[j * 6 + 4]);
        inp_ptr[1 + j * 6] =0.0625*(r4_minus_4_r2[j] - r1_4_minus_r3[j]);
        inp_ptr[2 + j * 6] =0.0625*(r4_minus_4_r2[j] + r1_4_minus_r3[j]);
        inp_ptr[3 + j * 6] =0.0625*(r4_minus_r2_[j] - r1_minus_r3_x2[j]);
        inp_ptr[4 + j * 6] =0.0625*(r4_minus_r2_[j] + r1_minus_r3_x2[j]);
        inp_ptr[5 + j * 6] =0.0625*(4 * tmp[j * 6 + 1] - 5 * tmp[j * 6 + 3] + tmp[j * 6 + 5]);
    }
}

void vstr(__fp16* ptr,__fp16* v)
{
    for(int i=0;i<4;i++)
    {
        ptr[i]=v[i];
    }
}
void fadd(__fp16* v0,__fp16* v1,__fp16* v2)
{
    for(int i=0;i<4;i++)
    {
        v0[i]=v1[i]+v2[i];
    }
}
void fsub(__fp16* v0,__fp16* v1,__fp16* v2)
{
    for(int i=0;i<4;i++)
    {
        v0[i]=v1[i]-v2[i];
    }
}
void fmul(__fp16* v0,__fp16*v1,__fp16 s)
{
    for(int i=0;i<4;i++)
    {
        v0[i]=v1[i]*s;
    }
}
void vld4(__fp16* ptr,__fp16* v0,__fp16* v1,__fp16* v2,__fp16* v3)
{
    for(int i=0;i<4;i++)
    {
        v0[i]=ptr[i*4];
        v1[i]=ptr[i*4+1];
        v2[i]=ptr[i*4+2];
        v3[i]=ptr[i*4+3];
    }
}
void ld_end(__fp16* ptr,__fp16* v0,__fp16* v1)
{
    for(int i=0;i<4;i++)
    {
        v0[i]=ptr[4+i*4];
        v1[i]=ptr[4+i*4+1];
    }
}
void add_s(__fp16* v0,__fp16* v1,__fp16* v2,__fp16 s)
{
    for(int i=0;i<4;i++)
    {
        v0[i]=v1[i]+s*v2[i];
    }
}
void sub_s(__fp16* v0,__fp16* v1,__fp16* v2,__fp16 s)
{
    for(int i=0;i<4;i++)
    {
        v0[i]=v1[i]-s*v2[i];
    }
}
void rsub_s(__fp16* v0,__fp16* v1,__fp16* v2,__fp16 s)
{
    for(int i=0;i<4;i++)
    {
        v0[i]=s*v1[i]-v2[i];
    }
}
void fm_sub(__fp16* v0,__fp16* v1,__fp16* v2,__fp16 s)
{
    for(int i=0;i<4;i++)
    {
        v0[i]=(v1[i]-v2[i]);
        v0[i]=s*v0[i];
    }
}
void fmls(__fp16* v0,__fp16* v1,__fp16 s)
{
    for(int i=0;i<4;i++)
    {
        v0[i]=v0[i]-s*v1[i];
    }
}
void fmla(__fp16* v0,__fp16* v1,__fp16 s)
{
    for(int i=0;i<4;i++)
    {
        v0[i]=v0[i]+s*v1[i];
    }
}
void trans_inp_4_cpu(__fp16* inp, __fp16* out,  int inw, int inc)
{
    __fp16* inp0 = inp;
    __fp16* inp1 = inp0 + inw;
    __fp16* inp2 = inp1 + inw;
    __fp16* inp3 = inp2 + inw;
    __fp16* inp4 = inp3 + inw;
    __fp16* inp5 = inp4 + inw;
    __fp16 coef[4]={2,4,5,0.0625};
    __fp16 a[8]={0.0625,0.125,0.25,0.3125,1,2,4,5};//0.0625*{1,2,4,5};
    int x17= inc*4*6;
    int x4=inc*4;
    __fp16 mid[6][6][4];
    __fp16 r4_minus_r2[6][4];
    __fp16 r1_4_minus_r3[6][4];
    __fp16 r4_minus_4_r2[6][4];
    __fp16 r1_minus_r3_x2[6][4];
    
    __fp16 v[32][4];
    //===============load data ============
    //line1
    vld4( inp1,v[0],v[1],v[2],v[3]);
    ld_end( inp1,v[4],v[5]);
    //line2
    vld4( inp2,v[6],v[7],v[8],v[9]);
    ld_end( inp2,v[10],v[11]);
    //line3
    vld4( inp3,v[12],v[13],v[14],v[15]);
    ld_end( inp3,v[16],v[17]);
    //line4
    vld4( inp4,v[18],v[19],v[20],v[21]);
    ld_end( inp4,v[22],v[23]);

    for(int i=0;i<6;i++)
    {
        fmul(v[i+18],v[i+18],a[0]);
        fmul(v[i+12],v[i+12],a[0]);
        fmul(v[i+6],v[i+6],a[0]);
        fmul(v[i],v[i],a[0]);
    }
    for(int i=0;i<6;i++)
    {
        fadd(mid[1][i],v[i+12],v[i+18]);
        fmls(mid[1][i],v[i],a[6]);
        fmls(mid[1][i],v[i+6],a[6]);

        fsub(mid[2][i],v[i+18],v[i+12]);
        fmla(mid[2][i],v[i],a[6]);
        fmls(mid[2][i],v[i+6],a[6]);

        fsub(mid[3][i],v[i+18],v[i+6]);
        fmla(mid[3][i],v[i+12],a[5]);
        fmls(mid[3][i],v[i],a[5]);

        fsub(mid[4][i],v[i+18],v[i+6]);
        fmls(mid[4][i],v[i+12],a[5]);
        fmla(mid[4][i],v[i],a[5]);
    }
    for(int i = 0; i < 6; i++)
    {
        // 0
        mid[0][i][0]        = coef[3]*(coef[1] * inp0[i] - coef[2] * inp2[i] + inp4[i]);
        mid[5][i][0] = coef[3]*(coef[1] * inp1[i] - coef[2] * inp3[i] + inp5[i]);

        // 1
        mid[0][i][1] =        coef[3]*( coef[1] * inp0[i + 4] - coef[2] * inp2[i + 4] + inp4[i + 4]);
        mid[5][i][1] = coef[3]*(coef[1] * inp1[i + 4] - coef[2] * inp3[i + 4] + inp5[i + 4]);

        // 2
        mid[0][i][2] =       coef[3]*(coef[1] * inp0[i + 8] - coef[2] * inp2[i + 8] + inp4[i + 8]);
        mid[5][i][2] =coef[3]*( coef[1] * inp1[i + 8] - coef[2] * inp3[i + 8] + inp5[i + 8]);

        // 3
        mid[0][i][3] = coef[3]*(coef[1] * inp0[i + 12] - coef[2] * inp2[i + 12] + inp4[i + 12]);
        mid[5][i][3] = coef[3]*(coef[1] * inp1[i + 12] - coef[2] * inp3[i + 12] + inp5[i + 12]);
    }
    for(int i = 0; i < 6; i++)
    {
        fsub( r4_minus_r2[i], mid[i][4], mid[i][2]);
        rsub_s(r1_4_minus_r3[i],mid[i][1],mid[i][3],coef[1]);
        sub_s(r4_minus_4_r2[i],mid[i][4],mid[i][2],coef[1]);
        fm_sub(r1_minus_r3_x2[i],mid[i][1],mid[i][3],coef[0]);
    }
    for(int x=1;x<5;x++)
    {
        __fp16* po=out+ x17*x;
        fadd(v[10],mid[x][4],mid[x][3]);
        fmls(v[10],mid[x][1],a[6]);
        fmls(v[10],mid[x][2],a[6]);
        fmul(v[10],v[10],a[0]);
        vstr(po + x4  ,v[10]);

        fsub(v[10],mid[x][4],mid[x][3]);
        fmla(v[10],mid[x][1],a[6]);
        fmls(v[10],mid[x][2],a[6]);
        fmul(v[10],v[10],a[0]);
        vstr(po + x4*2  ,v[10]);

        fsub(v[10],mid[x][4],mid[x][2]);
        fmla(v[10],mid[x][3],a[5]);
        fmls(v[10],mid[x][1],a[5]);
        fmul(v[10],v[10],a[0]);
        vstr(po + x4*3  ,v[10]);

        fsub(v[10],mid[x][4],mid[x][2]);
        fmls(v[10],mid[x][3],a[5]);
        fmla(v[10],mid[x][1],a[5]);
        fmul(v[10],v[10],a[0]);
        vstr(po + x4*4  ,v[10]);

        for(int k = 0; k < 4; k++)
        {
            po[k ] =coef[3]*(
                coef[1] * mid[x][0][k] - coef[2] * mid[x][2][k] + mid[x][4][k]);
            po[k + x4*5] = 
                coef[3]*(coef[x] * mid[x][1][k] - coef[2] * mid[x][3][k] + mid[x][5][k]);
        }
    }
    for(int i = 0; i < 1; i++)
    {
        __fp16* po=out+ x17*i;
        fsub(v[0],r4_minus_4_r2[i],r1_4_minus_r3[i]);
        fadd(v[1],r4_minus_4_r2[i],r1_4_minus_r3[i]);
        fsub(v[2],r4_minus_r2[i],r1_minus_r3_x2[i]);
        fadd(v[3],r4_minus_r2[i],r1_minus_r3_x2[i]);


        fmul(v[0],v[0],coef[3]);
        fmul(v[1],v[1],coef[3]);
        fmul(v[2],v[2],coef[3]);
        fmul(v[3],v[3],coef[3]);

        vstr(po + x4  ,v[0]);
        vstr(po + x4*2,v[1]);
        vstr(po + x4*3,v[2]);
        vstr(po + x4*4,v[3]);

        for(int k = 0; k < 4; k++)
        {
            //[36][inc][4]   36_i * cin*4 + c*4+k
            po[k ] =coef[3]*(
                coef[1] * mid[i][0][k] - coef[2] * mid[i][2][k] + mid[i][4][k]);
            po[k + x4*5] = 
                coef[3]*(coef[1] * mid[i][1][k] - coef[2] * mid[i][3][k] + mid[i][5][k]);
        }
    }
    for(int i = 5; i < 6; i++)
    {
        __fp16* po=out+ x17*i;
        fsub(v[0],r4_minus_4_r2[i],r1_4_minus_r3[i]);
        fadd(v[1],r4_minus_4_r2[i],r1_4_minus_r3[i]);
        fsub(v[2],r4_minus_r2[i],r1_minus_r3_x2[i]);
        fadd(v[3],r4_minus_r2[i],r1_minus_r3_x2[i]);


        fmul(v[0],v[0],coef[3]);
        fmul(v[1],v[1],coef[3]);
        fmul(v[2],v[2],coef[3]);
        fmul(v[3],v[3],coef[3]);

        vstr(po + x4  ,v[0]);
        vstr(po + x4*2,v[1]);
        vstr(po + x4*3,v[2]);
        vstr(po + x4*4,v[3]);

        for(int k = 0; k < 4; k++)
        {
            //[36][inc][4]   36_i * cin*4 + c*4+k
            po[k ] =coef[3]*(
                coef[1] * mid[i][0][k] - coef[2] * mid[i][2][k] + mid[i][4][k]);
            po[k + x4*5] = 
                coef[3]*(coef[1] * mid[i][1][k] - coef[2] * mid[i][3][k] + mid[i][5][k]);
        }
    }
}

static void tran_input_4block(const __fp16* input, __fp16* trans_inp, int inc, int nn_block0, int nn_block,
                                     int block_w, int in_hw, int inw)
{
    int idxh[4];
    int idxw[4];

    for(int ib = nn_block0; ib < nn_block; ib++)
    {
        __fp16* inp_ptr_4tile = trans_inp + ib * BLOCK_HW_UNIT * ELEM_SIZE * inc;
        idxh[0] = (ib * 4) / block_w;
        idxh[3] = (ib * 4 + 3) / block_w;
        idxw[0] = (ib * 4) % block_w;
        if(idxh[0] == idxh[3])
        {
            __fp16* temp_inp_ptr = ( __fp16* )(input + idxh[0] * 4 * inw + idxw[0] * 4);

            for(int c = 0; c < inc; c++)
            {
                //trans_inp_4_cpu(temp_inp_ptr,inp_ptr_4tile+c*4,inw,inc);
                __fp16 ker00[8] = {0.0625,0.125,0.25,0.3125,1,2,4,5};//0.0625*{1,2,4,5};};
                wino_trans_inp4_fp16(( const __fp16* )temp_inp_ptr, 
                                     ( const __fp16* ) inp_ptr_4tile + c * 4, 
                                     ker00, 
                                     inw,
                                     inc*8,//inc*4*sizeof(__fp16)
                                     in_hw);
                temp_inp_ptr += in_hw;
            }
        }
        else
        {
            idxh[1] = (ib * 4 + 1) / block_w;
            idxh[2] = (ib * 4 + 2) / block_w;
            idxw[1] = (ib * 4 + 1) % block_w;
            idxw[2] = (ib * 4 + 2) % block_w;
            idxw[3] = (ib * 4 + 3) % block_w;
            __fp16 buffer0[inc * ELEM_SIZE * BLOCK_HW_UNIT];
            __fp16* buffer = buffer0;

            for(int c = 0; c < inc; c++)
            {
                trans_inp_1tile(( __fp16* )input, buffer, idxh[0], idxw[0], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( __fp16* )input, buffer, idxh[1], idxw[1], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( __fp16* )input, buffer, idxh[2], idxw[2], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( __fp16* )input, buffer, idxh[3], idxw[3], c, in_hw, inw);
                buffer += ELEM_SIZE;
            }
            // interleave 
            __fp16* tmp_inp = inp_ptr_4tile;
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

//tran_inp [block_hw/4][36][inc][4] -> [36][block_hw/4][inc][4]
static inline void tran_input_4block_1(const __fp16* input, __fp16* trans_inp, int inc, int nn_block0, int nn_block,
                                     int block_w, int in_hw, int inw,int block_hw)
{
    int idxh[4];
    int idxw[4];
    // int s_size = block_hw*inc*sizeof(__fp16);
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
        
        // if(idxh[0] == idxh[3])
        // {
        //     __fp16* temp_inp_ptr = ( __fp16* )(input + idxh[0] * 4 * inw + idxw[0] * 4);
        //     for(int c = 0; c < inc; c++)
        //     {
        //         __fp16 ker00[4] = {1, 2, 4, 5};
        //         tran_inp_4(( const __fp16* )temp_inp_ptr, 
        //                     ( const __fp16* )trans_inp + c * 4 + off_set0 , 
        //                     ker00, 
        //                     inw, 
        //                     s_size,in_hw);
        //         temp_inp_ptr += in_hw;
        //     }
        // }
        // else
        {
            __fp16 buffer0[inc * ELEM_SIZE * BLOCK_HW_UNIT];
            __fp16* buffer = buffer0;

            for(int c = 0; c < inc; c++)
            {
                trans_inp_1tile(( __fp16* )input, buffer, idxh[0], idxw[0], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( __fp16* )input, buffer, idxh[1], idxw[1], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( __fp16* )input, buffer, idxh[2], idxw[2], c, in_hw, inw);
                buffer += ELEM_SIZE;
                trans_inp_1tile(( __fp16* )input, buffer, idxh[3], idxw[3], c, in_hw, inw);
                buffer += ELEM_SIZE;
            }
            // interleave
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                __fp16* tmp_inp = trans_inp + s*block_hw*inc + off_set0;
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

//tran_inp [block_resi][36][inc] -> [36][block_resi][inc]
static inline void tran_input_resi_block_1(const __fp16* input, __fp16* trans_inp, int inc, int nn_block, int resi_block,
                                         int block_hw, int block_w, int in_hw, int inw)
{
    for(int ib = resi_block; ib < block_hw; ib++)
    {
        int off_set0 = ib * inc;

        __fp16 buffer0[ELEM_SIZE * inc];
        __fp16* buffer = buffer0;
        for(int c = 0; c < inc; c++)
        {
            int ih = ib / block_w;
            int jw = ib % block_w;
            trans_inp_1tile(( __fp16* )input, buffer, ih, jw, c, in_hw, inw);
            buffer += ELEM_SIZE;
        }
        // interleave
        for(int s = 0; s < ELEM_SIZE; s++)
        {
            __fp16* tmp_inp = trans_inp + s*block_hw*inc + off_set0;
            for(int i = 0; i < inc; i++)
            {
                *tmp_inp = buffer0[i * ELEM_SIZE + s];
                tmp_inp++;
            }
        }
        // end interleave
    }
}

static inline void tran_input_resi_block(const __fp16* input, __fp16* trans_inp, int inc, int nn_block, int resi_block,
                                         int block_hw, int block_w, int in_hw, int inw)
{
    __fp16* inp_ptr = trans_inp + nn_block * BLOCK_HW_UNIT * ELEM_SIZE * inc;
    for(int ib = resi_block; ib < block_hw; ib++)
    {
        __fp16 buffer0[ELEM_SIZE * inc];
        __fp16* buffer = buffer0;
        for(int c = 0; c < inc; c++)
        {
            int ih = ib / block_w;
            int jw = ib % block_w;
            trans_inp_1tile(( __fp16* )input, buffer, ih, jw, c, in_hw, inw);
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


static inline void pad_input1(const __fp16* input, __fp16* inp_padded, int inc, int inh, int inw, int padded_h,
                              int padded_w, int pad0, int pad1)
{
    int padded_hw = padded_h * padded_w;

    __fp16* pad_ptr;
    __fp16* inp_ptr = ( __fp16* )input;
    int resi_h = padded_h - pad0 - inh;
    int resi_w = padded_w - pad1 - inw;
    int pad_head = padded_w * pad0;
    for(int c = 0; c < inc; c++)
    {
        pad_ptr = inp_padded + c * padded_hw;

        // pad h_top
        for(int j=0;j<pad_head;j++)pad_ptr[j]=0;//memset(pad_ptr, 0, pad_head * sizeof(__fp16));
        pad_ptr += pad0 * padded_w;
    
        // pad h_mid
        for(int h = 0; h < inh; h++)
        {
            // pad w_left
            for(int j=0;j<pad1;j++)pad_ptr[j]=0; //memset(pad_ptr, 0, pad1 * sizeof(__fp16));

            // pad w_mid
            memcpy(pad_ptr + pad1, inp_ptr, inw * sizeof(__fp16));
            // pad w_end
            if(resi_w)
            {
                for(int j=0;j<resi_w;j++) pad_ptr[pad1 + inw + j]=0;
                //memset(pad_ptr + pad1 + inw, 0, resi_w * sizeof(__fp16));
            }
                
            inp_ptr += inw;
            pad_ptr += padded_w;
        }
        // pad h_bottom
        if(resi_h)
            memset(pad_ptr, 0, padded_w * resi_h*sizeof(__fp16));
    }
}

#ifdef __cplusplus
}
#endif    //__cplusplus

#endif    // __WINO_TRANS_INP_H__