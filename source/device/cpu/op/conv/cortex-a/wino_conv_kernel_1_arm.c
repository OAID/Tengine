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
 * Author: zhli@openailab.com
 */

#ifdef __aarch64__

#include "wino_conv_kernel_1_arm.h"

#include "api/c_api.h"
#include "utility/sys_port.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#include <arm_neon.h>


#define TILE 4
#define BLOCK_HW_UNIT 4
#define ELEM_SIZE ((TILE + 2) * (TILE + 2))

#define WINO_MAX(a, b) ((a) > (b) ? (a) : (b))
#define WINO_MIN(a, b) ((a) < (b) ? (a) : (b))

#define PER_OUT_CHAN 16
#define KER_COUT_UNIT 16
#define KER_COUT_UNIT4 4
void tran_inp_4(float*, float*, float*, int, int, int);
void wino_sgemm_4x16_A72(float* output, const float* input, const float* kernel, long cin, short stride_save);
void wino_sgemm_4x4_A72(float* output, const float* input, const float* kernel, long cin, short stride_save);
void wino_sgemm_1x16(float* output, const float* input, const float* kernel, long cin);
void wino_sgemm_1x4(float* output, const float* input, const float* kernel, long cin);
void tran_out_4(float*, float*, int, float*, float*, int);

#define INTERLEAVE_KERNEL_UNIT(cout_idx_p,cout_unit,cin,ker_src,ker_dst,ELEM_SIZE,i,j,s){          \
    for(i = 0; i < cin; i++){                                                                      \
        for(j = 0; j < cout_unit; j++){                                                            \
            *ker_dst = ker_src[((cout_idx_p + j) * cin + i) * ELEM_SIZE + s];                      \
             ker_dst++;                                                                            \
        }                                                                                          \
    }}

static inline void trans_kernel_f43(float* ker, float* trans_ker)
{
    /*
    float G[18]={
      1./4   ,     0.   ,     0.     ,
      -1./6  ,   -1./6  ,    -1./6   ,
      -1./6  ,    1./6  ,    -1./6   ,
      1./24  ,   1./12  ,    1./6    ,
      1./24  ,   -1./12 ,    1./6    ,
      0.     ,    0.    ,     1.
    };
    float GT[18]={
      1./4 ,  -1./6, -1./6 ,  1./24, 1./24 ,  0.,
      0.,     -1./6,  1./6 ,  1./12, -1./12 ,  0.,
      0.,     -1./6,  -1./6 ,  1./6, 1./6 ,  1.
    };
    */
    float tmp[18] = {0};

    float neg_r0_add_r2_x_1_6[6];    // (r0+r2)*1./6
    float r0_1_4_add_r2_x_1_6[6];    // (r0*1/4 + r2)*1./6
    float r1_1_6[6];    // r1*1/6
    float r1_1_12[6];    // r1*1/12
    float s_1_6 = 1. / 6.f;
    for (int j = 0; j < 3; j++)
    {
        neg_r0_add_r2_x_1_6[j] = -(ker[j] + ker[6 + j]) * s_1_6;
        r0_1_4_add_r2_x_1_6[j] = (ker[j] * 0.25 + ker[6 + j]) * s_1_6;
        r1_1_6[j] = ker[3 + j] * s_1_6;
        r1_1_12[j] = r1_1_6[j] * 0.5;
    }
    for (int j = 0; j < 3; j++)
    {
        tmp[j] = ker[j] * 0.25;
        tmp[3 + j] = -r1_1_6[j] + neg_r0_add_r2_x_1_6[j];
        tmp[6 + j] = r1_1_6[j] + neg_r0_add_r2_x_1_6[j];
        tmp[9 + j] = r1_1_12[j] + r0_1_4_add_r2_x_1_6[j];
        tmp[12 + j] = -r1_1_12[j] + r0_1_4_add_r2_x_1_6[j];
        tmp[15 + j] = ker[6 + j];
    }
    // gemm(6,3,3,G,ker,tmp); done
    int idx;
    for (int j = 0; j < 6; j++)
    {
        idx = j * 3;
        neg_r0_add_r2_x_1_6[j] = -(tmp[idx] + tmp[idx + 2]) * s_1_6;
        r0_1_4_add_r2_x_1_6[j] = (tmp[idx] * 0.25 + tmp[idx + 2]) * s_1_6;
        r1_1_6[j] = tmp[idx + 1] * s_1_6;
        r1_1_12[j] = r1_1_6[j] * 0.5;
    }

    for (int j = 0; j < 6; j++)
    {
        idx = j * 6;
        trans_ker[idx] = tmp[j * 3] * 0.25;
        trans_ker[idx + 1] = -r1_1_6[j] + neg_r0_add_r2_x_1_6[j];
        trans_ker[idx + 2] = r1_1_6[j] + neg_r0_add_r2_x_1_6[j];
        trans_ker[idx + 3] = r1_1_12[j] + r0_1_4_add_r2_x_1_6[j];
        trans_ker[idx + 4] = -r1_1_12[j] + r0_1_4_add_r2_x_1_6[j];
        trans_ker[idx + 5] = tmp[j * 3 + 2];
    }
    // gemm(6,6,3,tmp,GT,trans_ker); done
}

static inline void transform_kernel_f43_tile(struct tensor* filter, float* trans_ker)
{
    int outc = filter->dims[0];
    int inc = filter->dims[1];
    float* kernel = ( float* )filter->data;
    float* ker_ptr = trans_ker;

    for (int i = 0; i < outc; i++)
    {
        for (int j = 0; j < inc; j++)
        {
            trans_kernel_f43(( float* )(kernel + 9 * (j + i * inc)), ker_ptr);
            ker_ptr += ELEM_SIZE;
        }
    }
}

// ker0 [cout][cin][ELEM_SIZE]
// ker1 [ELEM_SIZE][cout//KER_COUT_UNIT][cin][KER_COUT_UNIT]
static inline void interleave_kernel_1(float* ker0, float* ker1, int cout, int cin)
{
    int i,j;
    float* ker1_ptr = ker1;
    for(int s = 0; s < ELEM_SIZE; s++)
    {
        int p;
        //cout 16
        for(p = 0; p < (cout& -KER_COUT_UNIT); p+=KER_COUT_UNIT){
            INTERLEAVE_KERNEL_UNIT(p,KER_COUT_UNIT,cin,ker0,ker1_ptr,ELEM_SIZE,i,j,s);
        }
        //cout 4
        for(p = (cout & -KER_COUT_UNIT); p < (cout & -KER_COUT_UNIT4); p += KER_COUT_UNIT4){
            INTERLEAVE_KERNEL_UNIT(p,KER_COUT_UNIT4,cin,ker0,ker1_ptr,ELEM_SIZE,i,j,s);
        }
        // cout 1
        for(p=(cout & -KER_COUT_UNIT4); p < cout; p ++){
            INTERLEAVE_KERNEL_UNIT(p,1,cin,ker0,ker1_ptr,ELEM_SIZE,i,j,s);
        }
    }
}

static inline void pad_input1(const float* input, float* inp_padded, int inc, int inh, int inw, int padded_h,
                              int padded_w, int pad0, int pad1)
{
    int padded_hw = padded_h * padded_w;

    float* pad_ptr;
    float* inp_ptr = ( float* )input;
    int resi_h = padded_h - pad0 - inh;
    int resi_w = padded_w - pad1 - inw;
    for (int c = 0; c < inc; c++)
    {
        pad_ptr = inp_padded + c * padded_hw;
        // pad h_top
        memset(pad_ptr, 0, padded_w * pad0 * sizeof(float));
        pad_ptr += pad0 * padded_w;
        // pad h_mid
        for (int h = 0; h < inh; h++)
        {
            // pad w_left
            memset(pad_ptr, 0, pad1 * sizeof(float));
            // pad w_mid
            memcpy(pad_ptr + pad1, inp_ptr, inw * sizeof(float));
            // pad w_end
            memset(pad_ptr + pad1 + inw, 0, resi_w * sizeof(float));

            inp_ptr += inw;
            pad_ptr += padded_w;
        }
        // pad h_bottom
        memset(pad_ptr, 0, padded_w * resi_h * sizeof(float));
    }
}

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

    for (int j = 0; j < 6; j++)
    {
        r1_add_r2[j] = inp1[j] + inp2[j];
        r1_minus_r2[j] = inp1[j] - inp2[j];
        r3_add_r4[j] = inp3[j] + inp4[j];
        r3_minus_r4[j] = inp3[j] - inp4[j];
        r4_minus_r2[j] = inp4[j] - inp2[j];
        r1_minus_r3[j] = inp1[j] - inp3[j];
    }

    for (int j = 0; j < 6; j++)
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
    for (int j = 0; j < 6; j++)
    {
        r4_minus_r2_[j] = tmp[j * 6 + 4] - tmp[j * 6 + 2];
        r1_4_minus_r3[j] = 4 * tmp[j * 6 + 1] - tmp[j * 6 + 3];
        r4_minus_4_r2[j] = tmp[j * 6 + 4] - 4 * tmp[j * 6 + 2];
        r1_minus_r3_x2[j] = 2 * (tmp[j * 6 + 1] - tmp[j * 6 + 3]);
    }
    for (int j = 0; j < 6; j++)
    {
        inp_ptr[j * 6] = 4 * tmp[j * 6] - 5 * tmp[j * 6 + 2] + tmp[j * 6 + 4];
        inp_ptr[1 + j * 6] = r4_minus_4_r2[j] - r1_4_minus_r3[j];
        inp_ptr[2 + j * 6] = r4_minus_4_r2[j] + r1_4_minus_r3[j];
        inp_ptr[3 + j * 6] = r4_minus_r2_[j] - r1_minus_r3_x2[j];
        inp_ptr[4 + j * 6] = r4_minus_r2_[j] + r1_minus_r3_x2[j];
        inp_ptr[5 + j * 6] = 4 * tmp[j * 6 + 1] - 5 * tmp[j * 6 + 3] + tmp[j * 6 + 5];
    }
}

static inline void trans_inp_4_cpu(float* inp, float* inp_ptr, int inw, int s_size)
{
    float* inp0 = inp;
    float* inp1 = inp0 + inw;
    float* inp2 = inp1 + inw;
    float* inp3 = inp2 + inw;
    float* inp4 = inp3 + inw;
    float* inp5 = inp4 + inw;

    float mid[36 * 4] = {0};

    float r4_minus_r2[24];
    float r1_4_minus_r3[24];
    float r4_minus_4_r2[24];
    float r1_minus_r3_x2[24];
    for (int i = 0; i < 6; i++)
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
    //====================================================================
    // for(int i = 0; i < 6; i++)
    // {
    //     for(int k = 0; k < 4; k++)
    //     {
    //         mid[(6 + i) * 4 + k] = r4_minus_4_r2[i * 4 + k] - r1_4_minus_r3[i * 4 + k];
    //         mid[(12 + i) * 4 + k] = r4_minus_4_r2[i * 4 + k] + r1_4_minus_r3[i * 4 + k];
    //         mid[(18 + i) * 4 + k] = r4_minus_r2[i * 4 + k] - r1_minus_r3_x2[i * 4 + k];
    //         mid[(24 + i) * 4 + k] = r4_minus_r2[i * 4 + k] + r1_minus_r3_x2[i * 4 + k];
    //     }
    // }
    float32x4_t r0 = vld1q_f32(r4_minus_4_r2);
    float32x4_t r1 = vld1q_f32(r4_minus_4_r2 + 4);
    float32x4_t r2 = vld1q_f32(r4_minus_4_r2 + 8);
    float32x4_t r3 = vld1q_f32(r4_minus_4_r2 + 12);
    float32x4_t r4 = vld1q_f32(r4_minus_4_r2 + 16);
    float32x4_t r5 = vld1q_f32(r4_minus_4_r2 + 20);

    float32x4_t r0_ = vld1q_f32(r1_4_minus_r3);
    float32x4_t r1_ = vld1q_f32(r1_4_minus_r3 + 4);
    float32x4_t r2_ = vld1q_f32(r1_4_minus_r3 + 8);
    float32x4_t r3_ = vld1q_f32(r1_4_minus_r3 + 12);
    float32x4_t r4_ = vld1q_f32(r1_4_minus_r3 + 16);
    float32x4_t r5_ = vld1q_f32(r1_4_minus_r3 + 20);

    float32x4_t line0_0 = vld1q_f32(mid);
    float32x4_t line0_1 = vld1q_f32(mid + 4);
    float32x4_t line0_2 = vld1q_f32(mid + 8);
    float32x4_t line0_3 = vld1q_f32(mid + 12);
    float32x4_t line0_4 = vld1q_f32(mid + 16);
    float32x4_t line0_5 = vld1q_f32(mid + 20);

    float32x4_t line1_0 = vsubq_f32(r0, r0_);    // mid[(6 + i) * 4 + k]   [1][0]
    float32x4_t line1_1 = vsubq_f32(r1, r1_);    // mid[(6 + i) * 4 + k]   [1][1]
    float32x4_t line1_2 = vsubq_f32(r2, r2_);    // mid[(6 + i) * 4 + k]   [1][2]
    float32x4_t line1_3 = vsubq_f32(r3, r3_);    // mid[(6 + i) * 4 + k]   [1][3]
    float32x4_t line1_4 = vsubq_f32(r4, r4_);    // mid[(6 + i) * 4 + k]   [1][4]
    float32x4_t line1_5 = vsubq_f32(r5, r5_);    // mid[(6 + i) * 4 + k]   [1][5]

    float32x4_t line2_0 = vaddq_f32(r0, r0_);    // mid[(12 + i) * 4 + k]  [2][0]
    float32x4_t line2_1 = vaddq_f32(r1, r1_);    // mid[(12 + i) * 4 + k]  [2][1]
    float32x4_t line2_2 = vaddq_f32(r2, r2_);    // mid[(12 + i) * 4 + k]  [2][2]
    float32x4_t line2_3 = vaddq_f32(r3, r3_);    // mid[(12 + i) * 4 + k]  [2][3]
    float32x4_t line2_4 = vaddq_f32(r4, r4_);    // mid[(12 + i) * 4 + k]  [2][4]
    float32x4_t line2_5 = vaddq_f32(r5, r5_);    // mid[(12 + i) * 4 + k]  [2][5]

    r0 = vld1q_f32(r4_minus_r2);
    r1 = vld1q_f32(r4_minus_r2 + 4);
    r2 = vld1q_f32(r4_minus_r2 + 8);
    r3 = vld1q_f32(r4_minus_r2 + 12);
    r4 = vld1q_f32(r4_minus_r2 + 16);
    r5 = vld1q_f32(r4_minus_r2 + 20);

    r0_ = vld1q_f32(r1_minus_r3_x2);
    r1_ = vld1q_f32(r1_minus_r3_x2 + 4);
    r2_ = vld1q_f32(r1_minus_r3_x2 + 8);
    r3_ = vld1q_f32(r1_minus_r3_x2 + 12);
    r4_ = vld1q_f32(r1_minus_r3_x2 + 16);
    r5_ = vld1q_f32(r1_minus_r3_x2 + 20);

    float32x4_t line5_0 = vld1q_f32(mid + 120);
    float32x4_t line5_1 = vld1q_f32(mid + 124);
    float32x4_t line5_2 = vld1q_f32(mid + 128);
    float32x4_t line5_3 = vld1q_f32(mid + 132);
    float32x4_t line5_4 = vld1q_f32(mid + 136);
    float32x4_t line5_5 = vld1q_f32(mid + 140);

    float32x4_t line3_0 = vsubq_f32(r0, r0_);    // mid[(18 + i) * 4 + k]   [3][0]
    float32x4_t line3_1 = vsubq_f32(r1, r1_);    // mid[(18 + i) * 4 + k]   [3][1]
    float32x4_t line3_2 = vsubq_f32(r2, r2_);    // mid[(18 + i) * 4 + k]   [3][2]
    float32x4_t line3_3 = vsubq_f32(r3, r3_);    // mid[(18 + i) * 4 + k]   [3][3]
    float32x4_t line3_4 = vsubq_f32(r4, r4_);    // mid[(18 + i) * 4 + k]   [3][4]
    float32x4_t line3_5 = vsubq_f32(r5, r5_);    // mid[(18 + i) * 4 + k]   [3][5]

    float32x4_t line4_0 = vaddq_f32(r0, r0_);    // mid[(24 + i) * 4 + k]  [4][0]
    float32x4_t line4_1 = vaddq_f32(r1, r1_);    // mid[(24 + i) * 4 + k]  [4][1]
    float32x4_t line4_2 = vaddq_f32(r2, r2_);    // mid[(24 + i) * 4 + k]  [4][2]
    float32x4_t line4_3 = vaddq_f32(r3, r3_);    // mid[(24 + i) * 4 + k]  [4][3]
    float32x4_t line4_4 = vaddq_f32(r4, r4_);    // mid[(24 + i) * 4 + k]  [4][4]
    float32x4_t line4_5 = vaddq_f32(r5, r5_);    // mid[(24 + i) * 4 + k]  [4][5]

    // r4_minus_r2[i * 4 + k]   i=0     = mid[0][4]
    r0 = vsubq_f32(line0_4, line0_2);
    r1 = vsubq_f32(line1_4, line1_2);
    r2 = vsubq_f32(line2_4, line2_2);
    r3 = vsubq_f32(line3_4, line3_2);
    r4 = vsubq_f32(line4_4, line4_2);
    r5 = vsubq_f32(line5_4, line5_2);

    r0_ = vsubq_f32(line0_1, line0_3);
    r1_ = vsubq_f32(line1_1, line1_3);
    r2_ = vsubq_f32(line2_1, line2_3);
    r3_ = vsubq_f32(line3_1, line3_3);
    r4_ = vsubq_f32(line4_1, line4_3);
    r5_ = vsubq_f32(line5_1, line5_3);

    float32x4_t const2 = vdupq_n_f32(2.f);
    r0_ = vmulq_f32(r0_, const2);
    r1_ = vmulq_f32(r1_, const2);
    r2_ = vmulq_f32(r2_, const2);
    r3_ = vmulq_f32(r3_, const2);
    r4_ = vmulq_f32(r4_, const2);
    r5_ = vmulq_f32(r5_, const2);

    vst1q_f32(inp_ptr + s_size * 3, vsubq_f32(r0, r0_));    // inp_ptr[ s_size * (3 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 9, vsubq_f32(r1, r1_));    // inp_ptr[ s_size * (3 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 15, vsubq_f32(r2, r2_));    // inp_ptr[ s_size * (3 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 21, vsubq_f32(r3, r3_));    // inp_ptr[ s_size * (3 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 27, vsubq_f32(r4, r4_));    // inp_ptr[ s_size * (3 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 33, vsubq_f32(r5, r5_));    // inp_ptr[ s_size * (3 + i * 6)]

    vst1q_f32(inp_ptr + s_size * 4, vaddq_f32(r0, r0_));    // inp_ptr[ s_size * (4 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 10, vaddq_f32(r1, r1_));    // inp_ptr[ s_size * (4 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 16, vaddq_f32(r2, r2_));    // inp_ptr[ s_size * (4 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 22, vaddq_f32(r3, r3_));    // inp_ptr[ s_size * (4 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 28, vaddq_f32(r4, r4_));    // inp_ptr[ s_size * (4 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 34, vaddq_f32(r5, r5_));    // inp_ptr[ s_size * (4 + i * 6)]

    float32x4_t const4 = vdupq_n_f32(4.f);
    float32x4_t const5 = vdupq_n_f32(-5.f);
    r0_ = vmulq_f32(line0_1, const4);    // line 1*4 ========
    r1_ = vmulq_f32(line1_1, const4);
    r2_ = vmulq_f32(line2_1, const4);
    r3_ = vmulq_f32(line3_1, const4);
    r4_ = vmulq_f32(line4_1, const4);
    r5_ = vmulq_f32(line5_1, const4);

    float32x4_t rr0_ = vsubq_f32(r0_, line0_3);    // line1*4-line3
    float32x4_t rr1_ = vsubq_f32(r1_, line1_3);
    float32x4_t rr2_ = vsubq_f32(r2_, line2_3);
    float32x4_t rr3_ = vsubq_f32(r3_, line3_3);
    float32x4_t rr4_ = vsubq_f32(r4_, line4_3);
    float32x4_t rr5_ = vsubq_f32(r5_, line5_3);

    r0 = vmulq_f32(line0_2, const4);
    r1 = vmulq_f32(line1_2, const4);
    r2 = vmulq_f32(line2_2, const4);
    r3 = vmulq_f32(line3_2, const4);
    r4 = vmulq_f32(line4_2, const4);
    r5 = vmulq_f32(line5_2, const4);

    r0 = vsubq_f32(line0_4, r0);    // line4 -4*line2
    r1 = vsubq_f32(line1_4, r1);
    r2 = vsubq_f32(line2_4, r2);
    r3 = vsubq_f32(line3_4, r3);
    r4 = vsubq_f32(line4_4, r4);
    r5 = vsubq_f32(line5_4, r5);

    vst1q_f32(inp_ptr + s_size * 1, vsubq_f32(r0, rr0_));    // inp_ptr[ s_size * (1 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 7, vsubq_f32(r1, rr1_));    // inp_ptr[ s_size * (1 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 13, vsubq_f32(r2, rr2_));    // inp_ptr[ s_size * (1 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 19, vsubq_f32(r3, rr3_));    // inp_ptr[ s_size * (1 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 25, vsubq_f32(r4, rr4_));    // inp_ptr[ s_size * (1 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 31, vsubq_f32(r5, rr5_));    // inp_ptr[ s_size * (1 + i * 6)]

    vst1q_f32(inp_ptr + s_size * 2, vaddq_f32(r0, rr0_));    // inp_ptr[ s_size * (2 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 8, vaddq_f32(r1, rr1_));    // inp_ptr[ s_size * (2 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 14, vaddq_f32(r2, rr2_));    // inp_ptr[ s_size * (2 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 20, vaddq_f32(r3, rr3_));    // inp_ptr[ s_size * (2 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 26, vaddq_f32(r4, rr4_));    // inp_ptr[ s_size * (2 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 32, vaddq_f32(r5, rr5_));    // inp_ptr[ s_size * (2 + i * 6)]

    r0_ = vaddq_f32(line0_5, r0_);    // 5 + 1*4
    r1_ = vaddq_f32(line1_5, r1_);
    r2_ = vaddq_f32(line2_5, r2_);
    r3_ = vaddq_f32(line3_5, r3_);
    r4_ = vaddq_f32(line4_5, r4_);
    r5_ = vaddq_f32(line5_5, r5_);

    r0 = vmulq_f32(line0_3, const5);
    r1 = vmulq_f32(line1_3, const5);
    r2 = vmulq_f32(line2_3, const5);
    r3 = vmulq_f32(line3_3, const5);
    r4 = vmulq_f32(line4_3, const5);
    r5 = vmulq_f32(line5_3, const5);
    vst1q_f32(inp_ptr + s_size * 5, vaddq_f32(r0, r0_));    // inp_ptr[ s_size * (5 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 11, vaddq_f32(r1, r1_));    // inp_ptr[ s_size * (5 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 17, vaddq_f32(r2, r2_));    // inp_ptr[ s_size * (5 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 23, vaddq_f32(r3, r3_));    // inp_ptr[ s_size * (5 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 29, vaddq_f32(r4, r4_));    // inp_ptr[ s_size * (5 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 35, vaddq_f32(r5, r5_));    // inp_ptr[ s_size * (5 + i * 6)]

    r0 = vmulq_f32(line0_0, const4);
    r1 = vmulq_f32(line1_0, const4);
    r2 = vmulq_f32(line2_0, const4);
    r3 = vmulq_f32(line3_0, const4);
    r4 = vmulq_f32(line4_0, const4);
    r5 = vmulq_f32(line5_0, const4);

    r0_ = vmulq_f32(line0_2, const5);
    r1_ = vmulq_f32(line1_2, const5);
    r2_ = vmulq_f32(line2_2, const5);
    r3_ = vmulq_f32(line3_2, const5);
    r4_ = vmulq_f32(line4_2, const5);
    r5_ = vmulq_f32(line5_2, const5);

    r0 = vaddq_f32(r0, line0_4);
    r1 = vaddq_f32(r1, line1_4);
    r2 = vaddq_f32(r2, line2_4);
    r3 = vaddq_f32(r3, line3_4);
    r4 = vaddq_f32(r4, line4_4);
    r5 = vaddq_f32(r5, line5_4);

    vst1q_f32(inp_ptr + s_size * 0, vaddq_f32(r0, r0_));    // inp_ptr[ s_size * (1 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 6, vaddq_f32(r1, r1_));    // inp_ptr[ s_size * (1 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 12, vaddq_f32(r2, r2_));    // inp_ptr[ s_size * (1 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 18, vaddq_f32(r3, r3_));    // inp_ptr[ s_size * (1 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 24, vaddq_f32(r4, r4_));    // inp_ptr[ s_size * (1 + i * 6)]
    vst1q_f32(inp_ptr + s_size * 30, vaddq_f32(r5, r5_));    // inp_ptr[ s_size * (1 + i * 6)]

    // for(int i = 0; i < 6; i++)
    // {
    //     for(int k = 0; k < 4; k++)
    //     {
    //         r4_minus_r2[i * 4 + k] = mid[(i * 6 + 4) * 4 + k] - mid[(i * 6 + 2) * 4 + k];
    //         r1_4_minus_r3[i * 4 + k] = 4 * mid[(i * 6 + 1) * 4 + k] - mid[(i * 6 + 3) * 4 + k];
    //         r4_minus_4_r2[i * 4 + k] = mid[(i * 6 + 4) * 4 + k] - 4 * mid[(i * 6 + 2) * 4 + k];
    //         r1_minus_r3_x2[i * 4 + k] = 2 * (mid[(i * 6 + 1) * 4 + k] - mid[(i * 6 + 3) * 4 + k]);
    //     }
    // }

    // for(int i = 1; i < 2; i++)
    // {

    //     for(int k = 0; k < 4; k++)
    //     {
    //         inp_ptr[k  + s_size * (i * 6)] =
    //             4 * mid[(i * 6) * 4 + k] - 5 * mid[(i * 6 + 2) * 4 + k] + mid[(i * 6 + 4) * 4 + k];
    // //         // inp_ptr[k + s_size * (1 + i * 6)] = r4_minus_4_r2[i * 4 + k] - r1_4_minus_r3[i * 4 + k];
    // //         // inp_ptr[k + s_size * (2 + i * 6)] = r4_minus_4_r2[i * 4 + k] + r1_4_minus_r3[i * 4 + k];
    // //         // inp_ptr[k + s_size * (3 + i * 6)] = r4_minus_r2[i * 4 + k] - r1_minus_r3_x2[i * 4 + k];
    // //         // inp_ptr[k + s_size * (4 + i * 6)] = r4_minus_r2[i * 4 + k] + r1_minus_r3_x2[i * 4 + k];
    // //         // inp_ptr[k + s_size * (5 + i * 6)] =
    // //         //     4 * mid[(i * 6 + 1) * 4 + k] - 5 * mid[(i * 6 + 3) * 4 + k] + mid[(i * 6 + 5) * 4 + k];
    //     }
    // }
}


// trans_input  [block_hw/4][ELEM_SIZE][inc][4]
static inline void tran_input_4block(const float* input, float* trans_inp, int inc, int block_h,
                                         int block_w, int inh, int inw)
{
    int in_hw = inh * inw;
    int block_hw = block_h * block_w;
    int nn_block = block_hw >> 2;
    int idxh[4];
    int idxw[4];

    for (int ib = 0; ib < nn_block; ib++)
    {
        float* inp_ptr_4tile = trans_inp + ib * 4 * ELEM_SIZE * inc;
        idxh[0] = (ib * 4) / block_w;
        idxh[1] = (ib * 4 + 1) / block_w;
        idxh[2] = (ib * 4 + 2) / block_w;
        idxh[3] = (ib * 4 + 3) / block_w;
        idxw[0] = (ib * 4) % block_w;
        idxw[1] = (ib * 4 + 1) % block_w;
        idxw[2] = (ib * 4 + 2) % block_w;
        idxw[3] = (ib * 4 + 3) % block_w;

        if (idxh[0] == idxh[3])
        {
            float* temp_inp_ptr = ( float* )(input + idxh[0] * 4 * inw + idxw[0] * 4);
            for (int c = 0; c < inc; c++)
            {
                float ker00[4] = {1, 2, 4, 5};
                tran_inp_4(temp_inp_ptr, inp_ptr_4tile + 4 * c, ker00, inw, inc * 16, in_hw);
                temp_inp_ptr += in_hw;
            }
        }
        else
        {
            float buffer0[inc * ELEM_SIZE * 4];
            float* buffer = buffer0;

            for (int c = 0; c < inc; c++)
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
            for (int s = 0; s < ELEM_SIZE; s++)
            {
                for (int i = 0; i < inc; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        *tmp_inp = buffer0[i * ELEM_SIZE * 4 + j * ELEM_SIZE + s];
                        tmp_inp++;
                    }
                }
            }
            // end interleave
        }
    }
}

// tran_inp [block_hw/4][36][inc][4] -> [36][block_hw/4][inc][4]
static inline void tran_input_4block_1(const float* input, float* trans_inp, int inc, int block_h, int block_w, int inh,
                                       int inw,int num_thread)
{
    int in_hw = inh * inw;
    int block_hw = block_h * block_w;
    int nn_block = block_hw >> 2;
    int idxh[4];
    int idxw[4];

    int s_size = block_hw * inc * sizeof(float);

#pragma omp parallel for num_threads(num_thread) shared(block_hw,nn_block,in_hw) private(idxh,idxw)
    for(int ib = 0; ib < nn_block; ib++)
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
                tran_inp_4(temp_inp_ptr, trans_inp + c * 4 + off_set0, ker00, inw, s_size, in_hw);
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

static inline void tran_input_resi_block(const float* input, float* trans_inp, int inc, int nn_block, int resi_block,
                                         int block_hw, int block_w, int in_hw, int inw)
{
    float* inp_ptr = trans_inp + nn_block * 4 * ELEM_SIZE * inc;
    for (int ib = resi_block; ib < block_hw; ib++)
    {
        float buffer0[ELEM_SIZE * inc];
        float* buffer = buffer0;
        for (int c = 0; c < inc; c++)
        {
            int ih = ib / block_w;
            int jw = ib % block_w;
            trans_inp_1tile(( float* )input, buffer, ih, jw, c, in_hw, inw);
            buffer += ELEM_SIZE;
        }
        // interleave
        for (int s = 0; s < ELEM_SIZE; s++)
        {
            for (int i = 0; i < inc; i++)
            {
                *inp_ptr = buffer0[i * ELEM_SIZE + s];
                inp_ptr++;
            }
        }
        // end interleave
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


static inline float do_activation(float value, int activation)
{
    if (activation >= 0)
        value = WINO_MAX(value, 0);
    if (activation == 6)
        value = WINO_MIN(value, 6);

    return value;
}

static inline void trans_output_f43(const float* mid, float* out, int outw, const float* bias_ptr, int activation)
{
    /*
   float AT[24]={
     1.,  1.,  1.,  1.,  1.,  0.,
     0.,  1., -1.,  2., -2.,  0.,
     0.,  1.,  1.,  4.,  4.,  0.,
     0.,  1., -1.,  8., -8.,  1.
   };


   float A[24]={
     1.,  0.,  0.,  0.,
     1.,  1.,  1.,  1.,
     1., -1.,  1., -1.,
     1.,  2.,  4.,  8.,
     1., -2.,  4., -8.,
     0.,  0.,  0.,  1.
   };
   */
    float tmp[24] = {0};

    float r1_add_r2[6];
    float r1_minus_r2[6];
    float r3_add_r4[6];
    float r3_minus_r4_x2[6];

    for (int j = 0; j < 6; j++)
    {
        r1_add_r2[j] = mid[6 * 1 + j] + mid[6 * 2 + j];
        r1_minus_r2[j] = mid[6 * 1 + j] - mid[6 * 2 + j];
        r3_add_r4[j] = mid[6 * 3 + j] + mid[6 * 4 + j];
        r3_minus_r4_x2[j] = (mid[6 * 3 + j] - mid[6 * 4 + j]) * 2;
    }
    for (int j = 0; j < 6; j++)
    {
        tmp[j] = mid[j] + r1_add_r2[j] + r3_add_r4[j];
        tmp[6 + j] = r1_minus_r2[j] + r3_minus_r4_x2[j];
        tmp[12 + j] = r1_add_r2[j] + 4 * r3_add_r4[j];
        tmp[18 + j] = r1_minus_r2[j] + 4 * r3_minus_r4_x2[j] + mid[6 * 5 + j];
    }

    float* out0 = out;
    float* out1 = out0 + outw;
    float* out2 = out1 + outw;
    float* out3 = out2 + outw;

    float _r1_add_r2[4];
    float _r1_minus_r2[4];
    float _r3_add_r4[4];
    float _r3_minus_r4_x2[4];
    int idx;
    for (int j = 0; j < 4; j++)
    {
        idx = 6 * j;
        _r1_add_r2[j] = tmp[idx + 1] + tmp[idx + 2];
        _r1_minus_r2[j] = tmp[idx + 1] - tmp[idx + 2];
        _r3_add_r4[j] = tmp[idx + 3] + tmp[idx + 4];
        _r3_minus_r4_x2[j] = (tmp[idx + 3] - tmp[idx + 4]) * 2;
    }
    if (bias_ptr)
    {
        float bias = bias_ptr[0];
        out0[0] = do_activation(tmp[0 * 6] + _r1_add_r2[0] + _r3_add_r4[0] + bias, activation);
        out1[0] = do_activation(tmp[1 * 6] + _r1_add_r2[1] + _r3_add_r4[1] + bias, activation);
        out2[0] = do_activation(tmp[2 * 6] + _r1_add_r2[2] + _r3_add_r4[2] + bias, activation);
        out3[0] = do_activation(tmp[3 * 6] + _r1_add_r2[3] + _r3_add_r4[3] + bias, activation);

        out0[1] = do_activation(_r1_minus_r2[0] + _r3_minus_r4_x2[0] + bias, activation);
        out1[1] = do_activation(_r1_minus_r2[1] + _r3_minus_r4_x2[1] + bias, activation);
        out2[1] = do_activation(_r1_minus_r2[2] + _r3_minus_r4_x2[2] + bias, activation);
        out3[1] = do_activation(_r1_minus_r2[3] + _r3_minus_r4_x2[3] + bias, activation);

        out0[2] = do_activation(_r1_add_r2[0] + 4 * _r3_add_r4[0] + bias, activation);
        out1[2] = do_activation(_r1_add_r2[1] + 4 * _r3_add_r4[1] + bias, activation);
        out2[2] = do_activation(_r1_add_r2[2] + 4 * _r3_add_r4[2] + bias, activation);
        out3[2] = do_activation(_r1_add_r2[3] + 4 * _r3_add_r4[3] + bias, activation);

        out0[3] = do_activation(_r1_minus_r2[0] + 4 * _r3_minus_r4_x2[0] + tmp[0 * 6 + 5] + bias, activation);
        out1[3] = do_activation(_r1_minus_r2[1] + 4 * _r3_minus_r4_x2[1] + tmp[1 * 6 + 5] + bias, activation);
        out2[3] = do_activation(_r1_minus_r2[2] + 4 * _r3_minus_r4_x2[2] + tmp[2 * 6 + 5] + bias, activation);
        out3[3] = do_activation(_r1_minus_r2[3] + 4 * _r3_minus_r4_x2[3] + tmp[3 * 6 + 5] + bias, activation);
    }
    else
    {
        out0[0] = do_activation(tmp[0 * 6] + _r1_add_r2[0] + _r3_add_r4[0], activation);
        out1[0] = do_activation(tmp[1 * 6] + _r1_add_r2[1] + _r3_add_r4[1], activation);
        out2[0] = do_activation(tmp[2 * 6] + _r1_add_r2[2] + _r3_add_r4[2], activation);
        out3[0] = do_activation(tmp[3 * 6] + _r1_add_r2[3] + _r3_add_r4[3], activation);

        out0[1] = do_activation(_r1_minus_r2[0] + _r3_minus_r4_x2[0], activation);
        out1[1] = do_activation(_r1_minus_r2[1] + _r3_minus_r4_x2[1], activation);
        out2[1] = do_activation(_r1_minus_r2[2] + _r3_minus_r4_x2[2], activation);
        out3[1] = do_activation(_r1_minus_r2[3] + _r3_minus_r4_x2[3], activation);

        out0[2] = do_activation(_r1_add_r2[0] + 4 * _r3_add_r4[0], activation);
        out1[2] = do_activation(_r1_add_r2[1] + 4 * _r3_add_r4[1], activation);
        out2[2] = do_activation(_r1_add_r2[2] + 4 * _r3_add_r4[2], activation);
        out3[2] = do_activation(_r1_add_r2[3] + 4 * _r3_add_r4[3], activation);

        out0[3] = do_activation(_r1_minus_r2[0] + 4 * _r3_minus_r4_x2[0] + tmp[0 * 6 + 5], activation);
        out1[3] = do_activation(_r1_minus_r2[1] + 4 * _r3_minus_r4_x2[1] + tmp[1 * 6 + 5], activation);
        out2[3] = do_activation(_r1_minus_r2[2] + 4 * _r3_minus_r4_x2[2] + tmp[2 * 6 + 5], activation);
        out3[3] = do_activation(_r1_minus_r2[3] + 4 * _r3_minus_r4_x2[3] + tmp[3 * 6 + 5], activation);
    }
}

static inline void trans_output_f43_ordinary(const float* mid, float* out, const float* bias_ptr)
{
    /*
    float AT[24]={
      1.,  1.,  1.,  1.,  1.,  0.,
      0.,  1., -1.,  2., -2.,  0.,
      0.,  1.,  1.,  4.,  4.,  0.,
      0.,  1., -1.,  8., -8.,  1.
    };
    float A[24]={
      1.,  0.,  0.,  0.,
      1.,  1.,  1.,  1.,
      1., -1.,  1., -1.,
      1.,  2.,  4.,  8.,
      1., -2.,  4., -8.,
      0.,  0.,  0.,  1.
    };
    */

    float tmp[24] = {0};

    float r1_add_r2[6];
    float r1_minus_r2[6];
    float r3_add_r4[6];
    float r3_minus_r4_x2[6];

    for (int j = 0; j < 6; j++)
    {
        r1_add_r2[j] = mid[6 * 1 + j] + mid[6 * 2 + j];
        r1_minus_r2[j] = mid[6 * 1 + j] - mid[6 * 2 + j];
        r3_add_r4[j] = mid[6 * 3 + j] + mid[6 * 4 + j];
        r3_minus_r4_x2[j] = (mid[6 * 3 + j] - mid[6 * 4 + j]) * 2;
    }
    for (int j = 0; j < 6; j++)
    {
        tmp[j] = mid[j] + r1_add_r2[j] + r3_add_r4[j];
        tmp[6 + j] = r1_minus_r2[j] + r3_minus_r4_x2[j];
        tmp[12 + j] = r1_add_r2[j] + 4 * r3_add_r4[j];
        tmp[18 + j] = r1_minus_r2[j] + 4 * r3_minus_r4_x2[j] + mid[6 * 5 + j];
    }
    float _r1_add_r2[4];
    float _r1_minus_r2[4];
    float _r3_add_r4[4];
    float _r3_minus_r4_x2[4];
    int idx;
    for (int j = 0; j < 4; j++)
    {
        idx = 6 * j;
        _r1_add_r2[j] = tmp[idx + 1] + tmp[idx + 2];
        _r1_minus_r2[j] = tmp[idx + 1] - tmp[idx + 2];
        _r3_add_r4[j] = tmp[idx + 3] + tmp[idx + 4];
        _r3_minus_r4_x2[j] = (tmp[idx + 3] - tmp[idx + 4]) * 2;
    }
    if (bias_ptr)
    {
        float bias = bias_ptr[0];
        for (int j = 0; j < 4; j++)
        {
            idx = j * 4;
            out[idx] = bias + tmp[j * 6] + _r1_add_r2[j] + _r3_add_r4[j];
            out[idx + 1] = bias + _r1_minus_r2[j] + _r3_minus_r4_x2[j];
            out[idx + 2] = bias + _r1_add_r2[j] + 4 * _r3_add_r4[j];
            out[idx + 3] = bias + _r1_minus_r2[j] + 4 * _r3_minus_r4_x2[j] + tmp[j * 6 + 5];
        }
    }
    else
    {
        for (int j = 0; j < 4; j++)
        {
            idx = j * 4;
            out[idx] = tmp[j * 6] + _r1_add_r2[j] + _r3_add_r4[j];
            out[idx + 1] = _r1_minus_r2[j] + _r3_minus_r4_x2[j];
            out[idx + 2] = _r1_add_r2[j] + 4 * _r3_add_r4[j];
            out[idx + 3] = _r1_minus_r2[j] + 4 * _r3_minus_r4_x2[j] + tmp[j * 6 + 5];
        }
    }
}

static inline void transform_output_f43_1tile(const float* buffer_ptr, float* out, int p_idx, int idx_blockhw,
                                              int block_h, int block_w, int out_hw, int outw, int resi_h, int resi_w,
                                              int KER_COUT_UNIT_, const float* bias, int activation)
{
    float tmp_buffer[TILE * TILE];
    const float* bias_ptr = NULL;
    for (int p = 0; p < KER_COUT_UNIT_; p++)
    {
        int cout_idx = p_idx + p;
        if (bias)
        {
            bias_ptr = (bias + cout_idx);
        }
        float* out_ptr = out + cout_idx * out_hw;
        int i_h = idx_blockhw / block_w;
        int j_w = idx_blockhw % block_w;
        if ((resi_h == 0 && resi_w == 0) || (resi_h == 0 && (j_w < block_w - 1)) ||
            (resi_w == 0 && (i_h < block_h - 1)) || ((j_w < block_w - 1) && (i_h < block_h - 1)))
        {
            trans_output_f43(buffer_ptr, out_ptr + (i_h * TILE * outw + j_w * TILE), outw, bias_ptr, activation);
        }
        else
        {
            int ret_h = TILE - resi_h;
            if (i_h < block_h - 1)
                ret_h = TILE;
            int ret_w = TILE - resi_w;
            if (j_w < block_w - 1)
                ret_w = TILE;

            // tmp_buffer
            trans_output_f43_ordinary(buffer_ptr, tmp_buffer, bias_ptr);
            float* out_pointer = out_ptr + (i_h * TILE * outw + j_w * TILE);
            for (int hh = 0; hh < ret_h; hh++)
            {
                for (int ww = 0; ww < ret_w; ww++)
                {
                    out_pointer[hh * outw + ww] = do_activation(tmp_buffer[hh * TILE + ww], activation);
                }
            }
        }
        buffer_ptr += ELEM_SIZE;
    }
}

static inline void transform_output_f43_4tile(float* buffer_ptr, float* out, int p_idx, int block_idx, int block_h,
                                              int block_w, int outh, int outw, int resi_h, int resi_w,
                                              int KER_COUT_UNIT_, const float* bias, int activation)
{
    int out_hw = outh * outw;
    float tmp_buffer[TILE * TILE];

    int idx_h[4];
    int idx_w[4];
    idx_h[0] = (block_idx) / block_w;
    idx_h[1] = (block_idx + 1) / block_w;
    idx_h[2] = (block_idx + 2) / block_w;
    idx_h[3] = (block_idx + 3) / block_w;

    idx_w[0] = (block_idx) % block_w;
    idx_w[1] = (block_idx + 1) % block_w;
    idx_w[2] = (block_idx + 2) % block_w;
    idx_w[3] = (block_idx + 3) % block_w;

    float* bias_ptr = NULL;
    for (int p = 0; p < KER_COUT_UNIT_; p++)
    {
        int cout_idx = p_idx + p;
        float* out_ptr = out + cout_idx * out_hw;
        if (bias)
        {
            bias_ptr = ( float* )bias + cout_idx;
        }
        for (int ii = 0; ii < 4; ii++)
        {
            int i_h = idx_h[ii];
            int j_w = idx_w[ii];
            if ((resi_h == 0 && resi_w == 0) || (resi_h == 0 && (j_w < block_w - 1)) ||
                (resi_w == 0 && (i_h < block_h - 1)) || ((j_w < block_w - 1) && (i_h < block_h - 1)))
            {
                trans_output_f43(buffer_ptr, out_ptr + (i_h * TILE * outw + j_w * TILE), outw, bias_ptr, activation);
            }    // direct use_out_ptr
            else
            {
                int ret_h = TILE - resi_h;
                if (i_h < block_h - 1)
                    ret_h = TILE;
                int ret_w = TILE - resi_w;
                if (j_w < block_w - 1)
                    ret_w = TILE;

                // tmp_buffer
                trans_output_f43_ordinary(buffer_ptr, tmp_buffer, bias_ptr);
                float* out_pointer = out_ptr + (i_h * TILE * outw + j_w * TILE);
                for (int hh = 0; hh < ret_h; hh++)
                {
                    for (int ww = 0; ww < ret_w; ww++)
                    {
                        out_pointer[hh * outw + ww] = do_activation(tmp_buffer[hh * 4 + ww], activation);
                    }
                }
            }    // end else, tmp_buff
            buffer_ptr += ELEM_SIZE;
        }
    }
}

// trans_input  [block_hw/4][ELEM_SIZE][inc][4]
// kernel       [out_c/PER_OUT_CHAN][ELEM_SIZE][in_c][PER_OUT_CHAN]
static void wino_sgemm_4x16_1(const float* ker, const float* inp, float* output, int cin, int cout_end,
                              int block_h, int block_w,  int out_c, int num_thread, int s, int cpu_affinity)
{
    int block_hw = block_h * block_w;

    #pragma omp parallel for num_threads(num_thread)
    for (int p = 0; p < (cout_end & -PER_OUT_CHAN); p += PER_OUT_CHAN)
    {
        float * out_ptr = output + p * ELEM_SIZE * block_hw;
        float * out_ptr1 ;
        int i;
        
        for (i = 0; i < (block_hw & -4); i += 4)
        {
            out_ptr1 = out_ptr + i * ELEM_SIZE * KER_COUT_UNIT;

            int offset = s * block_hw * cin + i * cin;
            int offset_ker = s * cin * out_c + p * cin;

//#ifdef __aarch64__
            wino_sgemm_4x16_A72(out_ptr1 + s * BLOCK_HW_UNIT, inp + offset, ker + offset_ker, cin, 1);
        }
        
        for(; i < block_hw ;i++)
        {
            out_ptr1 = out_ptr + i * ELEM_SIZE * KER_COUT_UNIT;

            int offset_ker = s * cin * out_c + p * cin;
            int offset = s * block_hw * cin + i * cin;

            wino_sgemm_1x16(out_ptr1 + s * KER_COUT_UNIT, inp + offset, ker + offset_ker, cin);
        }
    }
}

void wino_sgemm_4x4_1(const float* ker, const float* inp, float* output, int cin, int cout_start,
                    int cout_end, int block_h, int block_w, int out_c, int activation, int s, int num_thread, int cpu_affinity)
{
    int block_start = 0;
    int block_hw = block_h * block_w;
    int block_end = block_hw;

#pragma omp parallel for num_threads(num_thread)
    for (int p = (cout_start & -KER_COUT_UNIT4); p < (cout_end & -KER_COUT_UNIT4); p += KER_COUT_UNIT4)
    {
        float* out_ptr = output + p * ELEM_SIZE * block_hw;

        int i = 0;
        for(i = (block_start & -4); i < (block_end & -4); i += 4)
        {
            float* out_ptr1 = out_ptr + i * ELEM_SIZE * KER_COUT_UNIT4;
            int offset = s * block_hw * cin + i * cin;
            int offset_ker = s * cin * out_c + p * cin;
//#ifdef __aarch64__
            wino_sgemm_4x4_A72(out_ptr1 + s * BLOCK_HW_UNIT, inp + offset, ker + offset_ker, cin, 1);
        }
        for(; i < block_end; i++)
        {
            float* out_ptr1 = out_ptr + i * ELEM_SIZE * KER_COUT_UNIT4;

            int offset_ker = s * cin * out_c + p * cin;
            int offset = s * block_hw * cin + i * cin;

            wino_sgemm_1x4(out_ptr1 + s * KER_COUT_UNIT4, inp + offset, ker + offset_ker, cin);
        }
    }
    for (int p = (cout_end & -KER_COUT_UNIT4); p < cout_end; p++)
    {
        float* out_ptr = output + p * ELEM_SIZE * block_hw;
        float* ker_ = (float*)(ker + s * cin * out_c + p * cin);
        int i = 0;
        for (i = (block_start & -4); i < (block_end & -4); i += 4)
        {
            float* out_ptr1 = out_ptr + i * ELEM_SIZE + s * BLOCK_HW_UNIT;
            float* inp_ = (float*)(inp + s * block_hw * cin + i*cin);
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for (int k = 0; k < cin; k++)
            {
                sum0 += inp_[k * 4    ] * ker_[k];
                sum1 += inp_[k * 4 + 1] * ker_[k];
                sum2 += inp_[k * 4 + 2] * ker_[k];
                sum3 += inp_[k * 4 + 3] * ker_[k];
            }
            out_ptr1[0] = sum0;
            out_ptr1[1] = sum1;
            out_ptr1[2] = sum2;
            out_ptr1[3] = sum3;
        }
        for(; i < block_end; i++)
		{
            float* out_ptr1 = out_ptr + i * ELEM_SIZE + s;
            float* inp_ = (float*)(inp + s * block_hw * cin + i*cin);
            float sum0 = 0;
            for(int k = 0; k < cin; k++){
                sum0 += inp_[k] * ker_[k];
            }
            out_ptr1[0] = sum0;
        }
    }
}

/* transform output */
static inline void trans_output_p(float* trans_out_ptr,
                                  float* output, float* bias, int bias_term,
                                  int block_h, int block_w, int block_hw,
                                  int out_hw, int out_w, int resi_h, int resi_w,
                                  int activation,int p,int KER_COUT_UNIT_)
{
    int flag_outw = 1;
    if(out_w < 16)
        flag_outw = 0;
    int i;
    for(i=0; i< (block_hw & -BLOCK_HW_UNIT); i+=BLOCK_HW_UNIT){
        float* buffer_ptr = trans_out_ptr + i * KER_COUT_UNIT_ * ELEM_SIZE;
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
        int wino_out_4_tiles = 0;
        if(flag_outw){
            if((idx_h[0] == idx_h[3]) && (idx_h[0] < (block_h - 1)) && (idx_w[3] < (block_w - 1))){
                wino_out_4_tiles = 1;
            }
        }
        if(wino_out_4_tiles == 1){
            float* bias_ptr = NULL;
            for(int pss = 0; pss < KER_COUT_UNIT_; pss++){
                int cout_idx = p + pss;
                float* out_ptr = output + cout_idx * out_hw + idx_h[0] * TILE * out_w + idx_w[0] * TILE;
                if(bias_term){
                    bias_ptr = ( float* )(bias + cout_idx);
                }
                float ker00[4] = {2, 4, 8, 0};
                tran_out_4(buffer_ptr + pss * ELEM_SIZE * BLOCK_HW_UNIT, out_ptr, out_w * sizeof(float), ker00,
                           bias_ptr, activation);
            }
        }
        else{
            float tmp_buffer[TILE * TILE];
            const float* bias_ptr = NULL;
            for(int pss = 0; pss < KER_COUT_UNIT_; pss++){
                int cout_idx = p + pss;
                float* out_ptr = output + cout_idx * out_hw;
                if(bias_term){
                    bias_ptr = bias + cout_idx;
                }
                float buffer[BLOCK_HW_UNIT * ELEM_SIZE];
                float* buffer_ptr0 = buffer;
                float* mid_ptr = buffer_ptr + pss * BLOCK_HW_UNIT * ELEM_SIZE;
                for(int t = 0; t < BLOCK_HW_UNIT; t++){
                    for(int ss = 0; ss < ELEM_SIZE; ss++){
                        *buffer_ptr0 = mid_ptr[ss * BLOCK_HW_UNIT + t];
                        buffer_ptr0++;
                    }
                }
                for(int ii = 0; ii < BLOCK_HW_UNIT; ii++){
                    int i_h = idx_h[ii];
                    int j_w = idx_w[ii];
                    if((resi_h == 0 && resi_w == 0) || (resi_h == 0 && (j_w < block_w - 1)) ||
                       (resi_w == 0 && (i_h < block_h - 1)) || ((j_w < block_w - 1) && (i_h < block_h - 1))){
                        trans_output_f43(buffer + ii * ELEM_SIZE, out_ptr + (i_h * TILE * out_w + j_w * TILE),
                                         out_w, ( const float* )bias_ptr, activation);
                    }
                    else{
                        int ret_h = TILE - resi_h;
                        if(i_h < block_h - 1) ret_h = TILE;
                        int ret_w = TILE - resi_w;
                        if(j_w < block_w - 1) ret_w = TILE;
                        trans_output_f43_ordinary(buffer + ii * ELEM_SIZE, tmp_buffer, ( const float* )bias_ptr);
                        float* out_pointer = out_ptr + (i_h * TILE * out_w + j_w * TILE);
                        for(int hh = 0; hh < ret_h; hh++){
                            for(int ww = 0; ww < ret_w; ww++){
                                out_pointer[hh * out_w + ww] = do_activation(tmp_buffer[hh * 4 + ww], activation);
                            }
                        }
                    }
                }
            }
        }
    }
    for(; i < block_hw; i++){
        float* buffer_ptr = trans_out_ptr + i * KER_COUT_UNIT_ * ELEM_SIZE;
        float resi_buffer[KER_COUT_UNIT_ * ELEM_SIZE];
        float* buffer0 = resi_buffer;
        for(int pp = 0; pp < KER_COUT_UNIT_; pp++){
            for(int ss = 0; ss < ELEM_SIZE; ss++){
                *buffer0 = buffer_ptr[ss * KER_COUT_UNIT_ + pp];
                buffer0++;
            }
        }
        transform_output_f43_1tile(resi_buffer, output, p, i, block_h, block_w, out_hw, out_w, resi_h, resi_w,
                                   KER_COUT_UNIT_, bias, activation);
    }
}


// transform output
static inline void trans_output_1(float* trans_out, float* output, float* bias, int bias_term, int block_h, int block_w,
                                int cout_start, int cout_end, int out_hw, int out_w, int resi_h, int resi_w,
                                  int activation,int num_thread)
{
    int block_hw = block_h * block_w;
    int p;
    //cout 16
#pragma omp parallel for num_threads(num_thread) shared(block_hw)
    for(p = cout_start; p < (cout_end& -KER_COUT_UNIT); p+=KER_COUT_UNIT){
        trans_output_p(trans_out + p * block_hw * ELEM_SIZE,
                       output, bias, bias_term,
                       block_h, block_w, block_hw,
                       out_hw, out_w, resi_h, resi_w,
                       activation, p, KER_COUT_UNIT);
    }
    //cout 4
#pragma omp parallel for num_threads(num_thread) shared(block_hw)
    for(p = (cout_end & -KER_COUT_UNIT); p < (cout_end & -KER_COUT_UNIT4); p += KER_COUT_UNIT4){
        trans_output_p(trans_out + p * block_hw * ELEM_SIZE,
                       output, bias, bias_term,
                       block_h, block_w, block_hw,
                       out_hw, out_w, resi_h, resi_w,
                       activation, p, KER_COUT_UNIT4);
    }
    // cout 1
#pragma omp parallel for num_threads(num_thread) shared(block_hw)
    for(p=(cout_end & -KER_COUT_UNIT4); p < cout_end; p ++){
        trans_output_p(trans_out + p * block_hw * ELEM_SIZE,
                       output, bias, bias_term,
                       block_h, block_w, block_hw,
                       out_hw, out_w, resi_h, resi_w,
                       activation, p, 1);
    }
}

static int get_private_mem_size(struct tensor* filter, struct conv_param* param)
{
    int output_c = filter->dims[0];
    int input_c = filter->dims[1];
    int trans_ker_size = output_c * input_c * ELEM_SIZE * sizeof(float);
    return trans_ker_size + 128;    // caution
}

int wino_conv_hcl_prerun_1(struct tensor* input_tensor, struct tensor* filter_tensor,
                         struct tensor* output_tensor, struct conv_priv_info* priv_info, struct conv_param* param)
{
    // fTLOG_ERR(stderr,"run into wino_1 prerun.\n");
    int output_c = filter_tensor->dims[0];
    int input_c = filter_tensor->dims[1];
    int mem_size = get_private_mem_size(filter_tensor, param);
    float* trans_mem = ( float* )sys_malloc(mem_size);

    if (!priv_info->external_interleave_mem)
    {
        void* mem = sys_malloc(mem_size);
        priv_info->interleave_buffer = mem;
        priv_info->interleave_buffer_size = mem_size;
    }

    transform_kernel_f43_tile(filter_tensor, trans_mem);
    interleave_kernel_1(trans_mem, ( float* )priv_info->interleave_buffer, output_c, input_c);

    sys_free(trans_mem);

    return 0;
}

int wino_conv_hcl_run_1(struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* bias_tensor,
                      struct tensor* output_tensor, struct conv_priv_info* priv_info, struct conv_param* param,
                      int num_thread, int cpu_affinity)
{
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;

    // pad
    int pad_h0 = param->pad_h0;
    int pad_w0 = param->pad_w0;
    int act_type = param->activation;

    // input
    int batch = input_tensor->dims[0];
    int in_c = input_tensor->dims[1];
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];
    int input_size = in_c * in_h * in_w;
    int kernel_size = in_c * kernel_h * kernel_w;

    // output
    int out_c = output_tensor->dims[1];
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int out_hw = out_h * out_w;
    int output_size = out_c * out_h * out_w;

    /* wino param */
    int block_h = (out_h + TILE - 1) / TILE;
    int block_w = (out_w + TILE - 1) / TILE;
    int block_hw = block_h * block_w;
    int padded_in_h = block_h * TILE + 2;
    int padded_in_w = block_w * TILE + 2;
    int padded_in_hw = padded_in_h * padded_in_w;

    /* buffer addr */
    float* input_buf = ( float* )input_tensor->data;
    float* output_buf = ( float* )output_tensor->data;
    float* biases_buf = NULL;
    int bias_term = 0;

    if (bias_tensor != NULL)
    {
        biases_buf = ( float* )bias_tensor->data;
        bias_term = 1;
    }

    float* col_buf = ( float* )priv_info->im2col_buffer;
    float* interleave_buf = ( float* )priv_info->interleave_buffer;

    int inp_padded_size = sizeof(float) * (in_c * padded_in_hw + 2);

    int nn_out_c = (out_c / PER_OUT_CHAN) * PER_OUT_CHAN;

    int nn_block = block_hw >> 2;
    int resi_block = nn_block << 2;
    int resi_h = block_h * TILE - out_h;
    int resi_w = block_w * TILE - out_w;

    for (int n = 0; n < batch; n++)
    {
        float* input_padded = ( float* )sys_malloc(inp_padded_size);
        float* trans_inp = ( float* )sys_malloc(sizeof(float) * ELEM_SIZE * in_c * block_hw + 128);
        float* trans_out = ( float* )sys_malloc(sizeof(float) * ELEM_SIZE * out_c * block_hw);

        float* input = input_buf + n * input_size;
        float* output = output_buf + n * output_size;

        /* PAD input */
        pad_input1(input, input_padded, in_c, in_h, in_w, padded_in_h, padded_in_w, pad_h0, pad_w0);

        /* trans input */
        tran_input_4block_1(input_padded, trans_inp, in_c, block_h, block_w, padded_in_h, padded_in_w, num_thread);

        if (resi_block != block_hw)
        {
            tran_input_resi_block_1(input_padded, trans_inp, in_c, nn_block, resi_block, block_hw, block_w,
                                  padded_in_hw, padded_in_w);
        }
        sys_free(input_padded);

        /* gemm */
        for(int s = 0; s < ELEM_SIZE; s++)
        {
            wino_sgemm_4x16_1(interleave_buf, trans_inp, trans_out, in_c, nn_out_c, block_h, block_w,
                            out_c, num_thread, s, cpu_affinity);
            if (nn_out_c != out_c)
            {
               wino_sgemm_4x4_1(interleave_buf, trans_inp, trans_out, in_c, nn_out_c,
                                 out_c, block_h, block_w, out_c, act_type, s ,num_thread, cpu_affinity);
            }
        }
        sys_free(trans_inp);
        trans_output_1(trans_out, output, biases_buf, bias_term, block_h, block_w, 0, out_c, out_hw, out_w, resi_h, resi_w,
                       act_type,num_thread);

        sys_free(trans_out);
    }
    return 0;
}

#endif
