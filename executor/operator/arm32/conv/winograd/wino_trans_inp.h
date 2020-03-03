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
#include <arm_neon.h>

#ifdef __cplusplus
extern "C" {
#endif    //__cplusplus

// ------------------------------ INPUT ------------------------------------------

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
static inline void trans_inp_4tile(float* input, float* inp_ptr, int ih, int jw, int c, int in_hw, int inw)
{
    float tmp0[36] = {0};
    float tmp1[36] = {0};
    float tmp2[36] = {0};
    float tmp3[36] = {0};
    float* inp = ( float* )input + c * in_hw + ih * 4 * inw + jw * 4;
    float* inp0 = inp;
    float* inp1 = inp0 + inw;
    float* inp2 = inp1 + inw;
    float* inp3 = inp2 + inw;
    float* inp4 = inp3 + inw;
    float* inp5 = inp4 + inw;

    float r1_add_r2[18];
    float r3_add_r4[18];
    float r1_minus_r2[18];
    float r3_minus_r4[18];
    float r4_minus_r2[18];
    float r1_minus_r3[18];
    for(int j = 0; j < 18; j++)
    {
        r1_add_r2[j] = inp1[j] + inp2[j];
        r1_minus_r2[j] = inp1[j] - inp2[j];
        r3_add_r4[j] = inp3[j] + inp4[j];
        r3_minus_r4[j] = inp3[j] - inp4[j];
        r4_minus_r2[j] = inp4[j] - inp2[j];
        r1_minus_r3[j] = inp1[j] - inp3[j];
    }
    int jj, j3, j4;
    for(int j = 0; j < 6; j++)
    {
        tmp0[j] = 4 * inp0[j] - 5 * inp2[j] + inp4[j];
        tmp0[6 + j] = r3_add_r4[j] - 4 * r1_add_r2[j];
        tmp0[12 + j] = 4 * r1_minus_r2[j] - r3_minus_r4[j];
        tmp0[18 + j] = r4_minus_r2[j] - 2 * r1_minus_r3[j];
        tmp0[24 + j] = r4_minus_r2[j] + 2 * r1_minus_r3[j];
        tmp0[30 + j] = 4 * inp1[j] - 5 * inp3[j] + inp5[j];
        jj = j + 4;
        tmp1[j] = 4 * inp0[jj] - 5 * inp2[jj] + inp4[jj];
        tmp1[6 + j] = r3_add_r4[jj] - 4 * r1_add_r2[jj];
        tmp1[12 + j] = 4 * r1_minus_r2[jj] - r3_minus_r4[jj];
        tmp1[18 + j] = r4_minus_r2[jj] - 2 * r1_minus_r3[jj];
        tmp1[24 + j] = r4_minus_r2[jj] + 2 * r1_minus_r3[jj];
        tmp1[30 + j] = 4 * inp1[jj] - 5 * inp3[jj] + inp5[jj];
        j3 = j + 8;
        tmp2[j] = 4 * inp0[j3] - 5 * inp2[j3] + inp4[j3];
        tmp2[6 + j] = r3_add_r4[j3] - 4 * r1_add_r2[j3];
        tmp2[12 + j] = 4 * r1_minus_r2[j3] - r3_minus_r4[j3];
        tmp2[18 + j] = r4_minus_r2[j3] - 2 * r1_minus_r3[j3];
        tmp2[24 + j] = r4_minus_r2[j3] + 2 * r1_minus_r3[j3];
        tmp2[30 + j] = 4 * inp1[j3] - 5 * inp3[j3] + inp5[j3];
        j4 = j + 12;
        tmp3[j] = 4 * inp0[j4] - 5 * inp2[j4] + inp4[j4];
        tmp3[6 + j] = r3_add_r4[j4] - 4 * r1_add_r2[j4];
        tmp3[12 + j] = 4 * r1_minus_r2[j4] - r3_minus_r4[j4];
        tmp3[18 + j] = r4_minus_r2[j4] - 2 * r1_minus_r3[j4];
        tmp3[24 + j] = r4_minus_r2[j4] + 2 * r1_minus_r3[j4];
        tmp3[30 + j] = 4 * inp1[j4] - 5 * inp3[j4] + inp5[j4];
    }
    float r1_4_minus_r3[24];
    float r4_minus_4_r2[24];
    float r4_minus_r2_[24];
    float r1_minus_r3_x2[24];
    for(int j = 0; j < 6; j++)
    {
        r4_minus_r2_[j] = tmp0[j * 6 + 4] - tmp0[j * 6 + 2];
        r1_4_minus_r3[j] = 4 * tmp0[j * 6 + 1] - tmp0[j * 6 + 3];
        r4_minus_4_r2[j] = tmp0[j * 6 + 4] - 4 * tmp0[j * 6 + 2];
        r1_minus_r3_x2[j] = 2 * (tmp0[j * 6 + 1] - tmp0[j * 6 + 3]);
        jj = j + 6;
        r4_minus_r2_[jj] = tmp1[j * 6 + 4] - tmp1[j * 6 + 2];
        r1_4_minus_r3[jj] = 4 * tmp1[j * 6 + 1] - tmp1[j * 6 + 3];
        r4_minus_4_r2[jj] = tmp1[j * 6 + 4] - 4 * tmp1[j * 6 + 2];
        r1_minus_r3_x2[jj] = 2 * (tmp1[j * 6 + 1] - tmp1[j * 6 + 3]);
        j3 = j + 12;
        r4_minus_r2_[j3] = tmp2[j * 6 + 4] - tmp2[j * 6 + 2];
        r1_4_minus_r3[j3] = 4 * tmp2[j * 6 + 1] - tmp2[j * 6 + 3];
        r4_minus_4_r2[j3] = tmp2[j * 6 + 4] - 4 * tmp2[j * 6 + 2];
        r1_minus_r3_x2[j3] = 2 * (tmp2[j * 6 + 1] - tmp2[j * 6 + 3]);
        j4 = j + 18;
        r4_minus_r2_[j4] = tmp3[j * 6 + 4] - tmp3[j * 6 + 2];
        r1_4_minus_r3[j4] = 4 * tmp3[j * 6 + 1] - tmp3[j * 6 + 3];
        r4_minus_4_r2[j4] = tmp3[j * 6 + 4] - 4 * tmp3[j * 6 + 2];
        r1_minus_r3_x2[j4] = 2 * (tmp3[j * 6 + 1] - tmp3[j * 6 + 3]);
    }
    float* inp_ptr1 = inp_ptr + 36;
    float* inp_ptr2 = inp_ptr + 72;
    float* inp_ptr3 = inp_ptr + 108;
    for(int j = 0; j < 6; j++)
    {
        inp_ptr[j * 6] = 4 * tmp0[j * 6] - 5 * tmp0[j * 6 + 2] + tmp0[j * 6 + 4];
        inp_ptr[1 + j * 6] = r4_minus_4_r2[j] - r1_4_minus_r3[j];
        inp_ptr[2 + j * 6] = r4_minus_4_r2[j] + r1_4_minus_r3[j];
        inp_ptr[3 + j * 6] = r4_minus_r2_[j] - r1_minus_r3_x2[j];
        inp_ptr[4 + j * 6] = r4_minus_r2_[j] + r1_minus_r3_x2[j];
        inp_ptr[5 + j * 6] = 4 * tmp0[j * 6 + 1] - 5 * tmp0[j * 6 + 3] + tmp0[j * 6 + 5];
        jj = j + 6;
        inp_ptr1[j * 6] = 4 * tmp1[j * 6] - 5 * tmp1[j * 6 + 2] + tmp1[j * 6 + 4];
        inp_ptr1[1 + j * 6] = r4_minus_4_r2[jj] - r1_4_minus_r3[jj];
        inp_ptr1[2 + j * 6] = r4_minus_4_r2[jj] + r1_4_minus_r3[jj];
        inp_ptr1[3 + j * 6] = r4_minus_r2_[jj] - r1_minus_r3_x2[jj];
        inp_ptr1[4 + j * 6] = r4_minus_r2_[jj] + r1_minus_r3_x2[jj];
        inp_ptr1[5 + j * 6] = 4 * tmp1[j * 6 + 1] - 5 * tmp1[j * 6 + 3] + tmp1[j * 6 + 5];
        j3 = j + 12;
        inp_ptr2[j * 6] = 4 * tmp2[j * 6] - 5 * tmp2[j * 6 + 2] + tmp2[j * 6 + 4];
        inp_ptr2[1 + j * 6] = r4_minus_4_r2[j3] - r1_4_minus_r3[j3];
        inp_ptr2[2 + j * 6] = r4_minus_4_r2[j3] + r1_4_minus_r3[j3];
        inp_ptr2[3 + j * 6] = r4_minus_r2_[j3] - r1_minus_r3_x2[j3];
        inp_ptr2[4 + j * 6] = r4_minus_r2_[j3] + r1_minus_r3_x2[j3];
        inp_ptr2[5 + j * 6] = 4 * tmp2[j * 6 + 1] - 5 * tmp2[j * 6 + 3] + tmp2[j * 6 + 5];
        j4 = j + 18;
        inp_ptr3[j * 6] = 4 * tmp3[j * 6] - 5 * tmp3[j * 6 + 2] + tmp3[j * 6 + 4];
        inp_ptr3[1 + j * 6] = r4_minus_4_r2[j4] - r1_4_minus_r3[j4];
        inp_ptr3[2 + j * 6] = r4_minus_4_r2[j4] + r1_4_minus_r3[j4];
        inp_ptr3[3 + j * 6] = r4_minus_r2_[j4] - r1_minus_r3_x2[j4];
        inp_ptr3[4 + j * 6] = r4_minus_r2_[j4] + r1_minus_r3_x2[j4];
        inp_ptr3[5 + j * 6] = 4 * tmp3[j * 6 + 1] - 5 * tmp3[j * 6 + 3] + tmp3[j * 6 + 5];
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
                trans_inp_4_cpu(temp_inp_ptr, inp_ptr_4tile + c * 4, inw, inc * 4);
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
    int s_size = block_hw * inc;
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
                trans_inp_4_cpu(temp_inp_ptr, trans_inp + c * 4 + off_set0, inw, s_size);
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
static inline void tran_input_all(const float* input, float* trans_inp, int inc,int inc_start, int inc_end,
                                  int block_hw, int block_w, int in_hw, int inw)
{
    int idxh[4];
    int idxw[4];
    int ib;
    int c_length = inc_end - inc_start;
    for(ib = 0; ib < (block_hw &-4); ib+=4)
    {
        float* inp_ptr_4tile = trans_inp + ib * ELEM_SIZE * inc;
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
                trans_inp_4_cpu(temp_inp_ptr + c*in_hw, inp_ptr_4tile + c * 4, inw, inc * 4);
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
            // interleave [c-c][4][36]->[36][c][4]
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                float* tmp_inp = inp_ptr_4tile + s*inc*4 + inc_start*4;
                for(int i = 0; i < c_length; i++)
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
    
    for(ib=(block_hw & -4); ib < block_hw; ib++)
    {
        float* inp_ptr = trans_inp + ib * ELEM_SIZE * inc;
        float buffer0[ELEM_SIZE * c_length];
        float* buffer = buffer0;
        for(int c = inc_start; c < inc_end; c++)
        {
            int ih = ib / block_w;
            int jw = ib % block_w;
            trans_inp_1tile(( float* )input, buffer, ih, jw, c, in_hw, inw);
            buffer += ELEM_SIZE;
        }
        // interleave [c][36]->[36][c]
        for(int s = 0; s < ELEM_SIZE; s++)
        {
            float* tmp_inp = inp_ptr + s* inc + inc_start;
            for(int i = 0; i < c_length; i++)
            {
                *tmp_inp = buffer0[i * ELEM_SIZE + s];
                tmp_inp++;
            }
        }
        // end interleave
    }
}

// tran_inp [block_hw/4][36][inc][4]
static inline void transform_input(const float* input, float* trans_inp, int inc, int inc_start, int inc_end,
                                int block_w, int in_hw, int inw, int block_hw)
{
    int idxh[4];
    int idxw[4];
    int ib;
    int c_length = inc_end - inc_start;
    for(ib = 0; ib < (block_hw &-4); ib+=4)
    {
        float* inp_ptr_4tile = trans_inp + ib * ELEM_SIZE * inc;
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
                trans_inp_4_cpu(temp_inp_ptr + c *in_hw, 
                                inp_ptr_4tile + c * 4, inw, inc * 4);
            }
        }
        else
        {
            //buffer[c-c,4,36]
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
                float* tmp_inp = inp_ptr_4tile + s*inc*4 + inc_start*4;
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
        int off_set0 = ib * inc * ELEM_SIZE;
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
            float* tmp_inp = trans_inp + s * inc* 4 + off_set0 + inc_start;
            for(int i = 0; i < c_length; i++)
            {
                *tmp_inp = buffer0[i * ELEM_SIZE + s];
                tmp_inp++;
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