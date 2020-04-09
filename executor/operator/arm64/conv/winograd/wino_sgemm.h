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

//#define D_WINO

extern void wino_sgemm_4x16_A72(float* output, const float* input, const float* kernel, long cin, short stride_save);
extern void wino_sgemm_4x4_A72(float* output, const float* input, const float* kernel, long cin, short stride_save);
extern void wino_sgemm_1x16(float* output, const float* input, const float* kernel, long cin);
extern void wino_sgemm_1x4(float* output, const float* input, const float* kernel, long cin);
extern void tran_out_4(float* buffer, float* output, int out_w, float* ker00, float* bias, int activation);

void wino_sgemm_4x4(const float* ker, const float* inp, float* output, const float* bias, int bias_term,
                                   int cin, int cpu_type, int cout_start, int cout_end, int block_start, int block_end,
                                   int block_h, int block_w, int out_hw, int out_w, int resi_h, int resi_w,
                                   int activation);

void wino_sgemm_4x16(const float* ker, const float* inp, float* output, const float* bias, int bias_term,
                                   int cin, int cpu_type, int cout_start, int cout_end, int block_start, int block_end,
                                   int block_h, int block_w, int out_hw, int out_w, int resi_h, int resi_w,
                                   int activation);

// inp [block_hw//4][36][cin][4] ->[36][block_hw//4][cin][4]
// ker [cout//16][36][cin][16] -> [36][cout//16][cin][16]
// mid [cout//16]([block//4][36][16][4] + block_i[36][16]
void wino_sgemm_4x16_1(const float* ker, const float* inp, float* trans_out, int cin, int cpu_type,
                                     int cout_start, int cout_end, int block_start, int block_end, int block_hw,
                                     int outc, int s);

void wino_sgemm_4x4_1(const float* ker, const float* inp, float* trans_out, int cin, int cpu_type,
                                     int cout_start, int cout_end, int block_start, int block_end, int block_hw,
                                     int outc, int s);

void wino_sgemm_4x16_nhwc(const float* ker, const float* inp, float* output, const float* bias,
                                        int bias_term, int cin, int cpu_type, int cout_start, int cout_end,
                                        int block_start, int block_end, int block_h, int block_w, int out_c, int out_w,
                                        int resi_h, int resi_w, int activation);

void wino_sgemm_4x16_func(const float* ker, const float* inp, float* output, const float* bias, int bias_term,
                                   int cin, int cpu_type, int cout_start, int cout_end, int is_4block,int block_start,int resi,
                                   int block_h, int block_w, int out_hw, int out_w, int resi_h, int resi_w,
                                   int activation);

void wino_sgemm_4x4_func(const float* ker, const float* inp, float* output, const float* bias, int bias_term,
                                   int cin, int cpu_type, int cout_start, int cout_end, int is_4block ,int block_start,int resi,
                                   int block_h, int block_w, int out_hw, int out_w, int resi_h, int resi_w,
                                   int activation);
#ifdef __cplusplus
}
#endif    //__cplusplus

#endif    // __WINO_SGENN_H__
