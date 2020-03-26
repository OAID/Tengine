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
 * Author: zpluo@openailab.com
 */

#ifndef __REF_RNN_KERNEL_H__
#define __REF_RNN_KERNEL_H__

#include <stdint.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct rnn_param
{
    float* init_h_data;
    float* bias;
    float* kernel;

    int seq_lens;
    int batch_size;
    int input_size;
    int output_len;
    int hidden_size;

};
 
void concat_axis_1_rnn(const float* a, const float* b, float* c, int m, int n1, int n2)
{
    int n = n1 + n2;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n1; j++)
        {
            c[j + i * n] = a[j + i * n1];
        }
        for(int j = 0; j < n2; j++)
        {
            c[j + i * n + n1] = b[j + i * n2];
        }
    }
}

void do_gemm_rnn(const float* a, const float* b, float* c, int m, int k, int n, int lda, int ldb, int ldc)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            c[i * n + j] = 0.f;
            for(int p = 0; p < k; p++)
            {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
}

bool do_RNN_step(const float* input, float* init_h, const float* kernel, const float* bias, int batch_size,
                    int input_size, int hidden_size)
{
    int input_total_size = input_size + hidden_size;
    int batch_cell_size = hidden_size * batch_size;

    float* ig = ( float* )malloc(batch_cell_size * sizeof(float));

    float* merged_input = ( float* )malloc(sizeof(float) * batch_size * (input_total_size));
    float* matmul_result = ( float* )malloc(sizeof(float) * batch_size * hidden_size);

    // merge input
    concat_axis_1_rnn(input, init_h, merged_input, batch_size, input_size, hidden_size);

    // do gemm
    do_gemm_rnn(merged_input, kernel, matmul_result, batch_size, input_total_size, hidden_size, input_total_size,
            hidden_size, hidden_size);

    // add bias
    if(bias)
    {
        for(int i = 0; i < batch_size; i++)
            for(int j = 0; j < hidden_size; j++)
                matmul_result[i * hidden_size + j] += bias[j];
    }
    // activation
    for(int i = 0; i < batch_cell_size; i++)
    {
        ig[i] = tanh(matmul_result[i]);
        init_h[i] = ig[i];
    }

    // free memory
    free(merged_input);
    free(matmul_result);
    free(ig);

    return true;
}

typedef int (*ref_rnn_t)(void* in_data, void* out_data, rnn_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_rnn_fp32.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
