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

#ifndef __REF_GRU_KERNEL_H__
#define __REF_GRU_KERNEL_H__

#include <stdint.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct gru_param
{
    float scale[2];
    int zero_point[2];
    float* init_h_data;
    float* bias;
    float* kernel;
    float* candidate_kernel;
    float* candidate_bias;
    float* fused_kernel;
    int seq_lens;
    int batch_size;
    int input_size;
    int output_len;
    int hidden_size;
    int mxnet_flag;
};

void sigmoid_gru(float* data, int size)
{
    for(int i = 0; i < size; i++)
    {
        data[i] = std::min(data[i], 30.0f);
        data[i] = std::max(data[i], -30.0f);

        data[i] = 1 / (1 + exp(-data[i]));
    }
}
/*
@ func_name: concat_axis_1
@ param:
    a:[m, n1]
    b:[m, n2]
    c:[m, n1 + n2]
*/
void concat_axis_1(const float* a, const float* b, float* c, int m, int n1, int n2)
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

void slice_axis_1(float* a, float* c, int m, int n, int st, int ed)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = st; j < ed; j++)
        {
            c[i * (ed - st) + j - st] = a[i * n + j];
        }
    }
}
void do_gemm(const float* a, const float* b, float* c, int m, int k, int n, int lda, int ldb, int ldc)
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

void do_gemm_mx(const float* a, const float* b, float* c, int m, int k, int n, int lda, int ldb, int ldc)
{
    float* tmp = (float* )malloc(n*k*sizeof(float));
    for(int a=0;a<k;a++)
    {
        for(int z=0;z<n;z++)
        {
            tmp[a*n+z]=b[z*k+a];
        }
    }

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            c[i * n + j] = 0.f;

            for(int p = 0; p < k; p++)
            {
                c[i * n + j] += a[i * k + p] * tmp[p * n + j];
            }
        }
    }
}

// void do_gemm(const float* a, const float* b, float* c, int m, int k, int n, int lda, int ldb, int ldc)
// {
//     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
// }
// void do_gemm_mx(const float* a, const float* b, float* c, int m, int k, int n, int lda, int ldb, int ldc)
// {
//     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
// }

bool do_GRU_step(const float* input, float* init_h, const float* kernel, const float* bias,
                 const float* candidate_kernel, const float* candidate_bias, int batch_size, int input_size,
                 int hidden_size, int mxnet_flag)
{
    if(mxnet_flag == 1)
    {
        float* i2h_mat = ( float* )malloc(sizeof(float) * batch_size * 3 * hidden_size);
        float* h2h_mat = ( float* )malloc(sizeof(float) * batch_size * 3 * hidden_size);

        float* i2h_r = ( float* )malloc(batch_size * hidden_size * sizeof(float));
        float* i2h_z = ( float* )malloc(batch_size * hidden_size * sizeof(float));
        float* i2h = ( float* )malloc(batch_size * hidden_size * sizeof(float));

        float* h2h_r = ( float* )malloc(batch_size * hidden_size * sizeof(float));
        float* h2h_z = ( float* )malloc(batch_size * hidden_size * sizeof(float));
        float* h2h = ( float* )malloc(batch_size * hidden_size * sizeof(float));

        float* r_g = ( float* )malloc(batch_size * hidden_size * sizeof(float));
        float* u_g = ( float* )malloc(batch_size * hidden_size * sizeof(float));
        float* next_h_tmp = ( float* )malloc(batch_size * hidden_size * sizeof(float));

        do_gemm_mx(input, kernel, i2h_mat, batch_size, input_size, 3 * hidden_size, input_size, input_size,
                   3 * hidden_size);

        for(int i = 0; i < batch_size; i++)
        {
            for(int j = 0; j < (3 * hidden_size); j++)
            {
                i2h_mat[i * (3 * hidden_size) + j] += bias[j];
            }
        }

        do_gemm_mx(init_h, candidate_kernel, h2h_mat, batch_size, hidden_size, 3 * hidden_size, hidden_size,
                   hidden_size, 3 * hidden_size);

        for(int i = 0; i < batch_size; i++)
        {
            for(int j = 0; j < (3 * hidden_size); j++)
            {
                h2h_mat[i * (3 * hidden_size) + j] += candidate_bias[j];
            }
        }
        slice_axis_1(i2h_mat, i2h_r, batch_size, 3 * hidden_size, 0, hidden_size);
        slice_axis_1(i2h_mat, i2h_z, batch_size, 3 * hidden_size, hidden_size, 2 * hidden_size);
        slice_axis_1(i2h_mat, i2h, batch_size, 3 * hidden_size, 2 * hidden_size, 3 * hidden_size);

        slice_axis_1(h2h_mat, h2h_r, batch_size, 3 * hidden_size, 0, hidden_size);
        slice_axis_1(h2h_mat, h2h_z, batch_size, 3 * hidden_size, hidden_size, 2 * hidden_size);
        slice_axis_1(h2h_mat, h2h, batch_size, 3 * hidden_size, 2 * hidden_size, 3 * hidden_size);

        for(int i = 0; i < batch_size * hidden_size; i++)
        {
            r_g[i] = i2h_r[i] + h2h_r[i];
        }
        sigmoid_gru(r_g, hidden_size * batch_size);
        for(int i = 0; i < batch_size * hidden_size; i++)
        {
            u_g[i] = i2h_z[i] + h2h_z[i];
        }
        sigmoid_gru(u_g, hidden_size * batch_size);

        for(int i = 0; i < batch_size * hidden_size; i++)
        {
            next_h_tmp[i] = tanh(i2h[i] + r_g[i] * h2h[i]);
        }

        for(int i = 0; i < batch_size * hidden_size; i++)
        {
            init_h[i] = u_g[i] * init_h[i] + (1 - u_g[i]) * next_h_tmp[i];
        }

        // free memory
        free(i2h_mat);
        free(h2h_mat);
        free(i2h_r);
        free(i2h_z);
        free(i2h);
        free(h2h_r);
        free(h2h_z);
        free(h2h);
        free(r_g);
        free(u_g);
        free(next_h_tmp);

        return true;
    }
    else
    {
        int input_total_size = input_size + hidden_size;
        int batch_cell_size = hidden_size * batch_size;

        float* merged_input = ( float* )malloc(sizeof(float) * batch_size * (input_total_size));
        float* matmul_result = ( float* )malloc(sizeof(float) * batch_size * 2 * hidden_size);
        float* r = ( float* )malloc(batch_cell_size * sizeof(float));
        float* u = ( float* )malloc(batch_cell_size * sizeof(float));
        float* c = ( float* )malloc(batch_cell_size * sizeof(float));
        float* r_state = ( float* )malloc(batch_cell_size * sizeof(float));
        float* candidate = ( float* )malloc(sizeof(float) * batch_size * hidden_size);
        // merge input
        concat_axis_1(input, init_h, merged_input, batch_size, input_size, hidden_size);
        // do gemm
        do_gemm(merged_input, kernel, matmul_result, batch_size, input_total_size, 2 * hidden_size, input_total_size,
                2 * hidden_size, 2 * hidden_size);
        // add bias

        for(int i = 0; i < batch_size; i++)
        {
            for(int j = 0; j < (2 * hidden_size); j++)
            {
                matmul_result[i * (2 * hidden_size) + j] += bias[j];
            }
        }

        sigmoid_gru(matmul_result, 2 * hidden_size * batch_size);
        slice_axis_1(matmul_result, r, batch_size, 2 * hidden_size, 0, hidden_size);
        slice_axis_1(matmul_result, u, batch_size, 2 * hidden_size, hidden_size, 2 * hidden_size);

        for(int i = 0; i < batch_cell_size; i++)
            r_state[i] = r[i] * init_h[i];

        concat_axis_1(input, r_state, merged_input, batch_size, input_size, hidden_size);
        // candidate kernerl

        do_gemm(merged_input, candidate_kernel, candidate, batch_size, input_total_size, hidden_size, input_total_size,
                hidden_size, hidden_size);
        // candidate bias

        for(int i = 0; i < batch_size; i++)
        {
            for(int j = 0; j < hidden_size; j++)
            {
                candidate[i * hidden_size + j] += candidate_bias[j];
            }
        }

        for(int i = 0; i < batch_cell_size; i++)
        {
            c[i] = tanh(candidate[i]);
        }

        for(int i = 0; i < batch_cell_size; i++)
        {
            init_h[i] = u[i] * init_h[i] + (1 - u[i]) * c[i];
        }
        // free memory
        free(merged_input);
        free(matmul_result);
        free(candidate);
        free(r);
        free(u);
        free(c);
        return true;
    }
}

typedef int (*ref_gru_t)(void* in_data, void* out_data, gru_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_gru_fp32.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
