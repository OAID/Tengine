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

#ifndef __REF_LSTM_KERNEL_H__
#define __REF_LSTM_KERNEL_H__

#include <stdint.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct lstm_param
{
    float scale[2];
    int zero_point[2];
    float* init_h_data;
    float* init_c_data;
    float* bias;
    float forget_bias;
    float* kernel;
    float* w_f_data;
    float* w_i_data;
    float* w_o_data;
    float* projection;
    float* h2h_kernel;
    float* h2h_bias;
    float* fused_kernel;

    int seq_lens;
    int batch_size;
    int input_size;
    int output_len;
    int hidden_size;
    int cell_size;
    int mxnet_flag;

};
 /*
    @ func_name: tanh
    */
void mytanh_lstm(float* data, int size)
{
    for(int i = 0; i < size; i++)
    {
        data[i] = tanh(data[i]);
    }
}

/*
@ func_name: sigmoid_lstm
*/
void sigmoid_lstm(float* data, int size)
{
    for(int i = 0; i < size; i++)
    {
        data[i] = std::min(data[i], 30.0f);
        data[i] = std::max(data[i], -30.0f);

        data[i] = 1 / (1 + exp(-data[i]));
    }
}

// void matmul(float* a, float* b, float* c, int m, int k, int n)
// {
//     // const enum CBLAS_ORDER Order = CblasRowMajor;
//     // const enum CBLAS_TRANSPOSE TransA = CblasNoTrans;
//     // const enum CBLAS_TRANSPOSE TransB = CblasNoTrans;
//     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b, n, 0.0, c, n);
// }

/*
@ func_name: add_b
@ param:
    a:[m, n]
    b:[n]
    c:[m, n]
*/
void add_b(float* a, float* b, float* c, int m, int n)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            c[i * n + j] = a[i * n + j] + b[j];
        }
    }
}
void mymatmul(float* a, float* b, float* c, int m, int k, int n)
{
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            c[i * n + j] = 0.0;
            for(int p = 0; p < k; ++p)
            {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
}
/*
@ func_name: matmul
@ param:
    a:[m, k]
    b:[k, n]
    bias:[n]
    c:[m, n]
*/
void matmul_add_b(float* a, float* b, float* bias, float* c, int m, int k, int n)
{
    // matmul(a, b, c, m, k, n);
    mymatmul(a, b, c, m, k, n);
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            c[i * n + j] += bias[j];
        }
    }
}

/*
@ func_name: multiply
@ requirement:
len = m1 * n1 = m2 * n2 = m3 * n3
@ param:
    a[m1,n1]
    b[m2,n2]
    c[m3,n3]
    len = m1 * n1 = m2 * n2 = m3 * n3
*/
void multiply(float* a, float* b, float* c, int len)
{
    for(int i = 0; i < len; i++)
    {
        c[i] = a[i] * b[i];
    }
}
void add(float* a, float* b, int len)
{
    for(int i = 0; i < len; i++)
    {
        a[i] += b[i];
    }
}
void slice_axis_1_lstm(float* a, float* c, int m, int n, int st, int ed)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = st; j < ed; j++)
        {
            c[i * (ed - st) + j - st] = a[i * n + j];
        }
    }
}
/*
@ func_name: slice_axis_0
@ param:
    st:slice start
    ed:slice end
    a:[m, n]
    c:[(ed-st),n]
*/
void slice_axis_0(float* a, float* c, int m, int n, int st, int ed)
{
    for(int i = st; i < ed; i++)
    {
        for(int j = 0; j < n; j++)
        {
            c[(i - st) * n + j] = a[i * n + j];
        }
    }
}
/*
@ func_name: concat_axis_1_lstm
@ param:
    a:[m, n1]
    b:[m, n2]
    c:[m, n1 + n2]
*/
void concat_axis_1_lstm(const float* a, const float* b, float* c, int m, int n1, int n2)
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

void do_gemm_lstm(const float* a, const float* b, float* c, int m, int k, int n, int lda, int ldb, int ldc)
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

void do_gemm_mx_lstm(const float* a, const float* b, float* c, int m, int k, int n, int lda, int ldb, int ldc)
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

bool do_LSTM_step(const float* input, float* init_h, float* init_c, const float* kernel, const float* bias,
                    const float* h2h_kernel, const float* h2h_bias, const float* w_f_data, const float* w_i_data,
                    const float* w_o_data, const float* projection, float forget_bias, int batch_size, int input_size,
                    int hidden_size, int cell_size, int mxnet_flag)
{
    if(mxnet_flag == 1)
    {
        int batch_cell_size = cell_size * batch_size;
        float* i2h = ( float* )malloc(sizeof(float) * batch_size * cell_size * 4);
        float* h2h = ( float* )malloc(sizeof(float) * batch_size * cell_size * 4);
        float* gates = ( float* )malloc(sizeof(float) * batch_size * cell_size * 4);

        float* ig = ( float* )malloc(batch_cell_size * sizeof(float));
        float* cg = ( float* )malloc(batch_cell_size * sizeof(float));
        float* fg = ( float* )malloc(batch_cell_size * sizeof(float));
        float* og = ( float* )malloc(batch_cell_size * sizeof(float));
        //                                 m         k              n
        do_gemm_mx_lstm(input, kernel, i2h, batch_size, input_size, 4 * cell_size, input_size, input_size,
                    4 * cell_size);

        if(bias)
        {
            for(int i = 0; i < batch_size; i++)
                for(int j = 0; j < 4 * cell_size; j++)
                    i2h[i * 4 * cell_size + j] += bias[j];
        }

        do_gemm_mx_lstm(init_h, h2h_kernel, h2h, batch_size, hidden_size, 4 * hidden_size, hidden_size, hidden_size,
                    4 * hidden_size);
        if(h2h_bias)
        {
            for(int i = 0; i < batch_size; i++)
                for(int j = 0; j < 4 * cell_size; j++)
                    h2h[i * 4 * cell_size + j] += h2h_bias[j];
        }

        for(int i = 0; i < batch_size * 4 * cell_size; i++)
            gates[i] = i2h[i] + h2h[i];

        slice_axis_1_lstm(gates, ig, batch_size, 4 * cell_size, 0, cell_size);
        slice_axis_1_lstm(gates, fg, batch_size, 4 * cell_size, cell_size, 2 * cell_size);
        slice_axis_1_lstm(gates, cg, batch_size, 4 * cell_size, 2 * cell_size, 3 * cell_size);
        slice_axis_1_lstm(gates, og, batch_size, 4 * cell_size, 3 * cell_size, 4 * cell_size);

        for(int i = 0; i < batch_size * cell_size; i++)
            fg[i] += 1;

        sigmoid_lstm(ig, batch_cell_size);
        sigmoid_lstm(fg, batch_cell_size);
        mytanh_lstm(cg, batch_cell_size);
        sigmoid_lstm(og, batch_cell_size);

        for(int i = 0; i < batch_cell_size; i++)
            init_c[i] = init_c[i] * fg[i] + cg[i] * ig[i];

        for(int i = 0; i < batch_cell_size; i++)
        {
            init_h[i] = tanh(init_c[i]) * og[i];
        }

        free(i2h);
        free(h2h);
        free(gates);
        free(ig);
        free(fg);
        free(cg);
        free(og);
        return true;
    }
    else
    {
        int input_total_size = input_size + hidden_size;
        int batch_cell_size = cell_size * batch_size;

        float* merged_input = ( float* )malloc(sizeof(float) * batch_size * (input_total_size));
        float* matmul_result = ( float* )malloc(sizeof(float) * batch_size * cell_size * 4);

        // merge input
        concat_axis_1_lstm(input, init_h, merged_input, batch_size, input_size, hidden_size);

        // do gemm
        do_gemm_lstm(merged_input, kernel, matmul_result, batch_size, input_total_size, 4 * cell_size, input_total_size,
                4 * cell_size, 4 * cell_size);

        // add bias
        if(bias)
        {
            for(int i = 0; i < batch_size; i++)
                for(int j = 0; j < 4 * cell_size; j++)
                    matmul_result[i * 4 * cell_size + j] += bias[j];
        }

        float* ig = ( float* )malloc(batch_cell_size * sizeof(float));
        float* cg = ( float* )malloc(batch_cell_size * sizeof(float));
        float* fg = ( float* )malloc(batch_cell_size * sizeof(float));
        float* og = ( float* )malloc(batch_cell_size * sizeof(float));

        slice_axis_1_lstm(matmul_result, ig, batch_size, 4 * cell_size, 0, cell_size);
        slice_axis_1_lstm(matmul_result, cg, batch_size, 4 * cell_size, cell_size, 2 * cell_size);
        slice_axis_1_lstm(matmul_result, fg, batch_size, 4 * cell_size, 2 * cell_size, 3 * cell_size);
        slice_axis_1_lstm(matmul_result, og, batch_size, 4 * cell_size, 3 * cell_size, 4 * cell_size);

        // forget gate
        for(int i = 0; i < batch_cell_size; i++)
            fg[i] += forget_bias;

        // peephole
        if(w_f_data)
        {
            for(int i = 0; i < batch_size; i++)
                for(int j = 0; j < cell_size; j++)
                {
                    fg[i * cell_size + j] += init_c[i * cell_size + j] * w_f_data[j];
                    ig[i * cell_size + j] += init_c[i * cell_size + j] * w_i_data[j];
                }
        }

        sigmoid_lstm(fg, batch_cell_size);
        sigmoid_lstm(ig, batch_cell_size);
        mytanh_lstm(cg, batch_cell_size);

        // get cell output
        for(int i = 0; i < batch_cell_size; i++)
            init_c[i] = init_c[i] * fg[i] + cg[i] * ig[i];

        if(w_o_data)
        {
            for(int i = 0; i < batch_size; i++)
                for(int j = 0; j < cell_size; j++)
                {
                    og[i * cell_size + j] += init_c[i * cell_size + j] * w_o_data[j];
                }
        }

        sigmoid_lstm(og, batch_cell_size);

        if(projection)
        {
            for(int i = 0; i < batch_cell_size; i++)
            {
                og[i] = tanh(init_c[i]) * og[i];
            }

            /*batchxcell_size * cell_sizexhidden_size --> batch* hidden_size*/
            do_gemm_lstm(og, projection, init_h, batch_size, cell_size, hidden_size, cell_size, hidden_size,
                    hidden_size);
        }
        else
        {
            for(int i = 0; i < batch_cell_size; i++)
            {
                init_h[i] = tanh(init_c[i]) * og[i];
            }
        }

        // free memory
        free(merged_input);
        free(matmul_result);
        free(ig);
        free(cg);
        free(fg);
        free(og);
        return true;
    }
}

typedef int (*ref_lstm_t)(void* in_data, void* out_data, lstm_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_lstm_fp32.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
