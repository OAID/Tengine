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
 * Copyright (c) 2018, Open AI Lab
 * Author: zpluo@openailab.com
 */
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>

#include "graph.hpp"
#include "logger.hpp"
#include "node_ops.hpp"
#include "operator/lstm.hpp"
#include "data_type.hpp"
#include "tensor_mem.hpp"
#include "tengine_errno.hpp"
#include "compiler_fp16.h"
#include <cblas.h>
#include <math.h>

namespace TEngine {

namespace LSTMRefImpl {

struct LSTMOps : public NodeOps
{
    Tensor* init_c_tensor;
    Tensor* init_h_tensor;
    Tensor* bias_tensor;
    Tensor* w_f_tensor;
    Tensor* w_i_tensor;
    Tensor* w_o_tensor;
    Tensor* proj_tensor;
    Tensor* kernel_tensor;
    Tensor* h2h_kernel_tensor;
    Tensor* h2h_bias_tensor;
    Tensor* fused_kernel_tensor;
    void* init_h_data;
    void* init_c_data;
    // bool dynamic_shape;
    LSTMOps(void)
    {
        init_c_tensor = nullptr;
        init_h_tensor = nullptr;
        kernel_tensor = nullptr;
        bias_tensor = nullptr;
        w_f_tensor = nullptr;
        w_i_tensor = nullptr;
        w_o_tensor = nullptr;
        proj_tensor = nullptr;
        init_h_data = nullptr;
        init_c_data = nullptr;
        h2h_kernel_tensor = nullptr;
        h2h_bias_tensor = nullptr;
        fused_kernel_tensor = nullptr;
    }

    /*
    @ func_name: tanh
    */
    void mytanh(float* data, int size)
    {
        for(int i = 0; i < size; i++)
        {
            data[i] = tanh(data[i]);
        }
    }

    /*
    @ func_name: sigmoid
    */
    void sigmoid(float* data, int size)
    {
        for(int i = 0; i < size; i++)
        {
            data[i] = std::min(data[i], 30.0f);
            data[i] = std::max(data[i], -30.0f);

            data[i] = 1 / (1 + exp(-data[i]));
        }
    }
    void matmul(float* a, float* b, float* c, int m, int k, int n)
    {
        // const enum CBLAS_ORDER Order = CblasRowMajor;
        // const enum CBLAS_TRANSPOSE TransA = CblasNoTrans;
        // const enum CBLAS_TRANSPOSE TransB = CblasNoTrans;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b, n, 0.0, c, n);
    }

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

    void do_gemm(const float* a, const float* b, float* c, int m, int k, int n, int lda, int ldb, int ldc)
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
    }

    void do_gemm_mx(const float* a, const float* b, float* c, int m, int k, int n, int lda, int ldb, int ldc)
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
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
            do_gemm_mx(input, kernel, i2h, batch_size, input_size, 4 * cell_size, input_size, input_size,
                       4 * cell_size);

            if(bias)
            {
                for(int i = 0; i < batch_size; i++)
                    for(int j = 0; j < 4 * cell_size; j++)
                        i2h[i * 4 * cell_size + j] += bias[j];
            }

            do_gemm_mx(init_h, h2h_kernel, h2h, batch_size, hidden_size, 4 * hidden_size, hidden_size, hidden_size,
                       4 * hidden_size);
            if(h2h_bias)
            {
                for(int i = 0; i < batch_size; i++)
                    for(int j = 0; j < 4 * cell_size; j++)
                        h2h[i * 4 * cell_size + j] += h2h_bias[j];
            }

            for(int i = 0; i < batch_size * 4 * cell_size; i++)
                gates[i] = i2h[i] + h2h[i];

            slice_axis_1(gates, ig, batch_size, 4 * cell_size, 0, cell_size);
            slice_axis_1(gates, fg, batch_size, 4 * cell_size, cell_size, 2 * cell_size);
            slice_axis_1(gates, cg, batch_size, 4 * cell_size, 2 * cell_size, 3 * cell_size);
            slice_axis_1(gates, og, batch_size, 4 * cell_size, 3 * cell_size, 4 * cell_size);

            for(int i = 0; i < batch_size * cell_size; i++)
                fg[i] += 1;

            sigmoid(ig, batch_cell_size);
            sigmoid(fg, batch_cell_size);
            mytanh(cg, batch_cell_size);
            sigmoid(og, batch_cell_size);

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
            concat_axis_1(input, init_h, merged_input, batch_size, input_size, hidden_size);

            // do gemm
            do_gemm(merged_input, kernel, matmul_result, batch_size, input_total_size, 4 * cell_size, input_total_size,
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

            slice_axis_1(matmul_result, ig, batch_size, 4 * cell_size, 0, cell_size);
            slice_axis_1(matmul_result, cg, batch_size, 4 * cell_size, cell_size, 2 * cell_size);
            slice_axis_1(matmul_result, fg, batch_size, 4 * cell_size, 2 * cell_size, 3 * cell_size);
            slice_axis_1(matmul_result, og, batch_size, 4 * cell_size, 3 * cell_size, 4 * cell_size);

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

            sigmoid(fg, batch_cell_size);
            sigmoid(ig, batch_cell_size);
            mytanh(cg, batch_cell_size);

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

            sigmoid(og, batch_cell_size);

            if(projection)
            {
                for(int i = 0; i < batch_cell_size; i++)
                {
                    og[i] = tanh(init_c[i]) * og[i];
                }

                /*batchxcell_size * cell_sizexhidden_size --> batch* hidden_size*/
                do_gemm(og, projection, init_h, batch_size, cell_size, hidden_size, cell_size, hidden_size,
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
    bool do_LSTM_int8(Tensor* input_tensor, Tensor* output_tensor, void* init_c_data, void* init_h_data,
                      Tensor* kernel_tensor, Tensor* bias_tensor, Tensor* h2h_kernel_tensor, Tensor* h2h_bias_tensor,
                      bool has_peephole, Tensor* w_f_tensor, Tensor* w_i_tensor, Tensor* w_o_tensor,
                      Tensor* fused_kernel_tensor, bool has_projection, Tensor* proj_tensor, bool mxnet_flag,
                      float forget_bias, int seq_lens, int batch_size, int input_size, int output_len, int hidden_size,
                      int cell_size)
    {
        int8_t* output = ( int8_t* )get_tensor_mem(output_tensor);
        int8_t* input = ( int8_t* )get_tensor_mem(input_tensor);
        float* init_h = ( float* )malloc(batch_size * hidden_size * sizeof(float));

        if(init_h == nullptr)
        {
            set_tengine_errno(ENOMEM);
            return false;
        }

        float* init_c = ( float* )malloc(batch_size * cell_size * sizeof(float));

        if(init_c == nullptr)
        {
            free(init_h);
            set_tengine_errno(ENOMEM);
            return false;
        }

        if(init_h_data)
        {
            for(int i = 0; i < batch_size; i++)
            {
                memcpy(init_h + i * hidden_size, init_h_data, hidden_size * sizeof(float));
                memcpy(init_c + i * cell_size, init_c_data, cell_size * sizeof(float));
            }
        }
        else
        {
            memset(init_h, 0x0, sizeof(batch_size * hidden_size * sizeof(float)));
            memset(init_c, 0x0, sizeof(batch_size * cell_size * sizeof(float)));
        }

        int8_t* kernel = nullptr;
        int8_t* bias = nullptr;
        int8_t* w_f_data = nullptr;
        int8_t* w_i_data = nullptr;
        int8_t* w_o_data = nullptr;
        int8_t* projection = nullptr;
        int8_t* h2h_kernel = nullptr;
        int8_t* h2h_bias = nullptr;
        int8_t* fused_kernel = nullptr;

        if(kernel_tensor)
            kernel = ( int8_t* )get_tensor_mem(kernel_tensor);

        if(bias_tensor)
            bias = ( int8_t* )get_tensor_mem(bias_tensor);

        if(h2h_kernel_tensor)
            h2h_kernel = ( int8_t* )get_tensor_mem(h2h_kernel_tensor);

        if(h2h_bias_tensor)
            h2h_bias = ( int8_t* )get_tensor_mem(h2h_bias_tensor);

        if(has_peephole)
        {
            w_f_data = ( int8_t* )get_tensor_mem(w_f_tensor);
            w_i_data = ( int8_t* )get_tensor_mem(w_i_tensor);
            w_o_data = ( int8_t* )get_tensor_mem(w_o_tensor);
        }
        // int bsize=2*cell_size*4;

        if(fused_kernel_tensor)
        {
            fused_kernel = ( int8_t* )get_tensor_mem(fused_kernel_tensor);
            int kernel_size = get_tensor_mem_size(fused_kernel_tensor) / sizeof(int8_t);
            kernel = fused_kernel;
            h2h_kernel = kernel + input_size * hidden_size * 4;
            bias = kernel + kernel_size - hidden_size * 4 * 2;
            h2h_bias = bias + hidden_size * 4;
        }
        if(has_projection)
            projection = ( int8_t* )get_tensor_mem(proj_tensor);

        float* input_f = nullptr;
        float* kernel_f = nullptr;
        float* bias_f = nullptr;
        float* w_f_data_f = nullptr;
        float* w_i_data_f = nullptr;
        float* w_o_data_f = nullptr;
        float* projection_f = nullptr;
        float* h2h_kernel_f = nullptr;
        float* h2h_bias_f = nullptr;
        auto i_quant = input_tensor->GetQuantParam();
        float i_scale = (*i_quant)[0].scale;
        // in
        input_f = ( float* )std::malloc(seq_lens * batch_size * input_size * sizeof(float));
        for(int i = 0; i < seq_lens * batch_size * input_size; i++)
            input_f[i] = input[i] * i_scale;
        auto k_quant = kernel_tensor->GetQuantParam();
        float k_scale = (*k_quant)[0].scale;
        auto bias_quant = bias_tensor->GetQuantParam();
        float bias_scale = (*bias_quant)[0].scale;

        if(mxnet_flag == 1)
        {
            kernel_f = ( float* )std::malloc(input_size * cell_size * 4 * sizeof(float));
            for(int i = 0; i < input_size * cell_size * 4; i++)
                kernel_f[i] = kernel[i] * k_scale;

            auto h2h_k_quant = h2h_kernel_tensor->GetQuantParam();
            float h2h_k_scale = (*h2h_k_quant)[0].scale;

            h2h_kernel_f = ( float* )std::malloc(cell_size * cell_size * 4 * sizeof(float));
            for(int i = 0; i < hidden_size * cell_size * 4; i++)
                h2h_kernel_f[i] = h2h_kernel[i] * h2h_k_scale;

            bias_f = ( float* )std::malloc(cell_size * 4 * sizeof(float));
            for(int i = 0; i < cell_size * 4; i++)
                bias_f[i] = bias[i] * bias_scale;

            auto h2h_b_quant = bias_tensor->GetQuantParam();
            float h2h_b_scale = (*h2h_b_quant)[0].scale;

            h2h_bias_f = ( float* )std::malloc(cell_size * 4 * sizeof(float));
            for(int i = 0; i < cell_size * 4; i++)
                h2h_bias_f[i] = h2h_bias[i] * h2h_b_scale;
        }
        else
        {
            kernel_f = ( float* )std::malloc((input_size + cell_size) * cell_size * 4 * sizeof(float));
            for(int i = 0; i < (input_size + cell_size) * cell_size * 4; i++)
                kernel_f[i] = kernel[i] * k_scale;

            bias_f = ( float* )std::malloc(cell_size * 4 * sizeof(float));
            for(int i = 0; i < cell_size * 4; i++)
                bias_f[i] = bias[i] * bias_scale;

            if(has_peephole)
            {
                auto wf_quant = w_f_tensor->GetQuantParam();
                float wf_scale = (*wf_quant)[0].scale;
                w_f_data_f = ( float* )std::malloc(cell_size * sizeof(float));
                for(int i = 0; i < cell_size; i++)
                    w_f_data_f[i] = w_f_data[i] * wf_scale;

                auto wi_quant = w_i_tensor->GetQuantParam();
                float wi_scale = (*wi_quant)[0].scale;
                w_i_data_f = ( float* )std::malloc(cell_size * sizeof(float));
                for(int i = 0; i < cell_size; i++)
                    w_i_data_f[i] = w_i_data[i] * wi_scale;

                auto wo_quant = w_o_tensor->GetQuantParam();
                float wo_scale = (*wo_quant)[0].scale;
                w_o_data_f = ( float* )std::malloc(cell_size * sizeof(float));
                for(int i = 0; i < cell_size; i++)
                    w_o_data_f[i] = w_o_data[i] * wo_scale;
            }
            if(has_projection)
            {
                auto pf_quant = proj_tensor->GetQuantParam();
                float pf_scale = (*pf_quant)[0].scale;
                projection_f = ( float* )std::malloc(cell_size * hidden_size * sizeof(float));
                for(int i = 0; i < cell_size * hidden_size; i++)
                    projection_f[i] = projection[i] * pf_scale;
            }
        }
        int8_t* init_h_f = ( int8_t* )std::malloc(batch_size * hidden_size * sizeof(int8_t));
        for(int i = 0; i < seq_lens; i++)
        {
            const float* seq_input = input_f + i * batch_size * input_size;

            if(!do_LSTM_step(seq_input, init_h, init_c, kernel_f, bias_f, h2h_kernel_f, h2h_bias_f, w_f_data_f,
                             w_i_data_f, w_o_data_f, projection_f, forget_bias, batch_size, input_size, hidden_size,
                             cell_size, mxnet_flag))
                return false;

            float output_max = 0.0f;
            for(int i = 0; i < batch_size * hidden_size; i++)
            {
                if(output_max < fabs(init_h[i]))
                    output_max = fabs(init_h[i]);
            }

            auto* o_quant = output_tensor->GetQuantParam();
            QuantParam q_param;
            q_param.scale = output_max / 127;
            o_quant->resize(0);
            o_quant->push_back(q_param);

            for(int i = 0; i < batch_size * hidden_size; i++)
            {
                init_h_f[i] = round(init_h[i] * 127 / output_max);
            }
            if(i + output_len >= seq_lens)
            {
                memcpy(output, init_h_f, hidden_size * batch_size * sizeof(int8_t));
                output += batch_size * hidden_size;
            }
        }
        if(has_projection)
            free(projection_f);
        if(has_peephole)
        {
            free(w_f_data_f);
            free(w_i_data_f);
            free(w_o_data_f);
        }
        if(mxnet_flag)
        {
            free(h2h_kernel_f);
            free(h2h_bias_f);
        }
        free(kernel_f);
        free(bias_f);
        free(init_h_f);
        free(init_h);
        free(init_c);
        return true;
    }
    bool do_LSTM_fp16(Tensor* input_tensor, Tensor* output_tensor, void* init_c_data, void* init_h_data,
                      Tensor* kernel_tensor, Tensor* bias_tensor, Tensor* h2h_kernel_tensor, Tensor* h2h_bias_tensor,
                      bool has_peephole, Tensor* w_f_tensor, Tensor* w_i_tensor, Tensor* w_o_tensor,
                      Tensor* fused_kernel_tensor, bool has_projection, Tensor* proj_tensor, bool mxnet_flag,
                      float forget_bias, int seq_lens, int batch_size, int input_size, int output_len, int hidden_size,
                      int cell_size)
    {
#if !defined(__ARM_ARCH) || __ARM_ARCH < 8 || __ARM_ARCH >= 8
        __fp16* output = ( __fp16* )get_tensor_mem(output_tensor);
        __fp16* input = ( __fp16* )get_tensor_mem(input_tensor);
        float* init_h = ( float* )malloc(batch_size * hidden_size * sizeof(float));

        if(init_h == nullptr)
        {
            set_tengine_errno(ENOMEM);
            return false;
        }

        float* init_c = ( float* )malloc(batch_size * cell_size * sizeof(float));

        if(init_c == nullptr)
        {
            free(init_h);
            set_tengine_errno(ENOMEM);
            return false;
        }

        if(init_h_data)
        {
            for(int i = 0; i < batch_size; i++)
            {
                memcpy(init_h + i * hidden_size, init_h_data, hidden_size * sizeof(float));
                memcpy(init_c + i * cell_size, init_c_data, cell_size * sizeof(float));
            }
        }
        else
        {
            memset(init_h, 0x0, sizeof(batch_size * hidden_size * sizeof(float)));
            memset(init_c, 0x0, sizeof(batch_size * cell_size * sizeof(float)));
        }

        __fp16* kernel = nullptr;
        __fp16* bias = nullptr;
        __fp16* w_f_data = nullptr;
        __fp16* w_i_data = nullptr;
        __fp16* w_o_data = nullptr;
        __fp16* projection = nullptr;
        __fp16* h2h_kernel = nullptr;
        __fp16* h2h_bias = nullptr;
        __fp16* fused_kernel = nullptr;

        if(kernel_tensor)
            kernel = ( __fp16* )get_tensor_mem(kernel_tensor);

        if(bias_tensor)
            bias = ( __fp16* )get_tensor_mem(bias_tensor);

        if(h2h_kernel_tensor)
            h2h_kernel = ( __fp16* )get_tensor_mem(h2h_kernel_tensor);

        if(h2h_bias_tensor)
            h2h_bias = ( __fp16* )get_tensor_mem(h2h_bias_tensor);

        if(has_peephole)
        {
            w_f_data = ( __fp16* )get_tensor_mem(w_f_tensor);
            w_i_data = ( __fp16* )get_tensor_mem(w_i_tensor);
            w_o_data = ( __fp16* )get_tensor_mem(w_o_tensor);
        }
        // int bsize=2*cell_size*4;

        if(fused_kernel_tensor)
        {
            fused_kernel = ( __fp16* )get_tensor_mem(fused_kernel_tensor);
            int kernel_size = get_tensor_mem_size(fused_kernel_tensor) / sizeof(__fp16);
            kernel = fused_kernel;
            h2h_kernel = kernel + input_size * hidden_size * 4;
            bias = kernel + kernel_size - hidden_size * 4 * 2;
            h2h_bias = bias + hidden_size * 4;
        }
        if(has_projection)
            projection = ( __fp16* )get_tensor_mem(proj_tensor);

        float* input_f = nullptr;
        float* kernel_f = nullptr;
        float* bias_f = nullptr;
        float* w_f_data_f = nullptr;
        float* w_i_data_f = nullptr;
        float* w_o_data_f = nullptr;
        float* projection_f = nullptr;
        float* h2h_kernel_f = nullptr;
        float* h2h_bias_f = nullptr;

        // in
        input_f = ( float* )std::malloc(seq_lens * batch_size * input_size * sizeof(float));
        for(int i = 0; i < seq_lens * batch_size * input_size; i++)
            input_f[i] = fp16_to_fp32(input[i]);
        if(mxnet_flag == 1)
        {
            kernel_f = ( float* )std::malloc(input_size * cell_size * 4 * sizeof(float));
            for(int i = 0; i < input_size * cell_size * 4; i++)
                kernel_f[i] = fp16_to_fp32(kernel[i]);

            h2h_kernel_f = ( float* )std::malloc(cell_size * cell_size * 4 * sizeof(float));
            for(int i = 0; i < hidden_size * cell_size * 4; i++)
                h2h_kernel_f[i] = fp16_to_fp32(h2h_kernel[i]);

            bias_f = ( float* )std::malloc(cell_size * 4 * sizeof(float));
            for(int i = 0; i < cell_size * 4; i++)
                bias_f[i] = fp16_to_fp32(bias[i]);

            h2h_bias_f = ( float* )std::malloc(cell_size * 4 * sizeof(float));
            for(int i = 0; i < cell_size * 4; i++)
                h2h_bias_f[i] = fp16_to_fp32(h2h_bias[i]);
        }
        else
        {
            kernel_f = ( float* )std::malloc((input_size + cell_size) * cell_size * 4 * sizeof(float));
            for(int i = 0; i < (input_size + cell_size) * cell_size * 4; i++)
                kernel_f[i] = fp16_to_fp32(kernel[i]);

            bias_f = ( float* )std::malloc(cell_size * 4 * sizeof(float));
            for(int i = 0; i < cell_size * 4; i++)
                bias_f[i] = fp16_to_fp32(bias[i]);

            if(has_peephole)
            {
                w_f_data_f = ( float* )std::malloc(cell_size * sizeof(float));
                for(int i = 0; i < cell_size; i++)
                    w_f_data_f[i] = fp16_to_fp32(w_f_data[i]);
                w_i_data_f = ( float* )std::malloc(cell_size * sizeof(float));
                for(int i = 0; i < cell_size; i++)
                    w_i_data_f[i] = fp16_to_fp32(w_i_data[i]);
                w_o_data_f = ( float* )std::malloc(cell_size * sizeof(float));
                for(int i = 0; i < cell_size; i++)
                    w_o_data_f[i] = fp16_to_fp32(w_o_data[i]);
            }
            if(has_projection)
            {
                projection_f = ( float* )std::malloc(cell_size * hidden_size * sizeof(float));
                for(int i = 0; i < cell_size * hidden_size; i++)
                    projection_f[i] = fp16_to_fp32(projection[i]);
            }
        }
        __fp16* init_h_f = ( __fp16* )std::malloc(batch_size * hidden_size * sizeof(__fp16));
        for(int i = 0; i < seq_lens; i++)
        {
            const float* seq_input = input_f + i * batch_size * input_size;

            if(!do_LSTM_step(seq_input, init_h, init_c, kernel_f, bias_f, h2h_kernel_f, h2h_bias_f, w_f_data_f,
                             w_i_data_f, w_o_data_f, projection_f, forget_bias, batch_size, input_size, hidden_size,
                             cell_size, mxnet_flag))
                return false;
            for(int i = 0; i < batch_size * hidden_size; i++)
            {
                init_h_f[i] = fp32_to_fp16(init_h[i]);
            }
            if(i + output_len >= seq_lens)
            {
                memcpy(output, init_h_f, hidden_size * batch_size * sizeof(__fp16));
                output += batch_size * hidden_size;
            }
        }
        if(has_projection)
            free(projection_f);
        if(has_peephole)
        {
            free(w_f_data_f);
            free(w_i_data_f);
            free(w_o_data_f);
        }
        if(mxnet_flag)
        {
            free(h2h_kernel_f);
            free(h2h_bias_f);
        }
        free(kernel_f);
        free(bias_f);
        free(init_h_f);
        free(init_h);
        free(init_c);
        return true;
#endif
    }
    bool do_LSTM(Tensor* input_tensor, Tensor* output_tensor, void* init_c_data, void* init_h_data,
                 Tensor* kernel_tensor, Tensor* bias_tensor, Tensor* h2h_kernel_tensor, Tensor* h2h_bias_tensor,
                 bool has_peephole, Tensor* w_f_tensor, Tensor* w_i_tensor, Tensor* w_o_tensor,
                 Tensor* fused_kernel_tensor, bool has_projection, Tensor* proj_tensor, bool mxnet_flag,
                 float forget_bias, int seq_lens, int batch_size, int input_size, int output_len, int hidden_size,
                 int cell_size)
    {
        float* output = ( float* )get_tensor_mem(output_tensor);
        float* input = ( float* )get_tensor_mem(input_tensor);
        float* init_h = ( float* )malloc(batch_size * hidden_size * sizeof(float));

        if(init_h == nullptr)
        {
            set_tengine_errno(ENOMEM);
            return false;
        }

        float* init_c = ( float* )malloc(batch_size * cell_size * sizeof(float));

        if(init_c == nullptr)
        {
            free(init_h);
            set_tengine_errno(ENOMEM);
            return false;
        }

        if(init_h_data)
        {
            for(int i = 0; i < batch_size; i++)
            {
                memcpy(init_h + i * hidden_size, init_h_data, hidden_size * sizeof(float));
                memcpy(init_c + i * cell_size, init_c_data, cell_size * sizeof(float));
            }
        }
        else
        {
            memset(init_h, 0x0, sizeof(batch_size * hidden_size * sizeof(float)));
            memset(init_c, 0x0, sizeof(batch_size * cell_size * sizeof(float)));
        }

        float* kernel = nullptr;
        float* bias = nullptr;
        float* w_f_data = nullptr;
        float* w_i_data = nullptr;
        float* w_o_data = nullptr;
        float* projection = nullptr;
        float* h2h_kernel = nullptr;
        float* h2h_bias = nullptr;
        float* fused_kernel = nullptr;

        if(kernel_tensor)
            kernel = ( float* )get_tensor_mem(kernel_tensor);

        if(bias_tensor)
            bias = ( float* )get_tensor_mem(bias_tensor);

        if(h2h_kernel_tensor)
            h2h_kernel = ( float* )get_tensor_mem(h2h_kernel_tensor);

        if(h2h_bias_tensor)
            h2h_bias = ( float* )get_tensor_mem(h2h_bias_tensor);

        if(has_peephole)
        {
            w_f_data = ( float* )get_tensor_mem(w_f_tensor);
            w_i_data = ( float* )get_tensor_mem(w_i_tensor);
            w_o_data = ( float* )get_tensor_mem(w_o_tensor);
        }
        // int bsize=2*cell_size*4;

        if(fused_kernel_tensor)
        {
            fused_kernel = ( float* )get_tensor_mem(fused_kernel_tensor);
            int kernel_size = get_tensor_mem_size(fused_kernel_tensor) / sizeof(float);
            kernel = fused_kernel;
            h2h_kernel = kernel + input_size * hidden_size * 4;
            bias = kernel + kernel_size - hidden_size * 4 * 2;
            h2h_bias = bias + hidden_size * 4;
        }
        if(has_projection)
            projection = ( float* )get_tensor_mem(proj_tensor);

        for(int i = 0; i < seq_lens; i++)
        {
            const float* seq_input = input + i * batch_size * input_size;

            if(!do_LSTM_step(seq_input, init_h, init_c, kernel, bias, h2h_kernel, h2h_bias, w_f_data, w_i_data,
                             w_o_data, projection, forget_bias, batch_size, input_size, hidden_size, cell_size,
                             mxnet_flag))
                return false;

            if(i + output_len >= seq_lens)
            {
                memcpy(output, init_h, hidden_size * batch_size * sizeof(float));
                output += batch_size * hidden_size;
            }
        }
        free(init_h);
        free(init_c);
        return true;
    }

    bool Prerun(Node* node)
    {
        LSTM* lstm_op = dynamic_cast<LSTM*>(node->GetOp());

        int in_num = node->GetInputNum();

        for(int count = 0; count < in_num; count++)
        {
            Tensor* temptensor = node->GetInputTensor(count);
            const std::string& name = temptensor->GetName();
            if(name.find(lstm_op->GetKernelName()) != std::string::npos &&
               name.find(lstm_op->GetProjectionName()) == std::string::npos)
            {
                kernel_tensor = temptensor;
            }
            if(name.find(lstm_op->GetInitCellName()) != std::string::npos)
            {
                init_c_tensor = temptensor;
            }
            if(name.find(lstm_op->GetInitHiddenName()) != std::string::npos)
            {
                init_h_tensor = temptensor;
            }
            if(name.find(lstm_op->GetBiasName()) != std::string::npos)
            {
                bias_tensor = temptensor;
            }
            if(name.find(lstm_op->GetPeepholeForgetName()) != std::string::npos)
            {
                w_f_tensor = temptensor;
            }
            if(name.find(lstm_op->GetPeepholeOutputName()) != std::string::npos)
            {
                w_o_tensor = temptensor;
            }
            if(name.find(lstm_op->GetPeepholeInputName()) != std::string::npos)
            {
                w_i_tensor = temptensor;
            }
            if(name.find(lstm_op->GetProjectionName()) != std::string::npos)
            {
                proj_tensor = temptensor;
            }
            if(name.find(lstm_op->Geti2hKernelName()) != std::string::npos)
            {
                kernel_tensor = temptensor;
            }
            if(name.find(lstm_op->Geti2hBiasName()) != std::string::npos)
            {
                bias_tensor = temptensor;
            }
            if(name.find(lstm_op->Geth2hKernelName()) != std::string::npos)
            {
                h2h_kernel_tensor = temptensor;
            }
            if(name.find(lstm_op->Geth2hBiasName()) != std::string::npos)
            {
                h2h_bias_tensor = temptensor;
            }
            if(name.find(lstm_op->GetFusedKernelName()) != std::string::npos)
            {
                fused_kernel_tensor = temptensor;
            }
        }

        if(init_c_tensor)
        {
            init_c_data = get_tensor_mem(init_c_tensor);
        }

        if(init_h_tensor)
        {
            init_h_data = get_tensor_mem(init_h_tensor);
        }

        return true;
    }

    bool Run(Node* node)
    {
        LSTM* lstm_op = dynamic_cast<LSTM*>(node->GetOp());
        LSTMParam* param = lstm_op->GetParam();

        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        // Tensor* kernel_tensor = node->GetInputTensor(1);
        int element_size = DataType::GetTypeSize(input_tensor->GetDataType());
        float forget_bias = param->forget_bias;

        bool has_peephole = param->has_peephole;
        bool has_projection = param->has_projection;

        int hidden_size = param->hidden_size;
        int cell_size = param->cell_size;
        int input_size = 0;

        const TShape& input_shape = input_tensor->GetShape();

        int seq_lens = input_shape.Shape(1);
        int batch_size = input_shape.Shape(0);
        int output_len = param->output_len;
        int mxnet_flag = param->mxnet_flag;

        if(mxnet_flag == 1)
        {
            seq_lens = input_shape.Shape(0);
            batch_size = input_shape.Shape(1);
            input_size = input_shape.Shape(2);
        }
        else
        {
            input_size = param->input_size;
        }

        // std::cout<<"inputmem: "<<input<<"\n";
        if(element_size == 4)
        {
            bool ret = do_LSTM(input_tensor, output_tensor, init_c_data, init_h_data, kernel_tensor, bias_tensor,
                               h2h_kernel_tensor, h2h_bias_tensor, has_peephole, w_f_tensor, w_i_tensor, w_o_tensor,
                               fused_kernel_tensor, has_projection, proj_tensor, mxnet_flag, forget_bias, seq_lens,
                               batch_size, input_size, output_len, hidden_size, cell_size);

            return ret;
        }
        else if(element_size == 2)
        {
            bool ret = do_LSTM_fp16(input_tensor, output_tensor, init_c_data, init_h_data, kernel_tensor, bias_tensor,
                                    h2h_kernel_tensor, h2h_bias_tensor, has_peephole, w_f_tensor, w_i_tensor,
                                    w_o_tensor, fused_kernel_tensor, has_projection, proj_tensor, mxnet_flag,
                                    forget_bias, seq_lens, batch_size, input_size, output_len, hidden_size, cell_size);

            return ret;
        }
        else if(element_size == 1)
        {
            bool ret = do_LSTM_int8(input_tensor, output_tensor, init_c_data, init_h_data, kernel_tensor, bias_tensor,
                                    h2h_kernel_tensor, h2h_bias_tensor, has_peephole, w_f_tensor, w_i_tensor,
                                    w_o_tensor, fused_kernel_tensor, has_projection, proj_tensor, mxnet_flag,
                                    forget_bias, seq_lens, batch_size, input_size, output_len, hidden_size, cell_size);

            return ret;
        }
        return false;
    }

    bool Postrun(Node* node)
    {
        return true;
    }
};

}    // namespace LSTMRefImpl

using namespace LSTMRefImpl;
void RegisterLSTMNodeExec(void)
{
    LSTMOps* ops = new LSTMOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "LSTM", ops);
}

}    // namespace TEngine
