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
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>

#include "graph.hpp"
#include "logger.hpp"
#include "node_ops.hpp"
#include "operator/rnn.hpp"
#include "tensor_mem.hpp"
#include "tengine_errno.hpp"
#include <cblas.h>
#include <math.h>

namespace TEngine {

namespace RNNRefImpl {

struct RNNOps : public NodeOps
{
    Tensor* init_h_tensor;
    Tensor* bias_tensor;
    void* init_h_data;

    RNNOps(void)
    {
        init_h_tensor = nullptr;
        bias_tensor = nullptr;
        init_h_data = nullptr;
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

    bool do_RNN_step(const float* input, float* init_h, const float* kernel, const float* bias,
                      int batch_size, int input_size, int hidden_size)
    {
        int input_total_size = input_size + hidden_size;
        int batch_cell_size = hidden_size * batch_size;

        float* ig = ( float* )malloc(batch_cell_size * sizeof(float));

        float* merged_input = ( float* )malloc(sizeof(float) * batch_size * (input_total_size));
        float* matmul_result = ( float* )malloc(sizeof(float) * batch_size * hidden_size );

        // merge input
        concat_axis_1(input, init_h, merged_input, batch_size, input_size, hidden_size);

        // do gemm
        do_gemm(merged_input, kernel, matmul_result, batch_size, input_total_size, hidden_size, input_total_size,
                hidden_size, hidden_size);

        // add bias
        if(bias)
        {
            for(int i = 0; i < batch_size; i++)
                for(int j = 0; j < hidden_size; j++)
                    matmul_result[i *hidden_size + j] += bias[j];
        }
        //activation
        for(int i = 0; i < batch_cell_size; i++)
        {
            ig[i] = tanh(matmul_result[i]);
            init_h[i]=ig[i];
        }

        // free memory
        free(merged_input);
        free(matmul_result);
        free(ig);

        return true;
    }

    bool do_RNN(const float* input, float* output, float* init_h, const float* kernel,
                 const float* bias, int seq_lens, int batch_size, int input_size,int output_len, int hidden_size)
    {
        for(int i = 0; i < seq_lens; i++)
        {
            const float* seq_input = input + i * batch_size * input_size;

            if(!do_RNN_step(seq_input, init_h, kernel, bias, batch_size, input_size, hidden_size))
                return false;
            //outputs [batch_size,seq_len,hidden_size]
            //final_state [batch_size,hidden_size]   
            if(i + output_len >= seq_lens)
            {
                memcpy(output, init_h, batch_size*hidden_size * sizeof(float));
                output += batch_size*hidden_size;
            }
        }

        return true;
    }

    bool Prerun(Node* node)
    {
        RNN* rnn_op = dynamic_cast<RNN*>(node->GetOp());

        int in_num = node->GetInputNum();

        for(int count = 0; count < in_num; count++)
        {
            Tensor* temptensor = node->GetInputTensor(count);
            const std::string& name = temptensor->GetName();

            if(name.find(rnn_op->GetInitHiddenName()) != std::string::npos)
            {
                init_h_tensor = temptensor;
            }
            if(name.find(rnn_op->GetBiasName()) != std::string::npos)
            {
                bias_tensor = temptensor;
            }
           
        }

        if(init_h_tensor)
        {
            init_h_data = get_tensor_mem(init_h_tensor);
        }

        return true;
    }

    bool Run(Node* node)
    {
        RNN* rnn_op = dynamic_cast<RNN*>(node->GetOp());
        RNNParam* param = rnn_op->GetParam();

        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        Tensor* kernel_tensor = node->GetInputTensor(1);

        int input_size = param->input_size;
        int hidden_size = param->hidden_size;

        float* output = ( float* )get_tensor_mem(output_tensor);
        float* input = ( float* )get_tensor_mem(input_tensor);

        const TShape& input_shape = input_tensor->GetShape();

        int seq_lens = input_shape.Shape(0);
        int batch_size = input_shape.Shape(1);
        int output_len = param->output_len;

        float* init_h = ( float* )malloc(batch_size * hidden_size * sizeof(float));

        if(init_h == nullptr)
        {
            set_tengine_errno(ENOMEM);
            return false;
        }

        if(init_h_data)
        {
            for(int i = 0; i < batch_size; i++)
            {
                memcpy(init_h + i * hidden_size, init_h_data, hidden_size * sizeof(float));
            }
        }
        else
        {
            memset(init_h, 0x0, sizeof(batch_size * hidden_size * sizeof(float)));
        }

        float* kernel = ( float* )get_tensor_mem(kernel_tensor);

        float* bias = nullptr;
  
        if(bias_tensor)
            bias = ( float* )get_tensor_mem(bias_tensor);

        bool ret = do_RNN(input, output, init_h, kernel, bias, seq_lens, batch_size, input_size, output_len, hidden_size);

        free(init_h);

        return ret;
    }

    bool Postrun(Node* node)
    {
        return true;
    }
};

}    // namespace RNNRefImpl

using namespace RNNRefImpl;
void RegisterRNNNodeExec(void)
{
    RNNOps* ops = new RNNOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "RNN", ops);
}

}    // namespace TEngine