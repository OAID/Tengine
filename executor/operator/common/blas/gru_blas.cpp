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
#include "operator/gru.hpp"
#include "tensor_mem.hpp"
#include "tengine_errno.hpp"
#include <cblas.h>
#include <math.h>

namespace TEngine {

namespace GRURefImpl {

struct GRUOps : public NodeOps
{
    Tensor* init_h_tensor;
    Tensor* kernel_tensor;
    Tensor* bias_tensor;
    Tensor* candidate_kernel_tensor;
    Tensor* candidate_bias_tensor;
    Tensor* fused_kernel_tensor;
    // bool dynamic_shape;
    void* init_h_data;

    GRUOps(void)
    {
        init_h_tensor = nullptr;
        bias_tensor = nullptr;
        init_h_data = nullptr;
        kernel_tensor=nullptr;
        candidate_kernel_tensor=nullptr;
        candidate_bias_tensor=nullptr;
        fused_kernel_tensor=nullptr;
    }

    void sigmoid(float* data, int size)
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
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
    }
    void do_gemm_mx(const float* a, const float* b, float* c, int m, int k, int n, int lda, int ldb, int ldc)
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
    }

    bool do_GRU_step(const float* input, float* init_h, const float* kernel, const float* bias,
                      const float* candidate_kernel,const float* candidate_bias,int batch_size, 
                      int input_size, int hidden_size,int mxnet_flag)
    {

        if(mxnet_flag==1)
        {
            float* i2h_mat = ( float* )malloc(sizeof(float) * batch_size *3* hidden_size);
            float* h2h_mat = ( float* )malloc(sizeof(float) * batch_size *3* hidden_size);
            
            float* i2h_r = ( float* )malloc(batch_size*hidden_size * sizeof(float));
            float* i2h_z = ( float* )malloc(batch_size*hidden_size * sizeof(float));
            float* i2h = ( float* )malloc(batch_size*hidden_size * sizeof(float));

            float* h2h_r = ( float* )malloc(batch_size*hidden_size * sizeof(float));
            float* h2h_z = ( float* )malloc(batch_size*hidden_size * sizeof(float));
            float* h2h = ( float* )malloc(batch_size*hidden_size * sizeof(float));

            float* r_g = ( float* )malloc(batch_size*hidden_size * sizeof(float));
            float* u_g = ( float* )malloc(batch_size*hidden_size * sizeof(float));
            float* next_h_tmp = ( float* )malloc(batch_size*hidden_size * sizeof(float));

            do_gemm_mx(input, kernel, i2h_mat, batch_size, input_size, 3*hidden_size, input_size,
                    input_size, 3*hidden_size);

            for(int i = 0; i < batch_size; i++)
            {
                for(int j = 0; j < (3*hidden_size); j++)
                {
                    i2h_mat[i *(3*hidden_size) + j] += bias[j];
                }
            }

            do_gemm_mx(init_h, candidate_kernel, h2h_mat, batch_size, hidden_size, 3*hidden_size, hidden_size,
                    hidden_size, 3*hidden_size);
            
            for(int i = 0; i < batch_size; i++)
            {
                for(int j = 0; j < (3*hidden_size); j++)
                {
                    h2h_mat[i *(3*hidden_size) + j] += candidate_bias[j];
                }
            }
            slice_axis_1(i2h_mat, i2h_r, batch_size, 3 * hidden_size, 0, hidden_size);
            slice_axis_1(i2h_mat, i2h_z, batch_size, 3 * hidden_size, hidden_size, 2*hidden_size);
            slice_axis_1(i2h_mat, i2h, batch_size, 3 * hidden_size, 2*hidden_size, 3*hidden_size);

            slice_axis_1(h2h_mat, h2h_r, batch_size, 3 * hidden_size, 0, hidden_size);
            slice_axis_1(h2h_mat, h2h_z, batch_size, 3 * hidden_size, hidden_size, 2*hidden_size);
            slice_axis_1(h2h_mat, h2h, batch_size, 3 * hidden_size, 2*hidden_size, 3*hidden_size);

            for(int i = 0; i < batch_size*hidden_size; i++)
            {
                r_g[i] = i2h_r[i]+h2h_r[i];
            }
            sigmoid(r_g,hidden_size * batch_size);
            for(int i = 0; i < batch_size*hidden_size; i++)
            {
                u_g[i] = i2h_z[i]+h2h_z[i];
            }
            sigmoid(u_g,hidden_size * batch_size);

            for(int i = 0; i < batch_size*hidden_size; i++)
            {
                next_h_tmp[i] = tanh(i2h[i]+r_g[i]*h2h[i]);
            }

            for(int i = 0; i < batch_size*hidden_size; i++)
            {
                init_h[i] = u_g[i] * init_h[i] + (1-u_g[i]) * next_h_tmp[i];
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
            float* matmul_result = ( float* )malloc(sizeof(float) * batch_size *2* hidden_size );
            float* r = ( float* )malloc(batch_cell_size * sizeof(float));
            float* u = ( float* )malloc(batch_cell_size * sizeof(float));
            float* c = ( float* )malloc(batch_cell_size * sizeof(float));
            float* r_state = ( float* )malloc(batch_cell_size * sizeof(float));
            float* candidate = ( float* )malloc(sizeof(float) * batch_size* hidden_size);

            // merge input
            concat_axis_1(input, init_h, merged_input, batch_size, input_size, hidden_size);
            // do gemm
            do_gemm(merged_input, kernel, matmul_result, batch_size, input_total_size, 2*hidden_size, input_total_size,
                    2*hidden_size, 2*hidden_size);
            // add bias
            
            
            for(int i = 0; i < batch_size; i++)
            {
                for(int j = 0; j < (2*hidden_size); j++)
                {
                    matmul_result[i *(2*hidden_size) + j] += bias[j];
                }
                
            }
            

            sigmoid(matmul_result,2*hidden_size * batch_size);
            slice_axis_1(matmul_result, r, batch_size, 2 * hidden_size, 0, hidden_size);
            slice_axis_1(matmul_result, u, batch_size, 2 * hidden_size, hidden_size, 2*hidden_size);


            for(int i = 0; i < batch_cell_size; i++)
                r_state[i] = r[i] * init_h[i];
            
            concat_axis_1(input, r_state, merged_input, batch_size, input_size, hidden_size);
            //candidate kernerl


            do_gemm(merged_input, candidate_kernel, candidate, batch_size, input_total_size, hidden_size, input_total_size,
                    hidden_size, hidden_size);
            //candidate bias
            
            for(int i = 0; i < batch_size; i++)
            {
                for(int j = 0; j < hidden_size; j++)
                {
                    candidate[i *hidden_size + j] += candidate_bias[j];
                }
            }
            

            for(int i = 0; i < batch_cell_size; i++)
            {
                c[i] = tanh(candidate[i]);
            }

            for(int i = 0; i < batch_cell_size; i++)
            {
                init_h[i] = u[i] * init_h[i] + (1-u[i]) * c[i];
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

    bool do_GRU(const float* input, float* output, float* init_h, const float* kernel,
                 const float* bias,const float* candidate_kernel,const float* candidate_bias, 
                 int seq_lens, int batch_size, int input_size,int output_len, int hidden_size,int mxnet_flag)
    {
        for(int i = 0; i < seq_lens; i++)
        {

            const float* seq_input = input + i * batch_size * input_size;
            if(!do_GRU_step(seq_input, init_h, kernel, bias, candidate_kernel,candidate_bias,batch_size, input_size, hidden_size,mxnet_flag))
            {   
                return false;
            }

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
        GRU* gru_op = dynamic_cast<GRU*>(node->GetOp());

        int in_num = node->GetInputNum();

        for(int count = 0; count < in_num; count++)
        {
            Tensor* temptensor = node->GetInputTensor(count);
            const std::string& name = temptensor->GetName();

            if(name.find(gru_op->GetInitHiddenName()) != std::string::npos)
            {
                init_h_tensor = temptensor;
            }
            if(name.find(gru_op->GetBiasName()) != std::string::npos)
            {
                bias_tensor = temptensor;
            }
            if(name.find(gru_op->GetKernelName()) != std::string::npos)
            {
                kernel_tensor = temptensor;
            }
            if(name.find(gru_op->GetCandidateKernelName()) != std::string::npos)
            {
                candidate_kernel_tensor = temptensor;
            }
            if(name.find(gru_op->GetCandidateBiasName()) != std::string::npos)
            {
                candidate_bias_tensor = temptensor;
            }
            if(name.find(gru_op->Geti2hweightName()) != std::string::npos)
            {
                kernel_tensor = temptensor;
            }
            if(name.find(gru_op->Geti2hbiasName()) != std::string::npos)
            {
                bias_tensor = temptensor;
            }
            if(name.find(gru_op->Geth2hweightName()) != std::string::npos)
            {
                candidate_kernel_tensor = temptensor;
            }
            if(name.find(gru_op->Geth2hbiasName()) != std::string::npos)
            {
                candidate_bias_tensor = temptensor;
            }
            if(name.find(gru_op->GetFusedKernelName()) != std::string::npos)
            {
                fused_kernel_tensor = temptensor;
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
        GRU* gru_op = dynamic_cast<GRU*>(node->GetOp());
        GRUParam* param = gru_op->GetParam();

        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        // Tensor* kernel_tensor = node->GetInputTensor(1);
        


        int input_size = 0;
        int hidden_size = param->hidden_size;

        float* output = ( float* )get_tensor_mem(output_tensor);
        // std::cout<<"ot::"<<output<<"\n";
        float* input = ( float* )get_tensor_mem(input_tensor);

        const TShape& input_shape = input_tensor->GetShape();

        int seq_lens = input_shape.Shape(0);
        int batch_size = input_shape.Shape(1);
        int output_len = param->output_len;
        int mxnet_flag = param->mxnet_flag;

        if(mxnet_flag==1)
        {
            input_size=input_shape.Shape(2);
            // kernel_tensor = node->GetInputTensor(1);
        }
        else
        {
            input_size = param->input_size;
        }
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

        float* kernel = nullptr;
        float* bias = nullptr;
        float* fused_kernel=nullptr;
        float* candidate_kernel = nullptr;
        float* candidate_bias = nullptr;
        
        if(kernel_tensor)
            kernel = ( float* )get_tensor_mem(kernel_tensor);
        
        if(bias_tensor)
            bias = ( float* )get_tensor_mem(bias_tensor);
            
        if(candidate_kernel_tensor)
            candidate_kernel = ( float* )get_tensor_mem(candidate_kernel_tensor);
        
        if(candidate_bias_tensor)
            candidate_bias = ( float* )get_tensor_mem(candidate_bias_tensor);
        
        if(fused_kernel_tensor)
        {
            // std::cout<<"fused_kernel\n";
            fused_kernel=( float* )get_tensor_mem(fused_kernel_tensor);
            kernel=fused_kernel;
            candidate_kernel=fused_kernel+input_size*hidden_size*3;
            bias=candidate_kernel+hidden_size*hidden_size*3;
            candidate_bias=bias+hidden_size*3;
        }

        bool ret = do_GRU(input, output, init_h, kernel, bias, candidate_kernel
        ,candidate_bias,seq_lens, batch_size, input_size, output_len, hidden_size,mxnet_flag);

        free(init_h);
        return ret;
    }

    bool Postrun(Node* node)
    {
        return true;
    }
};

}    // namespace GRURefImpl

using namespace GRURefImpl;
void RegisterGRUNodeExec(void)
{
    GRUOps* ops = new GRUOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "GRU", ops);
}

}    // namespace TEngine
