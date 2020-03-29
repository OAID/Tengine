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
 * Author: haoluo@openailab.com
 */

#include <vector>
#include <math.h>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/gru.hpp"
#include "kernel/gru/ref_gru_kernel.h"

namespace TEngine {

namespace RefGRUOps {

struct RefGRU : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    gru_param param_gru;
    ref_gru_t kernel_run;
    KernelRegistry<ref_gru_t> kernel_registry;

    Tensor* init_h_tensor;
    Tensor* kernel_tensor;
    Tensor* bias_tensor;
    Tensor* candidate_kernel_tensor;
    Tensor* candidate_bias_tensor;
    Tensor* fused_kernel_tensor;
    // bool dynamic_shape;
    float* init_h_data;

    RefGRU(void)
    {
        init_h_tensor = nullptr;
        bias_tensor = nullptr;
        init_h_data = nullptr;
        kernel_tensor = nullptr;
        candidate_kernel_tensor = nullptr;
        candidate_bias_tensor = nullptr;
        fused_kernel_tensor = nullptr;
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool RefGRU::Prerun(Node* node)
{

    GRU* gru_op = dynamic_cast<GRU*>(node->GetOp());

    int in_num = node->GetInputNum();

    for(int count = 0; count < in_num  ; count++)
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
        init_h_data = (float*)get_tensor_mem(init_h_tensor);
    }
    
    Tensor* input = node->GetInputTensor(0);

    // int weight_out = weight->GetShape().Shape(0);
    // if(weight_out == param.out_number)
    //     param.need_trans = 0;
    // else
    //     param.need_trans = 1;


    int layout = exec_attr->graph_layout;
    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefGRU::Run(Node* node)
{
    GRU* gru_op = dynamic_cast<GRU*>(node->GetOp());
    GRUParam* param = gru_op->GetParam();

    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    // Tensor* kernel_tensor = node->GetInputTensor(1);
    // int element_size = DataType::GetTypeSize(input_tensor->GetDataType());
    int input_size = 0;
    int hidden_size = param->hidden_size;

    const TShape& input_shape = input_tensor->GetShape();

    int seq_lens = input_shape.Shape(1);
    int batch_size = input_shape.Shape(0);
    int output_len = param->output_len;
    int mxnet_flag = param->mxnet_flag;

    if(mxnet_flag == 1)
    {
        input_size = input_shape.Shape(2);
        seq_lens = input_shape.Shape(0);
        batch_size = input_shape.Shape(1);
        // kernel_tensor = node->GetInputTensor(1);
    }
    else
    {
        input_size = param->input_size;
    }
    float* kernel = nullptr;
    float* bias = nullptr;
    float* fused_kernel = nullptr;
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
        fused_kernel = ( float* )get_tensor_mem(fused_kernel_tensor);
        kernel = fused_kernel;
        candidate_kernel = fused_kernel + input_size * hidden_size * 3;
        bias = candidate_kernel + hidden_size * hidden_size * 3;
        candidate_bias = bias + hidden_size * 3;
    }

    void* output = ( void* )get_tensor_mem(output_tensor);
        // std::cout<<"ot::"<<output<<"\n";
    void* input = ( void* )get_tensor_mem(input_tensor);

    param_gru.init_h_data=init_h_data;
    param_gru.bias=bias;
    param_gru.kernel=kernel;
    param_gru.candidate_kernel=candidate_kernel;
    param_gru.candidate_bias=candidate_bias;
    param_gru.fused_kernel=fused_kernel;
    param_gru.seq_lens=seq_lens;
    param_gru.batch_size=batch_size;
    param_gru.input_size=input_size;
    param_gru.output_len=output_len;
    param_gru.hidden_size=hidden_size;
    param_gru.mxnet_flag=mxnet_flag;

    if(kernel_run(input, output, &param_gru) < 0)
        return false;

    return true;
}

void RefGRU::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_gru_t )ref_gru_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_gru_t )ref_gru_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefGRU* ops = new RefGRU();

    LOG_DEBUG() << "Demo RefGRUOpOp is selected\n";

    return ops;
}

}    // namespace RefGRUOps

void RegisterRefGRUOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "GRU", RefGRUOps::SelectFunc, 1000);
}

}    // namespace TEngine
