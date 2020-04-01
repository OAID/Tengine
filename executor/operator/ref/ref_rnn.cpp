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
#include "operator/rnn.hpp"
#include "kernel/rnn/ref_rnn_kernel.h"

namespace TEngine {

namespace RefRNNOps {

struct RefRNN : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    rnn_param param_rnn;
    ref_rnn_t kernel_run;
    KernelRegistry<ref_rnn_t> kernel_registry;

    Tensor* init_h_tensor;
    Tensor* bias_tensor;
    float* init_h_data;

    RefRNN(void)
    {
        init_h_tensor = nullptr;
        bias_tensor = nullptr;
        init_h_data = nullptr;
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool RefRNN::Prerun(Node* node)
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
        init_h_data = (float* )get_tensor_mem(init_h_tensor);
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

bool RefRNN::Run(Node* node)
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



    param_rnn.init_h_data=init_h_data;
    param_rnn.bias=bias;
    param_rnn.kernel=kernel;
    param_rnn.seq_lens=seq_lens;
    param_rnn.batch_size=batch_size;
    param_rnn.input_size=input_size;
    param_rnn.output_len=output_len;
    param_rnn.hidden_size=hidden_size;

    if(kernel_run(input, output, &param_rnn) < 0)
        return false;

    return true;
}

void RefRNN::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_rnn_t )ref_rnn_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_rnn_t )ref_rnn_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefRNN* ops = new RefRNN();

    LOG_DEBUG() << "Demo RefGRUOpOp is selected\n";

    return ops;
}

}    // namespace RefRNNOps

void RegisterRefRNNOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "RNN", RefRNNOps::SelectFunc, 1000);
}

}    // namespace TEngine
