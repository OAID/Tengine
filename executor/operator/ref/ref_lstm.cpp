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
#include "operator/lstm.hpp"
#include "kernel/lstm/ref_lstm_kernel.h"

namespace TEngine {

namespace RefLSTMOps {

struct RefLSTM : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    lstm_param param_lstm;
    ref_lstm_t kernel_run;
    KernelRegistry<ref_lstm_t> kernel_registry;

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
    float* init_h_data;
    float* init_c_data;

    RefLSTM(void)
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
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool RefLSTM::Prerun(Node* node)
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
        init_c_data =(float*)get_tensor_mem(init_c_tensor);
    }

    if(init_h_tensor)
    {
        init_h_data =(float*)get_tensor_mem(init_h_tensor);
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

bool RefLSTM::Run(Node* node)
{
    LSTM* lstm_op = dynamic_cast<LSTM*>(node->GetOp());
    LSTMParam* param = lstm_op->GetParam();

    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    // Tensor* kernel_tensor = node->GetInputTensor(1);
    // int element_size = DataType::GetTypeSize(input_tensor->GetDataType());
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


    param_lstm.init_h_data=init_h_data;
    param_lstm.init_c_data=init_c_data;
    param_lstm.bias=bias;
    param_lstm.forget_bias=forget_bias;
    param_lstm.kernel=kernel;
    param_lstm.w_f_data=w_f_data;
    param_lstm.w_i_data=w_i_data;
    param_lstm.w_o_data=w_o_data;
    param_lstm.projection=projection;
    param_lstm.h2h_kernel=h2h_kernel;
    param_lstm.h2h_bias=h2h_bias;
    param_lstm.fused_kernel=fused_kernel;
    param_lstm.seq_lens=seq_lens;
    param_lstm.batch_size=batch_size;
    param_lstm.input_size=input_size;
    param_lstm.output_len=output_len;
    param_lstm.hidden_size=hidden_size;
    param_lstm.cell_size=cell_size;
    param_lstm.mxnet_flag=mxnet_flag;

    if(kernel_run(input, output, &param_lstm) < 0)
        return false;

    return true;
}

void RefLSTM::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_lstm_t )ref_lstm_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_lstm_t )ref_lstm_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefLSTM* ops = new RefLSTM();

    LOG_DEBUG() << "Demo RefGRUOpOp is selected\n";

    return ops;
}

}    // namespace RefLSTMOps

void RegisterRefLSTMOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "LSTM", RefLSTMOps::SelectFunc, 1000);
}

}    // namespace TEngine
