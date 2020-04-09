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
 * Author: ruizhang@openailab.com
 */

#include <iostream>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/shuffle_channel.hpp"

#include "kernel/shuffle_channel/ref_shuffle_channel_kernel.h"

namespace TEngine {
namespace RefShuffleChannelOps {
const int default_prio = 1500;
struct RefShuffleChannel : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    RefShuffleChannel()
    {
        kernel_run = nullptr;
        InitRegistry();
    }
    struct shuffle_channel_param op_param;
    ref_shuffle_channel_t kernel_run;
    void* out_data_ptrs;
    KernelRegistry<ref_shuffle_channel_t> kernel_registry;
};

void RefShuffleChannel::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_shuffle_channel_t )ref_shuffle_channel_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_shuffle_channel_t )ref_shuffle_channel_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_shuffle_channel_t )ref_shuffle_channel_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_shuffle_channel_t )ref_shuffle_channel_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_shuffle_channel_t )ref_shuffle_channel_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_shuffle_channel_t )ref_shuffle_channel_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_shuffle_channel_t )ref_shuffle_channel_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_shuffle_channel_t )ref_shuffle_channel_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefShuffleChannel::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    ShuffleChannel* shuffle_channel_op = dynamic_cast<ShuffleChannel*>(node->GetOp());
    ShuffleChannelParam* param = shuffle_channel_op->GetParam();
    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
    auto in_dim = input_tensor->GetShape().GetDim();
    op_param.group = param->group;
    
    op_param.n=input_tensor->GetShape().GetN();
    op_param.h=input_tensor->GetShape().GetH();
    op_param.w=input_tensor->GetShape().GetW();
    op_param.c=input_tensor->GetShape().GetC();

    if(input_tensor->GetDataType()==TENGINE_DT_FP32)
    {
        op_param.eletsize=sizeof(float);
    }
    else if(input_tensor->GetDataType()==TENGINE_DT_FP16)
    {
        op_param.eletsize=sizeof(__fp16);
    }
    else if(input_tensor->GetDataType()==TENGINE_DT_INT8)
    {
        op_param.eletsize=sizeof(int8_t);
    }
    else if(input_tensor->GetDataType()==TENGINE_DT_UINT8)
    {
        op_param.eletsize=sizeof(uint8_t);
    }
    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefShuffleChannel::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    int8_t* input = ( int8_t* )get_tensor_mem(input_tensor);
    auto in_dim = input_tensor->GetShape().GetDim();
    auto* in_quant = input_tensor->GetQuantParam();
    float out_scale=0;
    if(in_quant->size() > 0)
    {
        out_scale = (*in_quant)[0].scale;
    }
    const int data_type = input_tensor->GetDataType();
    
    Tensor* o_tensor = node->GetOutputTensor(0);
    out_data_ptrs = get_tensor_mem(o_tensor);
    // Set the int8 output quant param
    if(data_type == TENGINE_DT_INT8)
    {
        auto* o_quant = o_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = out_scale;
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }
    
    int ret = kernel_run(input, out_data_ptrs, &op_param);
    if(ret < 0)
        return false;
    return true;
}

bool RefShuffleChannel::Postrun(Node* node)
{
    return true;
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefShuffleChannel* ops = new RefShuffleChannel();

    LOG_DEBUG() << "RefShuffleChannel is selected\n";

    return ops;
}

}    // end namespace RefShuffleChannelOps

void RegisterRefShuffleChannel(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "ShuffleChannel", RefShuffleChannelOps::SelectFunc,
                                                  RefShuffleChannelOps::default_prio);
}
}    // namespace TEngine
