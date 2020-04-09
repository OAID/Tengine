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
#include <iostream>
#include <functional>
#include <stdlib.h>
#include "kernel_registry.hpp"
#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "tengine_errno.hpp"
#include "operator/copy.hpp"
#include <cmath>


#include "kernel/elu/ref_elu_kernel.h"
namespace TEngine {

namespace RefEluImpl {
// const int default_prio = 1500;
struct RefEluOps : public NodeOps
{
    bool Prerun(Node* node) override;
    bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    ref_elu_t kernel_run;
    elu_param op_param;
    KernelRegistry<ref_elu_t> kernel_registry;

    RefEluOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

void RefEluOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_elu_t )ref_elu_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_elu_t )ref_elu_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_elu_t )ref_elu_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_elu_t )ref_elu_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_elu_t )ref_elu_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_elu_t )ref_elu_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_elu_t )ref_elu_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_elu_t )ref_elu_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}
bool RefEluOps::OnBind(Node* node)
{
    // set the inplace feature
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);

    return true;
}

bool RefEluOps::Prerun(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    auto i_quant = input_tensor->GetQuantParam();
    auto o_quant = output_tensor->GetQuantParam();

    if(i_quant->size() == 1)
    {
        o_quant->resize(0);
        o_quant->push_back((*i_quant)[0]);
    }

    int layout = exec_attr->graph_layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, input_tensor->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }
    return true;
}

bool RefEluOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    // int element_size = DataType::GetTypeSize(input_tensor->GetDataType());
    const TShape& shape = input_tensor->GetShape();
    int elem_num = shape.GetSize();
    void* in_data = get_tensor_mem(input_tensor);
    void* out_data = get_tensor_mem(output_tensor);

    float scale = 1.f;
    int zero_point = 0;
    if(input_tensor->GetDataType() == TENGINE_DT_INT8 || input_tensor->GetDataType() == TENGINE_DT_UINT8)
    {
        auto quant_param = input_tensor->GetQuantParam();
        scale = (*quant_param)[0].scale;
        zero_point = (*quant_param)[0].zero_point;
        auto out_quant_param = output_tensor->GetQuantParam();
        out_quant_param->resize(0);
        out_quant_param->push_back((*quant_param)[0]);
        op_param.scale=scale;
        op_param.zero_point=zero_point;
    }
    int ret = kernel_run(in_data,out_data, elem_num ,&op_param);

    if(ret < 0)
        return false;
    else
        return true;
}

bool RefEluOps::Postrun(Node* node)
{

    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    RefEluOps* ops = new RefEluOps();
    return ops;
}

}    // namespace RefEluImpl

using namespace RefEluImpl;

void RegisterRefEluOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Elu", RefEluImpl::SelectFunc, 1000);
}

}    // namespace TEngine
