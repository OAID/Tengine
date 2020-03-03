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

#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/relu.hpp"

#include "kernel/relu/ref_relu_kernel.h"

namespace TEngine {

namespace RefReluOps {

struct ReluOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    // bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    ref_relu_t kernel_run;

    KernelRegistry<ref_relu_t> kernel_registry;

    ReluOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool ReluOps::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    auto i_quant = input_tensor->GetQuantParam();
    auto o_quant = output_tensor->GetQuantParam();

    if(input_tensor->GetDataType() == TENGINE_DT_INT8)
    {
        if(i_quant->size() == 1)
        {
            o_quant->resize(0);
            o_quant->push_back((*i_quant)[0]);
        }
    }
    if(!kernel_registry.GetKernel(kernel_run, layout, input_tensor->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        // printf("errorno: %d\n",ENOENT);
        return false;
    }

    return true;
}
#if 0
bool ReluOps::OnBind(Node* node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    return true;
}
#endif
bool ReluOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const TShape& shape = output_tensor->GetShape();
    int elem_num = shape.GetSize();
    ReLu* relu_op = dynamic_cast<ReLu*>(node->GetOp());
    ReLuParam* param = relu_op->GetParam();
    void* in_data = get_tensor_mem(input_tensor);
    void* out_data = get_tensor_mem(output_tensor);
    float negativeslope = param->negative_slope;

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
    }

    int ret = kernel_run(in_data, out_data, elem_num, negativeslope, scale, zero_point);

    if(ret < 0)
        return false;
    else
        return true;
}

void ReluOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_relu_t )ref_relu_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_relu_t )ref_relu_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_relu_t )ref_relu_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_relu_t )ref_relu_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_relu_t )ref_relu_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_relu_t )ref_relu_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_relu_t )ref_relu_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_relu_t )ref_relu_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    ReluOps* ops = new ReluOps();

    LOG_DEBUG() << "ReluOps RefOp is selected\n";

    return ops;
}

}    // namespace RefReluOps
void RegisterReluOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "ReLu", RefReluOps::SelectFunc, 1000);
}
}    // namespace TEngine
