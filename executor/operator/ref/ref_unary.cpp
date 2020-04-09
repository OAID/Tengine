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
 * Author: bzhang@openailab.com
 */

#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/unary.hpp"

#include "kernel/unary/ref_unary_kernel.h"

namespace TEngine {

namespace RefUnaryOps {

struct UnaryOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    // bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    ref_unary_t kernel_run;

    KernelRegistry<ref_unary_t> kernel_registry;

    UnaryOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool UnaryOps::Prerun(Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    int layout = exec_attr->graph_layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        // printf("errorno: %d\n",ENOENT);
        return false;
    }

    return true;
}
#if 0
bool UnaryOps::OnBind(Node* node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    return true;
}
#endif
bool UnaryOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const TShape& shape = output_tensor->GetShape();
    int elem_num = shape.GetSize();
    Unary* unary_op = dynamic_cast<Unary*>(node->GetOp());
    UnaryParam* param = unary_op->GetParam();
    void* in_data = get_tensor_mem(input_tensor);
    void* out_data = get_tensor_mem(output_tensor);
    int type = param->type;


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

    int ret = kernel_run(in_data, out_data, elem_num,  type, scale, zero_point);

    if(ret < 0)
        return false;
    else
        return true;
}

void UnaryOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_unary_t )ref_unary_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_unary_t )ref_unary_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_unary_t )ref_unary_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_unary_t )ref_unary_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_unary_t )ref_unary_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_unary_t )ref_unary_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_unary_t )ref_unary_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_unary_t )ref_unary_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif

}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    UnaryOps* ops = new UnaryOps();

    LOG_DEBUG() << "UnaryOps RefOp is selected\n";

    return ops;
}

}    // namespace RefUnaryOps

void RegisterRefUnaryOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Unary", RefUnaryOps::SelectFunc, 1000);
}

}    // namespace TEngine
