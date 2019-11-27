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

#include "kernel/tanh/ref_tanh_kernel.h"

namespace TEngine {

namespace RefTanhOps {

struct TanhOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    struct tanh_param op_param;
    ref_tanh_t kernel_run;

    KernelRegistry<ref_tanh_t> kernel_registry;

    TanhOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool TanhOps::Prerun(Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    Tensor* output = node->GetOutputTensor(0);
    int layout = exec_attr->graph_layout;

    if(output->GetDataType() == TENGINE_DT_UINT8)
    {
        auto output_quant = output->GetQuantParam();
        if(output_quant->size() < 1)
            return false;
        op_param.output_scale = (*output_quant)[0].scale;
        op_param.output_zero = (*output_quant)[0].zero_point;
    }

    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool TanhOps::OnBind(Node* node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    return true;
}

bool TanhOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    const TShape& shape = input_tensor->GetShape();
    int elem_num = shape.GetSize();
    void* data = get_tensor_mem(input_tensor);

    if(input_tensor->GetDataType() == TENGINE_DT_INT8 || input_tensor->GetDataType() == TENGINE_DT_UINT8)
    {
        auto input_quant = input_tensor->GetQuantParam();
        if(input_quant->size() < 1)
            return false;
        op_param.input_scale = (*input_quant)[0].scale;
        op_param.input_zero = (*input_quant)[0].zero_point;
    }

    int ret = kernel_run(data, elem_num, &op_param);

    if(ret < 0)
        return false;
    else
        return true;
}

void TanhOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_tanh_t )ref_tanh_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_tanh_t )ref_tanh_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_tanh_t )ref_tanh_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_tanh_t )ref_tanh_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_tanh_t )ref_tanh_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_tanh_t )ref_tanh_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_tanh_t )ref_tanh_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_tanh_t )ref_tanh_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    TanhOps* ops = new TanhOps();

    LOG_DEBUG() << "ReluOps RefOp is selected\n";

    return ops;
}

}    // namespace RefTanhOps
void RegisterTanhOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Tanh", RefTanhOps::SelectFunc, 1000);
}

}    // namespace TEngine
