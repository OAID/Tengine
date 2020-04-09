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
#include "operator/prelu.hpp"

#include "kernel/prelu/ref_prelu_kernel.h"

namespace TEngine {

namespace RefPreluOps {

struct PReluOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);
    prelu_param op_param;
    ref_prelu_t kernel_run;

    KernelRegistry<ref_prelu_t> kernel_registry;

    PReluOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool PReluOps::Prerun(Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    int layout = exec_attr->graph_layout;

    op_param.layout = layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool PReluOps::OnBind(Node* node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    return true;
}

bool PReluOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const TShape& shape = input_tensor->GetShape();
    const std::vector<int> dims = shape.GetDim();
    if(input_tensor->GetDataType() == TENGINE_DT_INT8 || input_tensor->GetDataType() == TENGINE_DT_UINT8)
    {
        auto quant_param = input_tensor->GetQuantParam();
        if(quant_param->size() != 1)
            return false;
        op_param.scale = (*quant_param)[0].scale;
        op_param.zero = (*quant_param)[0].zero_point;
    }
    int ret = -1;
    int dim0 = dims[0];
    int dim1 = dims[1];
    int dim2 = dims[2];
    int dim3 = dims[3];
    void* data = get_tensor_mem(input_tensor);
    void* out_data = get_tensor_mem(output_tensor);
    const Tensor* slope_tensor = node->GetInputTensor(1);
    float* slope = ( float* )get_tensor_mem(slope_tensor);
    ret = kernel_run(data, out_data, dim0, dim1, dim2, dim3, slope, &op_param);

    if(ret < 0)
        return false;
    else
        return true;
}

void PReluOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_prelu_t )ref_prelu_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_prelu_t )ref_prelu_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_prelu_t )ref_prelu_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_prelu_t )ref_prelu_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_prelu_t )ref_prelu_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_prelu_t )ref_prelu_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_prelu_t )ref_prelu_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_prelu_t )ref_prelu_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    PReluOps* ops = new PReluOps();

    LOG_DEBUG() << "ReluOps RefOp is selected\n";

    return ops;
}

}    // namespace RefPreluOps
void RegisterPreluOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "PReLU", RefPreluOps::SelectFunc, 1000);
}
}    // namespace TEngine