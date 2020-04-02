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

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/swap_axis.hpp"

#include "kernel/swap_axis/ref_swap_axis_kernel.h"

namespace TEngine {

namespace RefSwapAxisOps {

struct RefSwapAxis : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    int dims[5];
    ref_swap_axis_kernel_t kernel_run;
    KernelRegistry<ref_swap_axis_kernel_t> kernel_registry;
    RefSwapAxis(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

void RefSwapAxis::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_swap_axis_kernel_t )ref_swap_axis_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_swap_axis_kernel_t )ref_swap_axis_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_swap_axis_kernel_t )ref_swap_axis_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_swap_axis_kernel_t )ref_swap_axis_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_swap_axis_kernel_t )ref_swap_axis_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_swap_axis_kernel_t )ref_swap_axis_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_swap_axis_kernel_t )ref_swap_axis_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_swap_axis_kernel_t )ref_swap_axis_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefSwapAxis::Prerun(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    int layout = exec_attr->graph_layout;
    if(!kernel_registry.GetKernel(kernel_run, layout, input_tensor->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    std::vector<int>& in_dims = input_tensor->GetShape().GetDim();
    int in_dims_size = in_dims.size();
    SwapAxis* swap = dynamic_cast<SwapAxis*>(node->GetOp());
    SwapAxisParam* param_ = swap->GetParam();
    int dim0 = param_->dim_0;
    int dim1 = param_->dim_1;
    if(dim0 > dim1)
    {
        int tmp = dim0;
        dim0 = dim1;
        dim1 = tmp;
    }

    for(int i = 0; i < 5; i++)
        dims[i] = 1;
    // dim0
    for(int i = 0; i < dim0; i++)
        dims[0] *= in_dims[i];
    // dim1
    dims[1] = in_dims[dim0];
    // dim2
    for(int i = dim0 + 1; i < dim1; i++)
        dims[2] *= in_dims[i];
    // dim3
    dims[3] = in_dims[dim1];
    // dim4
    for(int i = dim1 + 1; i < in_dims_size; i++)
        dims[4] *= in_dims[i];

    return true;
}

bool RefSwapAxis::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);

    void* input_org = get_tensor_mem(input_tensor);
    void* output_org = get_tensor_mem(output_tensor);

    if(kernel_run(input_org, output_org, dims))
        return false;
    if(input_tensor->GetDataType() == TENGINE_DT_INT8)
    {
        auto* i_quant = input_tensor->GetQuantParam();
        auto* o_quant = output_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = (*i_quant)[0].scale;
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }

    return true;
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefSwapAxis* ops = new RefSwapAxis();

    LOG_DEBUG() << "RefSwapAxis is selected\n";

    return ops;
}

}    // namespace RefSwapAxisOps

void RegisterSwapAxisOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "SwapAxis", RefSwapAxisOps::SelectFunc, 1000);
}

}    // namespace TEngine
