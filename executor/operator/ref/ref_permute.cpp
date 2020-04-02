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
 * Author: jjzeng@openailab.com
 */

#include <iostream>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/permute.hpp"

#include "kernel/permute/ref_permute_kernel.h"

namespace TEngine {
namespace RefPermuteOps {

const int default_prio = 1500;
struct RefPermute : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    RefPermute()
    {
        kernel_run = nullptr;
        InitRegistry();
    }

    permute_param op_param;
    ref_permute_t kernel_run;
    KernelRegistry<ref_permute_t> kernel_registry;
};

void RefPermute::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_permute_t )ref_permute_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_permute_t )ref_permute_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_permute_t )ref_permute_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_permute_t )ref_permute_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_permute_t )ref_permute_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_permute_t )ref_permute_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_permute_t )ref_permute_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_permute_t )ref_permute_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefPermute::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;

    Permute* permute_op = dynamic_cast<Permute*>(node->GetOp());
    PermuteParam* param = permute_op->GetParam();

    op_param.order0 = param->order0;
    op_param.order1 = param->order1;
    op_param.order2 = param->order2;
    op_param.order3 = param->order3;

    Tensor* in_tensor = node->GetInputTensor(0);
    auto dims = in_tensor->GetShape().GetDim();
    for(std::size_t ii = 0; ii < dims.size(); ++ii)
    {
        op_param.in_dim[ii] = dims[ii];
    }
    op_param.layout = layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, in_tensor->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefPermute::Run(Node* node)
{
    Tensor* o_tensor = node->GetOutputTensor(0);
    void* output = get_tensor_mem(o_tensor);
    Tensor* i_tensor = node->GetInputTensor(0);
    const void* input = get_tensor_mem(i_tensor);
    float scale = 1;
    int data_type = i_tensor->GetDataType();
    auto* i_quant = i_tensor->GetQuantParam();
    if((*i_quant).size() != 0)
    {
        scale = (*i_quant)[0].scale;
    }

    int ret = kernel_run(input, output, &op_param);
    if(ret < 0)
        return false;

    if(data_type == TENGINE_DT_INT8)
    {
        auto* o_quant = o_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = scale;
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }

    return true;
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefPermute* ops = new RefPermute();

    LOG_DEBUG() << "Refpermute is selected\n";

    return ops;
}

}    // namespace RefPermuteOps

void RegisterRefPermute(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Permute", RefPermuteOps::SelectFunc,
                                                  RefPermuteOps::default_prio);
}
}    // namespace TEngine
