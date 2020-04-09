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

 * Author: bhu@openailab.com
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
#include "operator/argmin.hpp"
#include "kernel/argmin/ref_argmin_kernel.h"
#include <cmath>

namespace TEngine {

namespace RefArgMinImpl {
// const int default_prio = 1500;
struct RefArgMinOps : public NodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);
    RefArgMinOps()
    {
        kernel_run = nullptr;
        InitRegistry();
    }
    struct ref_argmin_param op_param;
    ref_argmin_kernel_t kernel_run;
    //    uint8_t** in_data_ptrs;
    KernelRegistry<ref_argmin_kernel_t> kernel_registry;
};

void RefArgMinOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_argmin_kernel_t )ref_argmin_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_argmin_kernel_t )ref_argmin_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_argmin_kernel_t )ref_argmin_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_argmin_kernel_t )ref_argmin_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_argmin_kernel_t )ref_argmin_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_argmin_kernel_t )ref_argmin_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_argmin_kernel_t )ref_argmin_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_argmin_kernel_t )ref_argmin_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefArgMinOps::Prerun(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
    int layout = exec_attr->graph_layout;

    const TShape& in_shape = input_tensor->GetShape();
    const std::vector<int>& in_dims = in_shape.GetDim();

    ArgMin* argmin_op = dynamic_cast<ArgMin*>(node->GetOp());
    ArgMinParam* param_ = argmin_op->GetParam();
    int axis = param_->axis;
    op_param.axis_size = in_dims.at(axis);
    int outer_size = 1;
    for(int i = 0; i < axis; ++i)
    {
        outer_size *= in_dims.at(i);
    }

    int inner_size = 1;
    const int dims_count = param_->dimension;
    for(int i = axis + 1; i < dims_count; ++i)
    {
        inner_size *= in_dims.at(i);
    }
    op_param.inner_size = inner_size;
    op_param.outer_size = outer_size;

    op_param.axis = param_->axis;

    op_param.dimension = param_->dimension;

    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }
    return true;
}

bool RefArgMinOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    void* input_data = ( void* )get_tensor_mem(input_tensor);
    Tensor* output_tensor = node->GetOutputTensor(0);
    void* out_data = ( void* )get_tensor_mem(output_tensor);
    node->DumpNode();
    int ret = kernel_run(input_data, out_data, &op_param);
    if(ret < 0)
        return false;

    return true;
}

bool RefArgMinOps::Postrun(Node* node)
{
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    RefArgMinOps* ops = new RefArgMinOps();
    return ops;
}

}    // namespace RefArgMinImpl

using namespace RefArgMinImpl;

void RegisterRefArgMinOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "ArgMin", RefArgMinImpl::SelectFunc, 1000);
}

}    // namespace TEngine
