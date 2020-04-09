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

#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"

#include "kernel/maximum/ref_maximum_kernel.h"

namespace TEngine {

namespace RefMaximumOps {

struct MaximumOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    struct maximum_param op_param;
    ref_maximum_t kernel_run;

    KernelRegistry<ref_maximum_t> kernel_registry;

    MaximumOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool MaximumOps::Prerun(Node* node)
{
    Tensor* input0 = node->GetInputTensor(0);
    Tensor* input1 = node->GetInputTensor(1);
    Tensor* output = node->GetOutputTensor(0);
    int layout = exec_attr->graph_layout;

    if(output->GetDataType() != input0->GetDataType() || output->GetDataType() != input1->GetDataType())
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    if(!kernel_registry.GetKernel(kernel_run, layout, input0->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool MaximumOps::OnBind(Node* node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    return true;
}

bool MaximumOps::Run(Node* node)
{
    Tensor* input_tensor_a = node->GetInputTensor(0);
    const TShape& shape_a = input_tensor_a->GetShape();
    int elem_num_a = shape_a.GetSize();
    void* data_a = get_tensor_mem(input_tensor_a);

    Tensor* input_tensor_b = node->GetInputTensor(1);
    const TShape& shape_b = input_tensor_b->GetShape();

    int elem_num_b = shape_b.GetSize();
    void* data_b = get_tensor_mem(input_tensor_b);

    Tensor* output_tensor = node->GetOutputTensor(0);
    void* output_data = get_tensor_mem(output_tensor);

    //printf("elem_num_a =  %d \n", elem_num_a);
    //printf("elem_num_a =  %d \n", elem_num_b);
    if(elem_num_a != elem_num_b)
    {
        LOG_ERROR() << "Tensor size is not equal\n";
        set_tengine_errno(ENOENT);
        return false;
    }

    int ret = kernel_run(data_a, data_b, output_data, elem_num_b, &op_param);
    if(ret < 0)
        return false;
    else
        return true;
}

void MaximumOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_maximum_t )ref_maximum_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_maximum_t )ref_maximum_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_maximum_t )ref_maximum_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_maximum_t )ref_maximum_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_maximum_t )ref_maximum_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_maximum_t )ref_maximum_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_maximum_t )ref_maximum_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_maximum_t )ref_maximum_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    MaximumOps* ops = new MaximumOps();

    LOG_DEBUG() << "Maximum RefOp is selected\n";
    return ops;
}

}    // namespace RefMaximumOps
void RegisterRefMaximumOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Maximum", RefMaximumOps::SelectFunc, 1000);
}

}    // namespace TEngine
