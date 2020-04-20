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
#include <math.h>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/matmul.hpp"
#include "kernel/matmul/ref_matmul_kernel.h"

namespace TEngine {

namespace RefMatMulOps {

struct RefMatMul : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    struct matmul_data param;
    ref_matmul_kernel_t kernel_run;
    KernelRegistry<ref_matmul_kernel_t> kernel_registry;

    RefMatMul(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool RefMatMul::Prerun(Node* node)
{
    Tensor* input0 = node->GetInputTensor(0);
    int layout = input0->GetShape().GetDataLayout();
    
    if(!kernel_registry.GetKernel(kernel_run, layout, input0->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefMatMul::Run(Node* node)
{
    if(kernel_run == nullptr)
        return false;

    Tensor* input0 = node->GetInputTensor(0);
    Tensor* input1 = node->GetInputTensor(1);
    auto dims = input0->GetShape().GetDim();
    int dim_size = (int)dims.size();
    if(dim_size == 4)
    {
        param.batch = input0->GetShape().Shape(0);
        param.c = input0->GetShape().Shape(1);
        param.h = input0->GetShape().Shape(2);
        param.w = input0->GetShape().Shape(3);
        param.k = input1->GetShape().Shape(3);
    }
    else if(dim_size == 3)
    {
        param.batch = 1; 
        param.c = input0->GetShape().Shape(0);
        param.h = input0->GetShape().Shape(1);
        param.w = input0->GetShape().Shape(2);
        param.k = input1->GetShape().Shape(2);
    }
    else if(dim_size == 2)
    {
        param.batch = 1; 
        param.c = 1 ; //input0->GetShape().Shape(0);
        param.h = input0->GetShape().Shape(0);
        param.w = input0->GetShape().Shape(1);
        param.k = input1->GetShape().Shape(1);
    }
    const void* input_data0 = get_tensor_mem(input0);
    void* input_data1 = get_tensor_mem(input1);

    Tensor* output = node->GetOutputTensor(0);
    void* output_data = get_tensor_mem(output);

    if(kernel_run(input_data0, input_data1, output_data, &param) < 0)
        return false;
    return true;
}

void RefMatMul::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_matmul_kernel_t )ref_matmul_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
#endif

}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefMatMul* ops = new RefMatMul();

    LOG_DEBUG() << "Demo RefMatMul is selected\n";

    return ops;
}

}    // namespace RefFCOps

void RegisterRefMatMulOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "MatMul", RefMatMulOps::SelectFunc, 1000);
}

}    // namespace TEngine
