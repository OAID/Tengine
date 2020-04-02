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
 * Author: ruizhang@openailab.com
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
#include "operator/add_n.hpp"
#include "kernel/add_n/ref_addn_kernel.h"
#include <cmath>

namespace TEngine {

namespace RefAddNImpl {
// const int default_prio = 1500;
struct RefAddNOps : public NodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);
    RefAddNOps()
    {
        kernel_run = nullptr;
        InitRegistry();
    }
    struct ref_addn_param op_param;
    ref_add_n_kernel_t kernel_run;
    uint8_t** in_data_ptrs;
    KernelRegistry<ref_add_n_kernel_t> kernel_registry;
};

void RefAddNOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_add_n_kernel_t )ref_addn_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_add_n_kernel_t )ref_addn_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_add_n_kernel_t )ref_addn_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_add_n_kernel_t )ref_addn_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_add_n_kernel_t )ref_addn_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_add_n_kernel_t )ref_addn_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_add_n_kernel_t )ref_addn_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_add_n_kernel_t )ref_addn_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefAddNOps::Prerun(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
    int layout = exec_attr->graph_layout;
    unsigned int input_num = node->GetInputNum();
    op_param.input_size = input_tensor->GetTotalSize();
    op_param.in_num = input_num;
    op_param.in_scale = new float[input_num];
    op_param.in_zero = new int[input_num];
    in_data_ptrs = new uint8_t*[input_num];

    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }
    return true;
}

bool RefAddNOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    const int data_type = input_tensor->GetDataType();

    for(int i = 0; i < op_param.in_num; ++i)
    {
        Tensor* input_tensor = node->GetInputTensor(i);
        auto* in_quant = input_tensor->GetQuantParam();
        if(in_quant->size())
        {
            op_param.in_scale[i] = (*in_quant)[0].scale;
            op_param.in_zero[i] = (*in_quant)[0].zero_point;
        }

        in_data_ptrs[i] = ( uint8_t* )get_tensor_mem(input_tensor);
    }

    Tensor* output_tensor = node->GetOutputTensor(0);
    uint8_t* out_data = ( uint8_t* )get_tensor_mem(output_tensor);
    memset(out_data, 0, op_param.input_size);
    if(data_type == TENGINE_DT_UINT8)
    {
        auto* o_quant = output_tensor->GetQuantParam();
        op_param.out_scale = (*o_quant)[0].scale;
        op_param.out_zero = (*o_quant)[0].zero_point;
    }
    int ret = kernel_run(in_data_ptrs, out_data, &op_param);
    if(ret < 0)
        return false;

    if(data_type == TENGINE_DT_INT8)
    {
        Tensor* o_tensor = node->GetOutputTensor(0);
        auto* o_quant = o_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = op_param.out_scale;
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }
    return true;
}

bool RefAddNOps::Postrun(Node* node)
{
    free(in_data_ptrs);
    free(op_param.in_scale);
    free(op_param.in_zero);

    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    RefAddNOps* ops = new RefAddNOps();
    return ops;
}

}    // namespace RefAddNImpl

using namespace RefAddNImpl;

void RegisterRefAddNOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Addn", RefAddNImpl::SelectFunc, 1000);
}

}    // namespace TEngine
