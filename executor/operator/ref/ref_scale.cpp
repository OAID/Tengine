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
 * Author: ruizhang@openailab.com
 */

#include <iostream>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/scale.hpp"

#include "kernel/scale/ref_scale_kernel.h"

namespace TEngine {
namespace RefScaleOps {
const int default_prio = 1500;
struct RefScale : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    RefScale()
    {
        kernel_run = nullptr;
        InitRegistry();
    }
    struct scale_param op_param;
    ref_scale_t kernel_run;
    int8_t** out_data_ptrs;
    KernelRegistry<ref_scale_t> kernel_registry;
};

void RefScale::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_scale_t )ref_scale_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_scale_t )ref_scale_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_scale_t )ref_scale_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_scale_t )ref_scale_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
// #ifdef CONFIG_KERNEL_INT8
//     kernel_registry.Register(( ref_scale_t )ref_scale_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
//     kernel_registry.Register(( ref_scale_t )ref_scale_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
// #endif
// #ifdef CONFIG_KERNEL_UINT8
//     kernel_registry.Register(( ref_scale_t )ref_scale_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
//     kernel_registry.Register(( ref_scale_t )ref_scale_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
// #endif
}

bool RefScale::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
    
    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefScale::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* gamma_tensor = node->GetInputTensor(1);
    Tensor* beta_tensor = node->GetInputTensor(2);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const TShape& shape = input_tensor->GetShape();


    void* input = get_tensor_mem(input_tensor);
    void* gamma = get_tensor_mem(gamma_tensor);
    void* output = get_tensor_mem(output_tensor);
    void* beta = nullptr;

    if(beta_tensor != nullptr)
    {
        beta = get_tensor_mem(beta_tensor);
    }
    if(input_tensor->GetDataType() == TENGINE_DT_INT8 || input_tensor->GetDataType() == TENGINE_DT_UINT8)
    {
        auto quant_param = input_tensor->GetQuantParam();
        op_param.scale[0] = (*quant_param)[0].scale;
        op_param.zero_point[0] = (*quant_param)[0].zero_point;
        op_param.scale[1] =op_param.scale[0];
        op_param.zero_point[1] =op_param.zero_point[0];

        auto out_quant_param = output_tensor->GetQuantParam();
        out_quant_param->resize(0);
        out_quant_param->push_back((*quant_param)[0]);
    }

    const std::vector<int> dims = shape.GetDim();
    int batch_number = dims[0];
    int channel_num = dims[1];
    int channel_size = dims[2] * dims[3];

    op_param.batch_number=batch_number;
    op_param.channel_size=channel_size;
    op_param.channel_number=channel_num;

    int ret = kernel_run(input,output,gamma,beta,&op_param);
    if(ret < 0)
        return false;
    return true;
}

bool RefScale::Postrun(Node* node)
{
    return true;
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefScale* ops = new RefScale();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    LOG_DEBUG() << "RefScale is selected\n";

    return ops;
}

}    // end namespace RefScaleOps

void RegisterRefScale(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Scale", RefScaleOps::SelectFunc,
                                                  RefScaleOps::default_prio);
}
}    // namespace TEngine
