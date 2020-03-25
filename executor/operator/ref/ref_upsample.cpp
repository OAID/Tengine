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
#include "operator/upsample.hpp"

#include "kernel/upsample/ref_upsample_kernel.h"

namespace TEngine {
namespace RefUpsampleOps {
const int default_prio = 1500;
struct RefUpsample : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    RefUpsample()
    {
        kernel_run = nullptr;
        InitRegistry();
    }
    struct upsample_param op_param;
    ref_upsample_t kernel_run;
    KernelRegistry<ref_upsample_t> kernel_registry;
};

void RefUpsample::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_upsample_t )ref_upsample_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_upsample_t )ref_upsample_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_upsample_t )ref_upsample_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_upsample_t )ref_upsample_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_upsample_t )ref_upsample_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_upsample_t )ref_upsample_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_upsample_t )ref_upsample_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_upsample_t )ref_upsample_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefUpsample::Prerun(Node* node)
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

bool RefUpsample::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);

    Upsample* upsample_op = dynamic_cast<Upsample*>(node->GetOp());
    UpsampleParam* param = upsample_op->GetParam();
    std::vector<int> dims = input_tensor->GetShape().GetDim();
    std::vector<int> out_dims = output_tensor->GetShape().GetDim();
    int scale = param->scale;
    int batch = out_dims[0];
    int channel = out_dims[1];
    int out_h = out_dims[2];
    int out_w = out_dims[3];
    int input_h = dims[2];
    int input_w = dims[3];

    void* input = (void* )get_tensor_mem(input_tensor);
    void* output = (void* )get_tensor_mem(output_tensor);

    op_param.scale=scale;
    op_param.batch=batch;
    op_param.channel=channel;
    op_param.out_h=out_h;
    op_param.out_w=out_w;
    op_param.input_h=input_h;
    op_param.input_w=input_w;
    if(input_tensor->GetDataType() == TENGINE_DT_INT8 || input_tensor->GetDataType() == TENGINE_DT_UINT8)
    {
        auto quant_param = input_tensor->GetQuantParam();
        auto out_quant_param = output_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = (*quant_param)[0].scale;
        out_quant_param->resize(0);
        out_quant_param->push_back(q_param);
    }

    int ret = kernel_run(input,output,&op_param);
    if(ret < 0)
        return false;
    return true;
}

bool RefUpsample::Postrun(Node* node)
{
    return true;
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefUpsample* ops = new RefUpsample();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    LOG_DEBUG() << "RefUpsample is selected\n";

    return ops;
}

}    // end namespace RefUpsampleOps

void RegisterRefUpsample(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Upsample", RefUpsampleOps::SelectFunc,
                                                  RefUpsampleOps::default_prio);
}
}    // namespace TEngine
