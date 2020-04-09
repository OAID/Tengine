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

#include "operator/resize.hpp"
#include "kernel/resize/ref_resize_kernel.h"

namespace TEngine {

namespace RefResizeOps {

struct ResizeOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    struct resize_param op_param;
    ref_resize_t kernel_run;

    KernelRegistry<ref_resize_t> kernel_registry;

    ResizeOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool ResizeOps::Prerun(Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    int layout = exec_attr->graph_layout;
    Resize* resize_op = dynamic_cast<Resize*>(node->GetOp());
    ResizeParam* param_ = resize_op->GetParam();
    op_param.scale_y = 1.f/param_->scale_w;
    op_param.scale_x = 1.f/param_->scale_h;
    op_param.type = param_->type;
    // printf("param_->type: %d\n",param_->type);

    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool ResizeOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const TShape& shape = input_tensor->GetShape();
    int layout = exec_attr->graph_layout;
    op_param.batch = shape.GetN();
    op_param.channel = shape.GetC();
    op_param.input_h = shape.GetH();
    op_param.input_w = shape.GetW();

    const TShape& shape1 = output_tensor->GetShape();
    op_param.output_h = shape1.GetH();
    op_param.output_w = shape1.GetW();

    float* input = ( float* )get_tensor_mem(input_tensor);
    float* output = ( float* )get_tensor_mem(output_tensor);
    int ret = -1;

    ret = kernel_run(input, output, &op_param, layout);

    if(ret < 0)
        return false;
    else
        return true;
}

bool ResizeOps::Postrun(Node* node)
{
    return true;
}

void ResizeOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_resize_t )ref_resize_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_resize_t )ref_resize_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_resize_t )ref_resize_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
#endif
    // #ifdef CONFIG_KERNEL_INT8
    //     kernel_registry.Register((ref_resize_t)ref_resize_int8,TENGINE_LAYOUT_NCHW,TENGINE_DT_INT8);
    //     kernel_registry.Register((ref_resize_t)ref_resize_int8,TENGINE_LAYOUT_NHWC,TENGINE_DT_INT8);
    // #endif

    // #ifdef CONFIG_KERNEL_UINT8
    //     kernel_registry.Register((ref_resize_t)ref_resize_uint8,TENGINE_LAYOUT_NCHW,TENGINE_DT_UINT8);
    //     kernel_registry.Register((ref_resize_t)ref_resize_uint8,TENGINE_LAYOUT_NHWC,TENGINE_DT_UINT8);
    // #endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    ResizeOps* ops = new ResizeOps();

    LOG_DEBUG() << "ReluOps RefOp is selected\n";

    return ops;
}

}    // namespace RefResizeOps
void RegisterResizeOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Resize", RefResizeOps::SelectFunc, 1000);
}
}    // namespace TEngine
