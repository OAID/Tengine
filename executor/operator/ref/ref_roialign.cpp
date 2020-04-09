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
 * Author: bzhang@openailab.com
 */

#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/roialign.hpp"

#include "kernel/roialign/ref_roialign_kernel.h"

namespace TEngine {

namespace RefRoialignOps {

struct RoialignOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    // bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    ref_roialign_t kernel_run;
    struct roialign_param op_param;

    KernelRegistry<ref_roialign_t> kernel_registry;

    RoialignOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool RoialignOps::Prerun(Node* node)
{
    //printf("Roialign prerun\n");
    Tensor* input = node->GetInputTensor(0);
    int layout = exec_attr->graph_layout;
    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        // printf("errorno: %d\n",ENOENT);
        return false;
    }
    Roialign* roialign_op = dynamic_cast<Roialign*>(node->GetOp());
    RoialignParam* param = roialign_op->GetParam();    
    op_param.pooled_width = param->pooled_width;
    op_param.pooled_height = param->pooled_height;
    op_param.spatial_scale = param->spatial_scale;
    return true;
}
#if 0
bool RoialignOps::OnBind(Node* node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    return true;
}
#endif
bool RoialignOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* roi_tensor = node->GetInputTensor(1);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const TShape& shape = input_tensor->GetShape();
    const TShape& shape_out = output_tensor->GetShape();

    int elem_num = shape.GetSize();
    void* in_data = get_tensor_mem(input_tensor);
    void* out_data = get_tensor_mem(output_tensor);
    float* roi = ( float* )get_tensor_mem(roi_tensor);
    op_param.channel = shape.GetC();
    op_param.in_height = shape.GetH();
    op_param.in_width = shape.GetW();
    op_param.out_height = shape_out.GetH();
    op_param.out_width = shape_out.GetW();

    float scale = 1.f;
    int zero_point = 0;
    if(input_tensor->GetDataType() == TENGINE_DT_INT8 || input_tensor->GetDataType() == TENGINE_DT_UINT8)
    {
        auto quant_param = input_tensor->GetQuantParam();
        scale = (*quant_param)[0].scale;
        zero_point = (*quant_param)[0].zero_point;
        auto out_quant_param = output_tensor->GetQuantParam();
        out_quant_param->resize(0);
        out_quant_param->push_back((*quant_param)[0]);
    }

    int ret = kernel_run(in_data, out_data, roi, elem_num,  &op_param, scale, zero_point);

    if(ret < 0)
        return false;
    else
        return true;
}

void RoialignOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_roialign_t )ref_roialign_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_roialign_t )ref_roialign_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_roialign_t )ref_roialign_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_roialign_t )ref_roialign_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_roialign_t )ref_roialign_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_roialign_t )ref_roialign_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_roialign_t )ref_roialign_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_roialign_t )ref_roialign_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif

}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RoialignOps* ops = new RoialignOps();

    LOG_DEBUG() << "RoialignOps RefOp is selected\n";

    return ops;
}

}    // namespace RefRoialignOps

void RegisterRefRoialignOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Roialign", RefRoialignOps::SelectFunc, 1000);
}

}    // namespace TEngine
