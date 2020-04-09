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
 * Author: bingzhang@openailab.com
 */

#include <iostream>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/spaceToBatchND.hpp"

#include "kernel/spaceToBatchND/ref_SpaceToBatchND_kernel.h"

namespace TEngine {
namespace RefSpaceToBatchNDOps {

const int default_prio = 1000;
struct RefSpaceToBatchND : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    RefSpaceToBatchND()
    {
        kernel_run = nullptr;
        InitRegistry();
    }

    struct spaceToBatchND_param op_param;
    ref_spaceToBatchND_t kernel_run;
    KernelRegistry<ref_spaceToBatchND_t> kernel_registry;
};

void RefSpaceToBatchND::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_spaceToBatchND_t )ref_spaceToBatchND_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_spaceToBatchND_t )ref_spaceToBatchND_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_spaceToBatchND_t )ref_spaceToBatchND_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_spaceToBatchND_t )ref_spaceToBatchND_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_spaceToBatchND_t )ref_spaceToBatchND_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_spaceToBatchND_t )ref_spaceToBatchND_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_spaceToBatchND_t )ref_spaceToBatchND_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_spaceToBatchND_t )ref_spaceToBatchND_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefSpaceToBatchND::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Tensor* output_tensor = node->GetOutputTensor(0);
    SpaceToBatchND* spaceToBatchND_op = dynamic_cast<SpaceToBatchND*>(node->GetOp());
    SpaceToBatchNDParam* param = spaceToBatchND_op->GetParam();

    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
    op_param.dilation_x = param->dilation_x;
    op_param.dilation_y = param->dilation_y;
    op_param.pad_left = param->pad_left;
    op_param.pad_top = param->pad_top;
    op_param.pad_bottom = param->pad_bottom;
    op_param.pad_right = param->pad_right;

    const TShape& out_shape = output_tensor->GetShape();
    op_param.out_dims[0] = out_shape.GetN();
    op_param.out_dims[1] = out_shape.GetH();
    op_param.out_dims[2] = out_shape.GetW();
    op_param.out_dims[3] = out_shape.GetC();

    const TShape& in_shape = input_tensor->GetShape();
    op_param.in_dims[0] = in_shape.GetN();
    op_param.in_dims[1] = in_shape.GetH();
    op_param.in_dims[2] = in_shape.GetW();
    op_param.in_dims[3] = in_shape.GetC();

    if(in_shape.GetDataLayout() == TENGINE_LAYOUT_NHWC){
        op_param.type = 1;
    } else {
        op_param.type = 0;
    }

    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefSpaceToBatchND::Run(Node* node)
{
    Tensor* output_tensor = node->GetOutputTensor(0);
    void* output = get_tensor_mem(output_tensor);
    int data_type = -1;

    Tensor* input_tensor = node->GetInputTensor(0);
    data_type = input_tensor->GetDataType();
    auto* in_quant = input_tensor->GetQuantParam();
    if((*in_quant).size() != 0)
    {
        op_param.in_scale = (*in_quant)[0].scale;
        op_param.in_zero = (*in_quant)[0].zero_point;
    }
    else
    {
        op_param.in_scale = 1;
        op_param.in_zero = 0;
    }

    const void* input_data = get_tensor_mem(input_tensor);

    auto* o_quant = output_tensor->GetQuantParam();
    if((*o_quant).size() != 0)
    {
        op_param.out_scale = (*o_quant)[0].scale;
        op_param.out_zero = (*o_quant)[0].zero_point;
    }
    else
    {
        op_param.out_scale = 1;
        op_param.out_zero = 0;
    }
    int ret = kernel_run(input_data, output, &op_param);
    if(ret < 0)
        return false;

    if(data_type == TENGINE_DT_INT8)
    {
        auto* o_quant = output_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = op_param.out_scale;
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }

    return true;
}


NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefSpaceToBatchND* ops = new RefSpaceToBatchND();

    LOG_DEBUG() << "RefSpaceToBatchND is selected\n";

    return ops;
}

}    // end namespace RefSpaceToBatchNDOps

void RegisterRefSpaceToBatchND(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "SpaceToBatchND", RefSpaceToBatchNDOps::SelectFunc,
                                                  RefSpaceToBatchNDOps::default_prio);
}
}    // namespace TEngine
