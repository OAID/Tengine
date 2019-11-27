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
#include "operator/batchToSpaceND.hpp"

#include "kernel/batchToSpaceND/ref_batchToSpaceND_kernel.h"

namespace TEngine {
namespace RefBatchToSpaceNDOps {

const int default_prio = 1000;
struct RefBatchToSpaceND : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    RefBatchToSpaceND()
    {
        kernel_run = nullptr;
        InitRegistry();
    }

    struct batchToSpaceND_param op_param;
    ref_batchToSpaceND_t kernel_run;
    KernelRegistry<ref_batchToSpaceND_t> kernel_registry;
};

void RefBatchToSpaceND::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_batchToSpaceND_t )ref_batchToSpaceND_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_batchToSpaceND_t )ref_batchToSpaceND_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_batchToSpaceND_t )ref_batchToSpaceND_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_batchToSpaceND_t )ref_batchToSpaceND_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_batchToSpaceND_t )ref_batchToSpaceND_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_batchToSpaceND_t )ref_batchToSpaceND_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_batchToSpaceND_t )ref_batchToSpaceND_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_batchToSpaceND_t )ref_batchToSpaceND_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefBatchToSpaceND::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Tensor* output_tensor = node->GetOutputTensor(0);
    BatchToSpaceND* batchToSpaceND_op = dynamic_cast<BatchToSpaceND*>(node->GetOp());
    BatchToSpaceNDParam* param = batchToSpaceND_op->GetParam();

    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
    op_param.dilation_x = param->dilation_x;
    op_param.dilation_y = param->dilation_y;
    op_param.crop_left = param->crop_left;
    op_param.crop_top = param->crop_top;
    op_param.crop_bottom = param->crop_bottom;
    op_param.crop_right = param->crop_right;

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


    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefBatchToSpaceND::Run(Node* node)
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
    RefBatchToSpaceND* ops = new RefBatchToSpaceND();

    LOG_DEBUG() << "RefBatchToSpaceND is selected\n";

    return ops;
}

}    // end namespace RefBatchToSpaceNDOps

void RegisterRefBatchToSpaceND(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "BatchToSpaceND", RefBatchToSpaceNDOps::SelectFunc,
                                                  RefBatchToSpaceNDOps::default_prio);
}
}    // namespace TEngine
