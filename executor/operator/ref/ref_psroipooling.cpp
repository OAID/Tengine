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
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <math.h>
#include <cmath>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"

#include "graph.hpp"
#include "operator/psroipooling.hpp"
#include "kernel/psroipooling/ref_psroipooling_kernel.h"

namespace TEngine {

namespace RefPsroipoolingOps {

struct RefPsroipooling : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);
    psroipooling_ref_param op_param;
    ref_psroipooling_kernel_t kernel_run;
    KernelRegistry<ref_psroipooling_kernel_t> kernel_registry;
    RefPsroipooling(void)
    {
        kernel_run = nullptr;
        InitRegistry();
    }
};
void RefPsroipooling::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_psroipooling_kernel_t )ref_psroipooling_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_psroipooling_kernel_t )ref_psroipooling_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_psroipooling_kernel_t )ref_psroipooling_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_psroipooling_kernel_t )ref_psroipooling_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_psroipooling_kernel_t )ref_psroipooling_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_psroipooling_kernel_t )ref_psroipooling_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_psroipooling_kernel_t )ref_psroipooling_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_psroipooling_kernel_t )ref_psroipooling_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefPsroipooling::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Tensor* input_tensor = node->GetInputTensor(0);
    if(!kernel_registry.GetKernel(kernel_run, layout, input_tensor->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }
    return true;
}
bool RefPsroipooling::Run(Node* node)
{
    Tensor* feat_tensor = node->GetInputTensor(0);
    Tensor* roi_tensor = node->GetInputTensor(1);
    Tensor* out_tensor = node->GetOutputTensor(0);
    TShape& roi_shape = roi_tensor->GetShape();
    void* featmap = ( void* )get_tensor_mem(feat_tensor);
    void* roi = ( void* )get_tensor_mem(roi_tensor);
    void* output = ( void* )get_tensor_mem(out_tensor);

    const std::vector<int>& dims = feat_tensor->GetShape().GetDim();
    Psroipooling* psroipooling_op = dynamic_cast<Psroipooling*>(node->GetOp());
    PsroipoolingParam* param = psroipooling_op->GetParam();
    op_param.spatial_scale = param->spatial_scale;
    op_param.out_h = param->pooled_h;
    op_param.out_w = param->pooled_w;
    op_param.channel = dims[1];
    op_param.in_h = dims[2];
    op_param.in_w = dims[3];
    op_param.num_rois = roi_shape.GetC();
    op_param.output_dim = param->output_dim;

    if(feat_tensor->GetDataType() == TENGINE_DT_INT8 || feat_tensor->GetDataType() == TENGINE_DT_UINT8)
    {
        auto* feat_quant = feat_tensor->GetQuantParam();
        auto* roi_quant = roi_tensor->GetQuantParam();
        if(feat_quant->size() != 1 || roi_quant->size() != 1)
        {
            return false;
        }
        op_param.feat_scale = (*feat_quant)[0].scale;
        op_param.roi_scale = (*roi_quant)[0].scale;
        if(feat_tensor->GetDataType() == TENGINE_DT_UINT8)
        {
            auto* o_quant = out_tensor->GetQuantParam();
            if(o_quant->size() != 1)
            {
                return false;
            }
            op_param.feat_zero = (*feat_quant)[0].zero_point;
            op_param.roi_zero = (*roi_quant)[0].zero_point;
            op_param.out_scale = (*o_quant)[0].scale;
            op_param.out_zero = (*o_quant)[0].zero_point;
        }
    }
    int ret = kernel_run(featmap, roi, output, &op_param);
    if(ret < 0)
        return false;
    if(feat_tensor->GetDataType() == TENGINE_DT_INT8)
    {
        auto* o_quant = out_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = op_param.out_scale;
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if((data_type != TENGINE_DT_FP32 && data_type != TENGINE_DT_FP16) || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    RefPsroipooling* ops = new RefPsroipooling();

    return ops;
}

}    // namespace RefRoiPoolingOps
using namespace RefPsroipoolingOps;

void RegisterRefPsroipoolingOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Psroipooling", RefPsroipoolingOps::SelectFunc, 2000);
}
}    // namespace TEngine
