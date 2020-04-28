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
 * Copyright (c) 2017, Open AI Lab
 * Author: ruizhang@openailab.com
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
#include "operator/priorbox.hpp"
#include "kernel/priorbox/ref_priorbox_kernel.h"

namespace TEngine {

namespace RefPriorBoxOps {

struct RefPriorbox : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);
    priorbox_ref_param op_param;
    ref_priorbox_kernel_t kernel_run;
    KernelRegistry<ref_priorbox_kernel_t> kernel_registry;
    RefPriorbox(void)
    {
        kernel_run = nullptr;
        InitRegistry();
    }
};
void RefPriorbox::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_priorbox_kernel_t )ref_priorbox_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_priorbox_kernel_t )ref_priorbox_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_priorbox_kernel_t )ref_priorbox_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_priorbox_kernel_t )ref_priorbox_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_priorbox_kernel_t )ref_priorbox_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_priorbox_kernel_t )ref_priorbox_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_priorbox_kernel_t )ref_priorbox_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_priorbox_kernel_t )ref_priorbox_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefPriorbox::Prerun(Node* node)
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

bool RefPriorbox::Run(Node* node)
{
    const Tensor* featmap_tensor = node->GetInputTensor(0);
    const Tensor* data_tensor = node->GetInputTensor(1);
    Tensor* output_tensor = node->GetOutputTensor(0);
    PriorBox* priorbox_op = dynamic_cast<PriorBox*>(node->GetOp());
    PriorBoxParam* param_ = priorbox_op->GetParam();
    const TShape& data_shape = data_tensor->GetShape();
    const int data_height = data_shape.GetH();
    const int data_width = data_shape.GetW();
    const TShape& featmap_shape = featmap_tensor->GetShape();
    const int feat_height = featmap_shape.GetH();
    const int feat_width = featmap_shape.GetW();

    if(param_->img_h == 0 || param_->img_w == 0)
    {
        op_param.image_h = data_height;
        op_param.image_w = data_width;
    }
    else
    {
        op_param.image_h = param_->img_h;
        op_param.image_w = param_->img_w;
    }

    if((int)param_->step_h == 0 || (int)param_->step_w == 0)
    {
        op_param.step_w = ( float )(op_param.image_w) / feat_width;
        op_param.step_h = ( float )(op_param.image_h) / feat_height;    
    }
    else
    {
        op_param.step_w = param_->step_w;
        op_param.step_h = param_->step_h;       
    }
    op_param.offset = param_->offset;
    op_param.num_priors = param_->num_priors_;
    op_param.out_dim = param_->out_dim_;
    op_param.max_size_num = ( int )param_->max_size.size();
    if(!param_->max_size.empty())
    {
        op_param.max_size = &(param_->max_size[0]);
    }
    op_param.min_size_num = ( int )param_->min_size.size();
    op_param.aspect_ratio_size = ( int )param_->aspect_ratio.size();
    if(!param_->aspect_ratio.empty())
    {
        op_param.aspect_ratio = &(param_->aspect_ratio[0]);      
    }
    if(!param_->min_size.empty())
    {
        op_param.min_size = &(param_->min_size[0]);   
    }
    if(!param_->variance.empty())
    {
        op_param.variance = &(param_->variance[0]);
    }

    op_param.clip = param_->clip;
    op_param.flip = param_->flip;
    op_param.image_size = param_->img_size;
    op_param.feature_h = feat_height;
    op_param.feature_w = feat_width;

    if(TENGINE_DT_UINT8 == output_tensor->GetDataType())
    {
        auto* out_quant = output_tensor->GetQuantParam();
        if(!out_quant->size())
        {
            op_param.out_scale = (*out_quant)[0].scale;
            op_param.out_zero = (*out_quant)[0].zero_point;
        }
    }
    void* output = ( void* )get_tensor_mem(output_tensor);
    // int elem_size=DataType::GetTypeSize(output_tensor->GetDataType());
    TShape& outShape = output_tensor->GetShape();
    std::vector<int> outDims = outShape.GetDim();
    int elem_size = outDims[0] * outDims[1] * outDims[2] * outDims[3];

    int ret = kernel_run(output, &(this->op_param), elem_size);
    if(ret < 0)
        return false;

    if(TENGINE_DT_INT8 == output_tensor->GetDataType())
    {
        auto* out_quant = output_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = op_param.out_scale;
        q_param.zero_point = 0;
        out_quant->resize(0);
        out_quant->push_back(q_param);
    }

    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    RefPriorbox* ops = new RefPriorbox();
    return ops;
}

}    // namespace RefPriorBoxOps
using namespace RefPriorBoxOps;

void RegisterRefPriorBox(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "PriorBox", RefPriorBoxOps::SelectFunc, 2000);
}
}    // namespace TEngine
