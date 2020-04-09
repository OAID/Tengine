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

#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>
#include <math.h>

#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/rpn.hpp"
#include "kernel/rpn/ref_rpn_kernel.h"

void ref_proposal_local_anchor(int feat_height, int feat_width, int feat_stride, std::vector<Anchor>& anchors,
                               float* local_anchors)
{
    int feat_size = feat_height * feat_width;
    int num_anchors = ( int )anchors.size();
    for(int i = 0; i < num_anchors; ++i)
    {
        for(int j = 0; j < feat_height; j++)
            for(int k = 0; k < feat_width; k++)
            {
                local_anchors[(i * 4 + 0) * feat_size + j * feat_width + k] = anchors[i].x0 + k * feat_stride;
                local_anchors[(i * 4 + 1) * feat_size + j * feat_width + k] = anchors[i].y0 + j * feat_stride;
                local_anchors[(i * 4 + 2) * feat_size + j * feat_width + k] = anchors[i].x1 + k * feat_stride;
                local_anchors[(i * 4 + 3) * feat_size + j * feat_width + k] = anchors[i].y1 + j * feat_stride;
            }
    }
}

namespace TEngine {

namespace RefRPNImpl {

struct RefRPNOps : public NodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    struct rpn_param param;
    ref_rpn_kernel_t kernel_run;
    KernelRegistry<ref_rpn_kernel_t> kernel_registry;

    RefRPNOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};
void RefRPNOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_rpn_kernel_t )ref_rpn_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_rpn_kernel_t )ref_rpn_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
#endif
    /*

    #ifdef CONFIG_KERNEL_INT8
        kernel_registry.Register((ref_pooling_kernel_t)ref_pooling_int8,TENGINE_LAYOUT_NCHW,TENGINE_DT_INT8);
        kernel_registry.Register((ref_pooling_kernel_t)ref_pooling_int8,TENGINE_LAYOUT_NHWC,TENGINE_DT_INT8);
    #endif

    #ifdef CONFIG_KERNEL_UINT8
        kernel_registry.Register((ref_pooling_kernel_t)ref_pooling_uint8,TENGINE_LAYOUT_NCHW,TENGINE_DT_UINT8);
        kernel_registry.Register((ref_pooling_kernel_t)ref_pooling_uint8,TENGINE_LAYOUT_NHWC,TENGINE_DT_UINT8);
    #endif
    */
}

bool RefRPNOps::Prerun(Node* node)
{
    RPN* RPN_op = dynamic_cast<RPN*>(node->GetOp());
    RPNParam* param_ = RPN_op->GetParam();
    param.feat_stride = param_->feat_stride;
    param.min_size = param_->min_size;
    param.per_nms_topn = param_->per_nms_topn;
    param.post_nms_topn = param_->post_nms_topn;
    param.nms_thresh = param_->nms_thresh;

    int layout = exec_attr->graph_layout;
    Tensor* input = node->GetInputTensor(0);

    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefRPNOps::Run(Node* node)
{
    RPN* RPN_op = dynamic_cast<RPN*>(node->GetOp());
    RPNParam* param_ = RPN_op->GetParam();

    const Tensor* score_tensor = node->GetInputTensor(0);
    const Tensor* featmap_tensor = node->GetInputTensor(1);
    const Tensor* info_tensor = node->GetInputTensor(2);
    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& out_shape = output_tensor->GetShape();

    const void* score_org = get_tensor_mem(score_tensor);
    void* featmap_org = get_tensor_mem(featmap_tensor);
    const float* info_org = ( float* )get_tensor_mem(info_tensor);
    void* output_org = get_tensor_mem(output_tensor);

    const TShape& featmap_shape = featmap_tensor->GetShape();
    const int feat_channel = featmap_shape.GetC();
    const int feat_height = featmap_shape.GetH();
    const int feat_width = featmap_shape.GetW();
    const int feat_size = feat_height * feat_width;

    const TShape& score_shape = score_tensor->GetShape();
    param.num_anchors = ( int )param_->anchors_.size();
    param.feat_chan = feat_channel;
    param.feat_height = feat_height;
    param.feat_width = feat_width;
    param.score_chan = score_shape.GetC();
    param.src_height = info_org[0];
    param.src_width = info_org[1];
    param.src_scale = info_org[2];

    // local_anchors (1, anchors_nums_ * 4, map_height_, map_width_);
    int size = param.num_anchors * 4 * feat_size;
    float* local_anchors = new float[size];

    ref_proposal_local_anchor(feat_height, feat_width, param.feat_stride, param_->anchors_, local_anchors);

    int output_num = kernel_run(score_org, featmap_org, local_anchors, output_org, &param);

    std::vector<int> outdim = {1, output_num, 4, 1};
    out_shape.SetDim(outdim);

    delete[] local_anchors;

    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    RefRPNOps* ops = new RefRPNOps();

    return ops;
}

}    // namespace RefRPNImpl

void RegisterRefRPNOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("reference", "RPN", RefRPNImpl::SelectFunc, 1000);
}

}    // namespace TEngine
