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

#include <vector>
#include <math.h>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/detection_postprocess.hpp"
#include "kernel/dpp/ref_dpp_kernel.h"

namespace TEngine {

namespace RefDetectionPostOps {

struct RefDetectionPost : public NodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    dpp_param param;
    ref_dpp_kernel_t kernel_run;
    KernelRegistry<ref_dpp_kernel_t> kernel_registry;

    RefDetectionPost(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool RefDetectionPost::Prerun(Node* node)
{
    if(node->GetInputNum() != 3 || node->GetOutputNum() != 4)
        return false;

    int layout = exec_attr->graph_layout;
    DetectionPostProcess* dpp_op = dynamic_cast<DetectionPostProcess*>(node->GetOp());
    DetectionPostProcessParam* param_ = dpp_op->GetParam();
    param.max_classes_per_detection = param_->max_classes_per_detection;
    param.nms_iou_threshold = param_->nms_iou_threshold;
    param.nms_score_threshold = param_->nms_score_threshold;
    param.num_classes = param_->num_classes;
    param.max_detections = param_->max_detections;
    param.scales[0] = param_->scales[0];
    param.scales[1] = param_->scales[1];
    param.scales[2] = param_->scales[2];
    param.scales[3] = param_->scales[3];

    Tensor* input = node->GetInputTensor(0);
    if(input->GetDataType() != TENGINE_DT_FP32 && input->GetDataType() != TENGINE_DT_FP16 &&
       input->GetDataType() != TENGINE_DT_UINT8)
        return false;
    param.num_boxes = input->GetShape().Shape(1);
    auto i_quant = input->GetQuantParam();

    Tensor* score = node->GetInputTensor(1);
    auto s_quant = score->GetQuantParam();

    Tensor* anchor = node->GetInputTensor(2);
    auto a_quant = anchor->GetQuantParam();

    if(input->GetDataType() == TENGINE_DT_UINT8)
    {
        if(i_quant->size() == 0 || s_quant->size() == 0 || a_quant->size() == 0)
        {
            std::cerr << "RefDetectionPost <UINT8> one quant is NONE: <" << i_quant->size() << "," << s_quant->size()
                      << "," << a_quant->size() << "\n";
            return false;
        }
        param.quant_scale[0] = (*i_quant)[0].scale;
        param.quant_scale[1] = (*s_quant)[0].scale;
        param.quant_scale[2] = (*a_quant)[0].scale;
        param.zero[0] = (*i_quant)[0].zero_point;
        param.zero[1] = (*s_quant)[0].zero_point;
        param.zero[2] = (*a_quant)[0].zero_point;
    }

    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefDetectionPost::Run(Node* node)
{
    if(kernel_run == nullptr)
        return false;

    // printf(" ********** run ref dpp\n");

    Tensor* input = node->GetInputTensor(0);
    const void* input_data = get_tensor_mem(input);
    Tensor* score = node->GetInputTensor(1);
    void* score_data = get_tensor_mem(score);
    Tensor* anchor = node->GetInputTensor(2);
    void* anchor_data = get_tensor_mem(anchor);

    Tensor* detect_boxes = node->GetOutputTensor(0);
    float* detect_boxes_data = ( float* )get_tensor_mem(detect_boxes);
    Tensor* detect_classes = node->GetOutputTensor(1);
    float* detect_classes_data = ( float* )get_tensor_mem(detect_classes);
    Tensor* detect_scores = node->GetOutputTensor(2);
    float* detect_scores_data = ( float* )get_tensor_mem(detect_scores);
    Tensor* detect_num = node->GetOutputTensor(3);
    float* detect_num_data = ( float* )get_tensor_mem(detect_num);

    if(kernel_run(input_data, score_data, anchor_data, detect_num_data, detect_classes_data, detect_scores_data,
                  detect_boxes_data, &param) < 0)
        return false;

    return true;
}

void RefDetectionPost::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_dpp_kernel_t )ref_dpp_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_dpp_kernel_t )ref_dpp_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_dpp_kernel_t )ref_dpp_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_dpp_kernel_t )ref_dpp_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_dpp_kernel_t )ref_dpp_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_dpp_kernel_t )ref_dpp_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefDetectionPost* ops = new RefDetectionPost();

    LOG_DEBUG() << "Demo RefDetectionPost is selected\n";

    return ops;
}

}    // namespace RefDetectionPostOps

void RegisterRefDetectionPostOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "DetectionPostProcess",
                                                  RefDetectionPostOps::SelectFunc, 1000);
}

}    // namespace TEngine
