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

#include <vector>
#include <math.h>
#include <algorithm>
#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/detection_output.hpp"
#include "kernel/detectionOutput/ref_detectionOuput_kernel.h"

namespace TEngine {

namespace RefDetectionOutputOps {

struct RefDetectionOutput : public NodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    ddo_param param;
    ref_DetectionOutput_kernel_t kernel_run;
    KernelRegistry<ref_DetectionOutput_kernel_t> kernel_registry;

    RefDetectionOutput(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool RefDetectionOutput::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;

    Tensor* input = node->GetInputTensor(0);
    if(input->GetDataType() != TENGINE_DT_FP32 && input->GetDataType() != TENGINE_DT_FP16 &&
       input->GetDataType() != TENGINE_DT_UINT8 && input->GetDataType() != TENGINE_DT_INT8)
        return false;

    DetectionOutput* ddo_op = dynamic_cast<DetectionOutput*>(node->GetOp());
    DetectionOutputParam* param_ = ddo_op->GetParam();

    param.num_classes = param_->num_classes;
    param.keep_top_k = param_->keep_top_k;
    param.nms_threshold = param_->nms_threshold;
    param.nms_top_k = param_->nms_top_k;
    param.confidence_threshold = param_->confidence_threshold;

    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefDetectionOutput::Run(Node* node)
{
    if(kernel_run == nullptr)
        return false;

    Tensor* loc_tensor = node->GetInputTensor(0);
    Tensor* conf_tensor = node->GetInputTensor(1);
    Tensor* priorbox_tensor = node->GetInputTensor(2);
    Tensor* output_tensor = node->GetOutputTensor(0);

    void* location = ( void* )get_tensor_mem(loc_tensor);
    void* confidence = ( void* )get_tensor_mem(conf_tensor);
    void* priorbox = ( void* )get_tensor_mem(priorbox_tensor);

    const std::vector<int>& PriDims = priorbox_tensor->GetShape().GetDim();

    auto l_quant = loc_tensor->GetQuantParam();
    auto c_quant = conf_tensor->GetQuantParam();
    auto p_quant = priorbox_tensor->GetQuantParam();

    if(output_tensor->GetDataType() == TENGINE_DT_INT8)
    {
        if(l_quant->size() == 0 || c_quant->size() == 0 || p_quant->size() == 0)
        {
            std::cerr << "RefDetectionPost <INT8> one quant is NONE: <" << l_quant->size() << "," << c_quant->size()
                      << "," << p_quant->size() << ">\n";
            return false;
        }
        param.scale[0] = (*l_quant)[0].scale;
        param.scale[1] = (*c_quant)[0].scale;
        param.scale[2] = (*p_quant)[0].scale;
    }

    if(kernel_run(location, confidence, priorbox, PriDims, &param) < 0)
    {
        return false;
    }

    int num_detected = param.bbox_rects.size();
    int total_size = num_detected * 6 * 4;
    void* mem_addr = mem_alloc(total_size);

    set_tensor_mem(output_tensor, mem_addr, total_size, mem_free);
    if(output_tensor->GetDataType() == TENGINE_DT_FP32)
    {
        float* output = ( float* )get_tensor_mem(output_tensor);
        TShape& out_shape = output_tensor->GetShape();
        std::vector<int> outdim = {1, num_detected, 6, 1};
        out_shape.SetDim(outdim);
        for(int i = 0; i < num_detected; i++)
        {
            const Box& r = param.bbox_rects[i];
            float* outptr = output + i * 6;
            outptr[0] = r.class_idx;
            outptr[1] = r.score;
            outptr[2] = r.x0;
            outptr[3] = r.y0;
            outptr[4] = r.x1;
            outptr[5] = r.y1;
        }
    }
    if(output_tensor->GetDataType() == TENGINE_DT_FP16)
    {
        __fp16* output = ( __fp16* )get_tensor_mem(output_tensor);
        TShape& out_shape = output_tensor->GetShape();
        std::vector<int> outdim = {1, num_detected, 6, 1};
        out_shape.SetDim(outdim);
        for(int i = 0; i < num_detected; i++)
        {
            const Box& r = param.bbox_rects[i];
            __fp16* outptr = output + i * 6;
            outptr[0] = fp32_to_fp16(r.class_idx);
            outptr[1] = fp32_to_fp16(r.score);
            outptr[2] = fp32_to_fp16(r.x0);
            outptr[3] = fp32_to_fp16(r.y0);
            outptr[4] = fp32_to_fp16(r.x1);
            outptr[5] = fp32_to_fp16(r.y1);
        }
    }
    if(output_tensor->GetDataType() == TENGINE_DT_INT8)
    {
        int8_t* output = ( int8_t* )get_tensor_mem(output_tensor);
        TShape& out_shape = output_tensor->GetShape();
        std::vector<int> outdim = {1, num_detected, 6, 1};
        out_shape.SetDim(outdim);
        for(int i = 0; i < num_detected; i++)
        {
            const Box& r = param.bbox_rects[i];
            int8_t* outptr = output + i * 6;

            outptr[0] = r.class_idx;
            outptr[1] = round(r.score / param.out_scale);
            outptr[2] = round(r.x0 / param.out_scale);
            outptr[3] = round(r.y0 / param.out_scale);
            outptr[4] = round(r.x1 / param.out_scale);
            outptr[5] = round(r.y1 / param.out_scale);
        }
        auto* o_quant = output_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = param.out_scale;
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }

    return true;
}

void RefDetectionOutput::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_DetectionOutput_kernel_t )ref_DetectionOutput_fp32, TENGINE_LAYOUT_NCHW,
                             TENGINE_DT_FP32);
    kernel_registry.Register(( ref_DetectionOutput_kernel_t )ref_DetectionOutput_fp32, TENGINE_LAYOUT_NHWC,
                             TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_DetectionOutput_kernel_t )ref_DetectionOutput_fp16, TENGINE_LAYOUT_NCHW,
                             TENGINE_DT_FP16);
    kernel_registry.Register(( ref_DetectionOutput_kernel_t )ref_DetectionOutput_fp16, TENGINE_LAYOUT_NHWC,
                             TENGINE_DT_FP16);
#endif

#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_DetectionOutput_kernel_t )ref_DetectionOutput_int8, TENGINE_LAYOUT_NCHW,
                             TENGINE_DT_INT8);
    kernel_registry.Register(( ref_DetectionOutput_kernel_t )ref_DetectionOutput_int8, TENGINE_LAYOUT_NHWC,
                             TENGINE_DT_INT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefDetectionOutput* ops = new RefDetectionOutput();

    LOG_DEBUG() << "Demo RefDetectionPost is selected\n";
    return ops;
}

}    // namespace RefDetectionOutputOps
using namespace RefDetectionOutputOps;
void RegisterRefDetectionOutput(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "DetectionOutput",
                                                  RefDetectionOutputOps::SelectFunc, 1000);
}

}    // namespace TEngine
