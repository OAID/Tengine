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
#include "operator/feature_match.hpp"
#include "kernel/feature_match/ref_feature_match_kernel.h"

namespace TEngine {

namespace RefFeatureMatchOps {

struct RefFeatureMatch : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    struct feature_match_data param;
    ref_feature_match_kernel_t kernel_run;
    KernelRegistry<ref_feature_match_kernel_t> kernel_registry;

    RefFeatureMatch(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool RefFeatureMatch::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    FeatureMatch* fm_op = dynamic_cast<FeatureMatch*>(node->GetOp());
    FMParam* param_ = fm_op->GetParam();
    param.out_number = param_->num_output;

    Tensor* input = node->GetInputTensor(0);
    auto i_quant = input->GetQuantParam();

    Tensor* weight = node->GetInputTensor(1);
    // int weight_out = weight->GetShape().Shape(0);
    // if(weight_out == param.out_number)
    //     param.need_trans = 0;
    // else
    //     param.need_trans = 1;
    auto w_quant = weight->GetQuantParam();

    Tensor* output = node->GetOutputTensor(0);
    auto o_quant = output->GetQuantParam();

    if(input->GetDataType() == TENGINE_DT_UINT8 || input->GetDataType() == TENGINE_DT_INT8)
    {
        if(i_quant->size() == 0 || w_quant->size() == 0 || o_quant->size() == 0)
        {
            std::cerr << "FC <UINT8> one quant is NONE: <" << i_quant->size() << "," << w_quant->size() << ","
                      << o_quant->size() << "\n";
            return false;
        }
        param.scale[0] = (*i_quant)[0].scale;
        param.scale[1] = (*w_quant)[0].scale;
        param.scale[2] = (*o_quant)[0].scale;
        param.zero[0] = (*i_quant)[0].zero_point;
        param.zero[1] = (*w_quant)[0].zero_point;
        param.zero[2] = (*o_quant)[0].zero_point;
    }

    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefFeatureMatch::Run(Node* node)
{
    if(kernel_run == nullptr)
        return false;

    Tensor* input = node->GetInputTensor(0);
    param.batch = input->GetShape().Shape(0);
    param.hidden = input->GetShape().GetSize() / param.batch;
    const void* input_data = get_tensor_mem(input);
    Tensor* weight = node->GetInputTensor(1);

    void* weight_data = get_tensor_mem(weight);

    Tensor* output = node->GetOutputTensor(0);
    void* output_data = get_tensor_mem(output);


    void* bias_data = nullptr;
    if(node->GetInputNum() > 2)
    {
        Tensor* bias = node->GetInputTensor(2);
        bias_data = get_tensor_mem(bias);
    }
    if(kernel_run(input_data, output_data, weight_data, bias_data, &param) < 0)
        return false;
    return true;
}

void RefFeatureMatch::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_feature_match_kernel_t )ref_fm_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_feature_match_kernel_t )ref_fm_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_feature_match_kernel_t )ref_fm_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_feature_match_kernel_t )ref_fm_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif

#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_feature_match_kernel_t )ref_fm_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_feature_match_kernel_t )ref_fm_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_feature_match_kernel_t )ref_fm_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_feature_match_kernel_t )ref_fm_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefFeatureMatch* ops = new RefFeatureMatch();

    LOG_DEBUG() << "Demo RefFeatureMatchOpOp is selected\n";

    return ops;
}

}    // namespace RefFeatureMatchOps

void RegisterRefFeatureMatchOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "FeatureMatch", RefFeatureMatchOps::SelectFunc, 1000);
}

}    // namespace TEngine
