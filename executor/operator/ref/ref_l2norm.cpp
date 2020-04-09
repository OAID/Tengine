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
#include "operator/l2normalization.hpp"
#include "kernel/l2norm/ref_l2norm_kernel.h"

namespace TEngine {

namespace RefL2NormOps {

struct RefL2Norm : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    struct l2norm_param param;
    ref_l2norm_t kernel_run;
    KernelRegistry<ref_l2norm_t> kernel_registry;

    RefL2Norm(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool RefL2Norm::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    // L2Normalization* fm_op = dynamic_cast<L2Normalization*>(node->GetOp());

    Tensor* input = node->GetInputTensor(0);
    auto i_quant = input->GetQuantParam();

    // int weight_out = weight->GetShape().Shape(0);
    // if(weight_out == param.out_number)
    //     param.need_trans = 0;
    // else
    //     param.need_trans = 1;

    Tensor* output = node->GetOutputTensor(0);
    auto o_quant = output->GetQuantParam();

    if(input->GetDataType() == TENGINE_DT_UINT8 || input->GetDataType() == TENGINE_DT_INT8)
    {
        if(i_quant->size() == 0 || o_quant->size() == 0)
        {
            std::cerr << "FC <UINT8> one quant is NONE: <" << i_quant->size() << ","
                      << o_quant->size() << "\n";
            return false;
        }
        param.scale[0] = (*i_quant)[0].scale;
        param.scale[1] = (*o_quant)[0].scale;
        param.zero_point[0] = (*i_quant)[0].zero_point;
        param.zero_point[1] = (*o_quant)[0].zero_point;
    }

    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefL2Norm::Run(Node* node)
{
    if(kernel_run == nullptr)
        return false;

    Tensor* input = node->GetInputTensor(0);
    void* input_data = get_tensor_mem(input);

    Tensor* output = node->GetOutputTensor(0);
    void* output_data = get_tensor_mem(output);


    const TShape& shape = input->GetShape();
    const std::vector<int> input_dims = shape.GetDim();
    int input_size = 1;
    int channel_size = input_dims[input_dims.size() - 1];
    
    for(unsigned int i = 0; i < input_dims.size(); i++)
    {
        input_size *= input_dims[i];
    }

    if(kernel_run(input_data, output_data, input_size,channel_size, &param) < 0)
        return false;
    return true;
}

void RefL2Norm::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_l2norm_t )ref_l2norm_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_l2norm_t )ref_l2norm_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_l2norm_t )ref_l2norm_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_l2norm_t )ref_l2norm_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif

#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_l2norm_t )ref_l2norm_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_l2norm_t )ref_l2norm_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_l2norm_t )ref_l2norm_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_l2norm_t )ref_l2norm_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefL2Norm* ops = new RefL2Norm();

    LOG_DEBUG() << "Demo RefL2NormOpOp is selected\n";

    return ops;
}

}    // namespace RefL2NormOps

void RegisterRefL2NormOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "L2Normalization", RefL2NormOps::SelectFunc, 1000);
}

}    // namespace TEngine
