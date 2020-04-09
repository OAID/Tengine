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
#include "operator/normalize.hpp"
#include "kernel/normalize/ref_normalize_kernel.h"

namespace TEngine {

namespace RefNormalizeOps {

struct RefNormalize : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);
    ref_normalize_param op_param;
    ref_normalize_kernel_t kernel_run;
    KernelRegistry<ref_normalize_kernel_t> kernel_registry;
    RefNormalize(void)
    {
        kernel_run = nullptr;
        InitRegistry();
    }
};

void RefNormalize::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_normalize_kernel_t )ref_normalize_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_normalize_kernel_t )ref_normalize_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_normalize_kernel_t )ref_normalize_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_normalize_kernel_t )ref_normalize_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_normalize_kernel_t )ref_normalize_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_normalize_kernel_t )ref_normalize_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_normalize_kernel_t )ref_normalize_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_normalize_kernel_t )ref_normalize_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefNormalize::Prerun(Node* node)
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

bool RefNormalize::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    // Normalize* normalize_op = dynamic_cast<Normalize*>(node->GetOp());
    // NormalizeParam* param_ = normalize_op->GetParam();

    TShape& shape = input_tensor->GetShape();
    std::vector<int> dims = shape.GetDim();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));

    op_param.layout = exec_attr->graph_layout;
    if(TENGINE_LAYOUT_NCHW == op_param.layout)
    {
        op_param.input_n = dims[0];
        op_param.input_h = dims[2];
        op_param.input_w = dims[3];
        op_param.input_c = dims[1];
    }
    else    // nhwc
    {
        op_param.input_n = dims[0];
        op_param.input_h = dims[1];
        op_param.input_w = dims[2];
        op_param.input_c = dims[3];
    }

    uint8_t* scale = NULL;
    if(node->GetInputNum() > 1)
    {
        const Tensor* scale_tensor = node->GetInputTensor(1);
        scale = ( uint8_t* )get_tensor_mem(scale_tensor);
    }
    uint8_t* input = ( uint8_t* )get_tensor_mem(input_tensor);
    uint8_t* output = ( uint8_t* )get_tensor_mem(output_tensor);
    if(TENGINE_DT_UINT8 == input_tensor->GetDataType() || TENGINE_DT_INT8 == input_tensor->GetDataType())
    {
        auto* in_quant = input_tensor->GetQuantParam();
        if(in_quant->size())
        {
            op_param.in_scale = (*in_quant)[0].scale;
            op_param.in_zero = (*in_quant)[0].zero_point;
        }
        if(node->GetInputNum() == 2)
        {
            Tensor* scale_tensor = node->GetInputTensor(1);
            auto* scale_quant = scale_tensor->GetQuantParam();
            if(scale_quant->size())
            {
                op_param.scale_scale = (*scale_quant)[0].scale;
                op_param.scale_zero = (*scale_quant)[0].zero_point;
            }
        }
    }
    if(TENGINE_DT_UINT8 == input_tensor->GetDataType())
    {
        auto* out_quant = output_tensor->GetQuantParam();
        if(out_quant->size())
        {
            op_param.out_scale = (*out_quant)[0].scale;
            op_param.out_zero = (*out_quant)[0].zero_point;
        }
    }
    int ret = kernel_run(input, output, scale, &(this->op_param));
    if(ret < 0)
        return false;

    if(TENGINE_DT_INT8 == input_tensor->GetDataType())
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
    RefNormalize* ops = new RefNormalize();

    return ops;
}

}    // namespace RefNormalizeOps

void RegisterRefNormlizeOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Normalize", RefNormalizeOps::SelectFunc, 2000);
}
}    // namespace TEngine
