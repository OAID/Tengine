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
 * Author: jingyou@openailab.com
 */

#include <iostream>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/lrn.hpp"

#include "kernel/lrn/ref_lrn_kernel.h"

namespace TEngine {
namespace RefLrnOps {
const int default_prio = 1500;
struct RefLrn : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool RunNHWC(Node* node);
    bool RunNCHW(Node* node);
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    RefLrn()
    {
        kernel_run = nullptr;
        InitRegistry();
    }

    ref_lrn_param op_param;
    ref_lrn_kernel_t kernel_run;
    KernelRegistry<ref_lrn_kernel_t> kernel_registry;
};

void RefLrn::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_lrn_kernel_t )ref_lrn_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_lrn_kernel_t )ref_lrn_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_lrn_kernel_t )ref_lrn_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_lrn_kernel_t )ref_lrn_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_lrn_kernel_t )ref_lrn_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_lrn_kernel_t )ref_lrn_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_lrn_kernel_t )ref_lrn_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_lrn_kernel_t )ref_lrn_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefLrn::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();

    LRN* lrn_op = dynamic_cast<LRN*>(node->GetOp());
    LRNParam* param = lrn_op->GetParam();

    op_param.layout = layout;
    op_param.alpha = param->alpha;
    op_param.beta = param->beta;
    op_param.bias = param->k;
    op_param.local_size = param->local_size;
    op_param.norm_region = param->norm_region;

    auto dims = input_tensor->GetShape().GetDim();
    for(unsigned int i = 0; i < dims.size(); i++)
    {
        op_param.dims[i] = dims[i];
    }

    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefLrn::Run(Node* node)
{
    if(exec_attr->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        return RunNCHW(node);
    }
    else
    {
        // TODO: support NHWC
        LOG_ERROR() << "LRN: TODO: support NHWC\n";
        return false;
    }
}

bool RefLrn::RunNCHW(Node* node)
{
    if(op_param.norm_region != 0)
    {
        LOG_ERROR() << "LRN: LRN_WITHIN_CHANNEL  TO BE IMPLEMENTED\n";
        return false;
    }
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    void* input = get_tensor_mem(input_tensor);
    void* output = get_tensor_mem(output_tensor);
    int data_type = input_tensor->GetDataType();

    auto dims = input_tensor->GetShape().GetDim();
    for(unsigned int i = 0; i < dims.size(); i++)
    {
        op_param.dims[i] = dims[i];
    }

    auto* in_quant = input_tensor->GetQuantParam();
    if((*in_quant).size() != 0)
    {
        op_param.scale[0] = (*in_quant)[0].scale;
        op_param.zero[0] = (*in_quant)[0].zero_point;
    }
    else
    {
        op_param.scale[0] = 1;
        op_param.zero[0] = 0;
    }

    auto* o_quant = output_tensor->GetQuantParam();
    if((*o_quant).size() != 0)
    {
        op_param.scale[1] = (*o_quant)[0].scale;
        op_param.zero[1] = (*o_quant)[0].zero_point;
    }
    else
    {
        op_param.scale[1] = 1;
        op_param.zero[1] = 0;
    }

    if(kernel_run(input, output, &op_param) < 0)
        return false;

    if(data_type == TENGINE_DT_INT8)
    {
        QuantParam q_param;
        q_param.scale = op_param.scale[1];
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }

    return true;
}

bool RefLrn::Postrun(Node* node)
{
    return true;
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefLrn* ops = new RefLrn();

    LOG_DEBUG() << "RefLrn is selected\n";

    return ops;
}

}    // end namespace RefLrnOps

void RegisterRefLrn(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "LRN", RefLrnOps::SelectFunc,
                                                  RefLrnOps::default_prio);
}
}    // namespace TEngine
