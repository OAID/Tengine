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
#include "operator/reorg.hpp"

#include "kernel/reorg/ref_reorg_kernel.h"

namespace TEngine {
namespace RefReorgOps {
const int default_prio = 1500;
struct RefReorg : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool RunNHWC(Node* node);
    bool RunNCHW(Node* node);
    void InitRegistry(void);

    RefReorg()
    {
        kernel_run = nullptr;
        InitRegistry();
    }

    ref_reorg_param op_param;
    ref_reorg_kernel_t kernel_run;
    KernelRegistry<ref_reorg_kernel_t> kernel_registry;
};

void RefReorg::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_reorg_kernel_t )ref_reorg_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_reorg_kernel_t )ref_reorg_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_reorg_kernel_t )ref_reorg_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_reorg_kernel_t )ref_reorg_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_reorg_kernel_t )ref_reorg_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_reorg_kernel_t )ref_reorg_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_reorg_kernel_t )ref_reorg_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_reorg_kernel_t )ref_reorg_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefReorg::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();

    Reorg* Reorg_op = dynamic_cast<Reorg*>(node->GetOp());
    ReorgParam* param = Reorg_op->GetParam();

    auto dims = input_tensor->GetShape().GetDim();
    op_param.w = dims[3];
    op_param.h = dims[2];
    op_param.c = dims[1];
    op_param.batch = dims[0];
    op_param.stride = param->stride;

    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefReorg::Run(Node* node)
{
    if(exec_attr->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        return RunNCHW(node);
    }
    else
    {
        LOG_ERROR() << "Reorg NHWC is not supported\n";
        return false;
    }
}

bool RefReorg::RunNCHW(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);

    void* input_org = get_tensor_mem(input_tensor);
    void* output_org = get_tensor_mem(output_tensor);

    // set quant param of output as input
    auto i_quant = input_tensor->GetQuantParam();
    auto o_quant = output_tensor->GetQuantParam();
    int data_type = input_tensor->GetDataType();
    if(data_type == TENGINE_DT_INT8 || data_type == TENGINE_DT_UINT8)
    {
        if(i_quant->size() != 1)
        {
            LOG_ERROR() << "Input quant param num is not 1\n";
            return false;
        }
        o_quant->resize(0);
        o_quant->push_back((*i_quant)[0]);
    }
    // run kernel
    if(kernel_run(input_org, output_org, &op_param) < 0)
        return false;


    return true;
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefReorg* ops = new RefReorg();

    LOG_DEBUG() << "RefReorg is selected\n";

    return ops;
}

}    // end namespace RefReorgOps

void RegisterRefReorg(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Reorg", RefReorgOps::SelectFunc,
                                                  RefReorgOps::default_prio);
}
}    // namespace TEngine
