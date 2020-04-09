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
#include "operator/region.hpp"

#include "kernel/region/ref_region_kernel.h"

namespace TEngine {
namespace RefRegionOps {
const int default_prio = 1500;
struct RefRegion : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool RunNHWC(Node* node);
    bool RunNCHW(Node* node);
    void InitRegistry(void);

    RefRegion()
    {
        kernel_run = nullptr;
        InitRegistry();
    }

    ref_region_param op_param;
    ref_region_kernel_t kernel_run;
    KernelRegistry<ref_region_kernel_t> kernel_registry;
};

void RefRegion::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_region_kernel_t )ref_region_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_region_kernel_t )ref_region_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_region_kernel_t )ref_region_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_region_kernel_t )ref_region_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_region_kernel_t )ref_region_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_region_kernel_t )ref_region_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_region_kernel_t )ref_region_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_region_kernel_t )ref_region_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefRegion::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();

    Region* region_op = dynamic_cast<Region*>(node->GetOp());
    RegionParam* param = region_op->GetParam();

    auto dims = input_tensor->GetShape().GetDim();
    for(unsigned int i = 0; i < dims.size(); i++)
    {
        op_param.dims[i] = dims[i];
    }

    op_param.num_box = param->num_box;
    op_param.num_class = param->num_classes;
    op_param.coords = param->coords;

    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefRegion::Run(Node* node)
{
    if(exec_attr->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        return RunNCHW(node);
    }
    else
    {
        LOG_ERROR() << "Region NHWC is not supported\n";
        return false;
    }
}

bool RefRegion::RunNCHW(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);

    void* input_org = get_tensor_mem(input_tensor);
    void* output_org = get_tensor_mem(output_tensor);

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

    // run kernel
    if(kernel_run(input_org, output_org, &op_param) < 0)
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

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefRegion* ops = new RefRegion();

    LOG_DEBUG() << "RefRegion is selected\n";

    return ops;
}

}    // end namespace RefRegionOps

void RegisterRefRegion(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Region", RefRegionOps::SelectFunc,
                                                  RefRegionOps::default_prio);
}
}    // namespace TEngine
