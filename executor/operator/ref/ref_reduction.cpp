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
 * Author: zpluo@openailab.com
 */

#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"

#include "operator/reduction.hpp"
#include "kernel/reduction/ref_reduce_kernel.h"

namespace TEngine {

namespace RefReductionOps {

struct RefReduction : public MTNodeOps
{
    bool Prerun(Node* node) override;

    bool Run(Node* node) override;
    void InitRegistry(void);

    ref_reduce_t kernel_run;
    reduce_param param;

    KernelRegistry<ref_reduce_t> kernel_registry;
    RefReduction(void)
    {
        InitRegistry();
    }
};

bool RefReduction::Prerun(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);

    int layout = exec_attr->graph_layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, input_tensor->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}
static int get_scale_zero(Tensor* itensor, Tensor* otensor, reduce_param* param)
{
    auto* i_quant = itensor->GetQuantParam();
    auto* o_quant = otensor->GetQuantParam();
    if(i_quant->size() != 1)
        return -1;
    param->scale[0] = (*i_quant)[0].scale;
    if(itensor->GetDataType() == TENGINE_DT_UINT8)
    {
        if(o_quant->size() != 1)
            return -1;

        param->scale[1] = (*o_quant)[0].scale;
        param->zero[1] = (*o_quant)[0].zero_point;

        param->zero[0] = (*i_quant)[0].zero_point;
    }
    return 0;
}

bool RefReduction::Run(Node* node)
{
    Reduction* reduction_op = dynamic_cast<Reduction*>(node->GetOp());
    ReductionParam* op_param = reduction_op->GetParam();

    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* out_tensor = node->GetOutputTensor(0);
    int element_size = DataType::GetTypeSize(out_tensor->GetDataType());
    int out_size = out_tensor->GetTotalSize() / element_size;

    if(input_tensor->GetDataType() == TENGINE_DT_INT8 || input_tensor->GetDataType() == TENGINE_DT_UINT8)
    {
        if(get_scale_zero(input_tensor, out_tensor, &param) < 0)
            return false;
    }
    const TShape& i_shape = input_tensor->GetShape();

    std::vector<int> dims = i_shape.GetDim();

    int dim0 = dims[0];
    int dim1 = dims[1];
    int dim2 = dims[2];
    int dim3 = dims[3];

    param.param_dim[0] = op_param->dim_0;
    param.param_dim[1] = op_param->dim_1;
    param.param_dim[2] = op_param->dim_2;
    param.param_dim[3] = op_param->dim_3;
    param.type = op_param->type;

    void* in_data = get_tensor_mem(input_tensor);
    void* out_data = get_tensor_mem(out_tensor);

    int ret = kernel_run(in_data, out_data, dim0, dim1, dim2, dim3, out_size, &param);
    if(input_tensor->GetDataType() == TENGINE_DT_INT8)
    {
        auto* o_quant = out_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = param.scale[1];
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }
    if(ret < 0)
        return false;
    else
        return true;
}

void RefReduction::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_reduce_t )ref_reduce_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_reduce_t )ref_reduce_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_reduce_t )ref_reduce_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_reduce_t )ref_reduce_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_reduce_t )ref_reduce_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_reduce_t )ref_reduce_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_reduce_t )ref_reduce_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_reduce_t )ref_reduce_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefReduction* ops = new RefReduction();

    LOG_DEBUG() << "Reduction RefOp is selected\n";

    return ops;
}

}    // namespace RefReductionOps
void RegisterReductionOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Reduction", RefReductionOps::SelectFunc, 1000);
}
}    // namespace TEngine
