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
 * Author: jjzeng@openailab.com
 */

#include <iostream>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/concat.hpp"

#include "kernel/concat/ref_concat_kernel.h"

namespace TEngine {
namespace RefConcatOps {

const int default_prio = 1500;
struct RefConcat : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    RefConcat()
    {
        kernel_run = nullptr;
        InitRegistry();
    }

    struct concat_param op_param;
    ref_concat_t kernel_run;
    void** input_data;
    KernelRegistry<ref_concat_t> kernel_registry;
};

void RefConcat::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_concat_t )ref_concat_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_concat_t )ref_concat_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_concat_t )ref_concat_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_concat_t )ref_concat_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_concat_t )ref_concat_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_concat_t )ref_concat_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_concat_t )ref_concat_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_concat_t )ref_concat_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefConcat::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Tensor* output_tensor = node->GetOutputTensor(0);
    Concat* concat_op = dynamic_cast<Concat*>(node->GetOp());
    ConcatParam* param = concat_op->GetParam();

    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
    op_param.axis = param->axis;

    int in_nums = node->GetInputNum();
    input_data = new void*[in_nums];
    op_param.input_shape = new shape_dim[in_nums];
    op_param.input_counts = in_nums;

    auto dims = output_tensor->GetShape().GetDim();
    op_param.output_dim = ( int )(dims.size());
    for(std::size_t ii = 0; ii < dims.size(); ++ii)
    {
        op_param.output_shape.dim[ii] = dims[ii];
    }

    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefConcat::Run(Node* node)
{
    Tensor* output_tensor = node->GetOutputTensor(0);
    void* output = get_tensor_mem(output_tensor);
    int data_type = -1;
    for(int ii = 0; ii < op_param.input_counts; ++ii)
    {
        Tensor* input_tensor = node->GetInputTensor(ii);
        data_type = input_tensor->GetDataType();
        auto* in_quant = input_tensor->GetQuantParam();
        if((*in_quant).size() != 0)
        {
            op_param.input_shape[ii].scale = (*in_quant)[0].scale;
            op_param.input_shape[ii].zero = (*in_quant)[0].zero_point;
        }
        else
        {
            op_param.input_shape[ii].scale = 1;
            op_param.input_shape[ii].zero = 0;
        }

        auto dims = input_tensor->GetShape().GetDim();
        op_param.input_dim = ( int )(dims.size());
        for(std::size_t jj = 0; jj < dims.size(); ++jj)
        {
            op_param.input_shape[ii].dim[jj] = dims[jj];
        }

        input_data[ii] = get_tensor_mem(input_tensor);
    }

    auto* o_quant = output_tensor->GetQuantParam();
    if((*o_quant).size() != 0)
    {
        op_param.output_shape.scale = (*o_quant)[0].scale;
        op_param.output_shape.zero = (*o_quant)[0].zero_point;
    }
    else
    {
        op_param.output_shape.scale = 1;
        op_param.output_shape.zero = 0;
    }

    const void** input = ( const void** )input_data;
    int ret = kernel_run(input, output, &op_param);
    if(ret < 0)
        return false;

    if(data_type == TENGINE_DT_INT8)
    {
        auto* o_quant = output_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = op_param.out_scale;
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }

    return true;
}

bool RefConcat::Postrun(Node* node)
{
    delete[] input_data;
    delete[] op_param.input_shape;
    return true;
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefConcat* ops = new RefConcat();

    LOG_DEBUG() << "Refconcat is selected\n";

    return ops;
}

}    // end namespace RefConcatOps

void RegisterRefConcat(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Concat", RefConcatOps::SelectFunc,
                                                  RefConcatOps::default_prio);
}
}    // namespace TEngine
