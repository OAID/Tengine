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

#include <iostream>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/split.hpp"

#include "kernel/split/ref_split_kernel.h"

namespace TEngine {
namespace RefSplitOps {
const int default_prio = 1500;
struct RefSplit : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    RefSplit()
    {
        kernel_run = nullptr;
        InitRegistry();
    }

    struct split_param op_param;
    ref_split_t kernel_run;
    void** output_data;
    KernelRegistry<ref_split_t> kernel_registry;
};

void RefSplit::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_split_t )ref_split_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_split_t )ref_split_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_split_t )ref_split_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_split_t )ref_split_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_split_t )ref_split_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_split_t )ref_split_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_split_t )ref_split_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_split_t )ref_split_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefSplit::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Tensor* output_tensor = node->GetOutputTensor(0);
    Split* split_op = dynamic_cast<Split*>(node->GetOp());
    SplitParam* param = split_op->GetParam();
    // op_param.squeeze_dim = param->squeeze_axis;
    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
    op_param.axis = param->axis;

    int out_nums = node->GetOutputNum();
    output_data = new void*[out_nums];

    op_param.output_shape = new shape_dim[out_nums];
    op_param.output_counts = out_nums;
    op_param.is_caffe = param->is_caffe;
    auto dims = output_tensor->GetShape().GetDim();
    op_param.output_dim = ( int )(dims.size());
    for(int i = 0; i < out_nums; i++)
    {
        for(std::size_t ii = 0; ii < dims.size(); ++ii)
        {
            op_param.output_shape[i].dim[ii] = dims[ii];
        }
    }

    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefSplit::Run(Node* node)
{
    Tensor* i_tensor = node->GetInputTensor(0);
    void* input = get_tensor_mem(i_tensor);
    int data_type = -1;

    data_type = i_tensor->GetDataType();
    auto* in_quant = i_tensor->GetQuantParam();
    if((*in_quant).size() != 0)
    {
        op_param.input_shape.scale = (*in_quant)[0].scale;
        op_param.out_scale = (*in_quant)[0].scale;
        op_param.input_shape.zero = (*in_quant)[0].zero_point;
    }
    else
    {
        op_param.input_shape.scale = 1;
        op_param.input_shape.zero = 0;
    }

    auto dims = i_tensor->GetShape().GetDim();
    op_param.input_dim = ( int )(dims.size());

    for(std::size_t jj = 0; jj < dims.size(); ++jj)
    {
        op_param.input_shape.dim[jj] = dims[jj];
    }

    for(int ii = 0; ii < op_param.output_counts; ++ii)
    {
        Tensor* o_tensor = node->GetOutputTensor(ii);
        auto* o_quant = o_tensor->GetQuantParam();
        if((*o_quant).size() != 0)
        {
            op_param.output_shape[ii].scale = (*o_quant)[0].scale;
            op_param.output_shape[ii].zero = (*o_quant)[0].zero_point;
        }
        else
        {
            op_param.output_shape[ii].scale = 1;
            op_param.output_shape[ii].zero = 0;
        }
        output_data[ii] = get_tensor_mem(o_tensor);
    }

    int ret = kernel_run(input, output_data, &op_param);
    if(ret < 0)
        return false;

    if(data_type == TENGINE_DT_INT8)
    {
        for(int ii = 0; ii < op_param.output_counts; ++ii)
        {
            Tensor* o_tensor = node->GetOutputTensor(ii);
            auto* o_quant = o_tensor->GetQuantParam();
            QuantParam q_param;
            q_param.scale = op_param.out_scale;
            o_quant->resize(0);
            o_quant->push_back(q_param);
        }
    }

    return true;
}

bool RefSplit::Postrun(Node* node)
{
    delete[] output_data;
    delete[] op_param.output_shape;
    return true;
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefSplit* ops = new RefSplit();

    LOG_DEBUG() << "Refconcat is selected\n";

    return ops;
}

}    // end namespace RefSplitOps

void RegisterSplitOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Split", RefSplitOps::SelectFunc,
                                                  RefSplitOps::default_prio);
}
}    // namespace TEngine
