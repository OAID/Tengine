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
 * Author: ruizhang@openailab.com
 */

#include <iostream>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/slice.hpp"

#include "kernel/slice/ref_slice_kernel.h"

namespace TEngine {
namespace RefSliceOps {
const int default_prio = 1500;
struct RefSlice : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    RefSlice()
    {
        kernel_run = nullptr;
        InitRegistry();
    }
    struct slice_param op_param;
    ref_slice_t kernel_run;
    int8_t** out_data_ptrs;
    KernelRegistry<ref_slice_t> kernel_registry;
};

void RefSlice::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_slice_t )ref_slice_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_slice_t )ref_slice_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_slice_t )ref_slice_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_slice_t )ref_slice_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_slice_t )ref_slice_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_slice_t )ref_slice_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_slice_t )ref_slice_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_slice_t )ref_slice_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefSlice::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Slice* slice_op = dynamic_cast<Slice*>(node->GetOp());
    SliceParam* param = slice_op->GetParam();
    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
    auto in_dim = input_tensor->GetShape().GetDim();
    unsigned int out_num = node->GetOutputNum();
    out_data_ptrs = new int8_t*[out_num];
    op_param.axis = param->axis;
    op_param.output_shape = new shape_dim[out_num];
    op_param.out_num = out_num;
    op_param.dim_num = ( int )(in_dim.size());
    op_param.iscaffe = param->iscaffe;
    op_param.ismxnet = param->ismxnet;
    op_param.isonnx = param->isonnx;
    op_param.isncnn = param->isncnn;
    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefSlice::Run(Node* node)
{
    Slice* slice_op = dynamic_cast<Slice*>(node->GetOp());
    SliceParam* param = slice_op->GetParam();
    Tensor* input_tensor = node->GetInputTensor(0);
    int8_t* input = ( int8_t* )get_tensor_mem(input_tensor);
    auto in_dim = input_tensor->GetShape().GetDim();
    auto* in_quant = input_tensor->GetQuantParam();
    if(in_quant->size() > 0)
    {
        op_param.out_scale = (*in_quant)[0].scale;
    }
    const int data_type = input_tensor->GetDataType();
    if(op_param.iscaffe||op_param.isncnn)
    {
        // set the input dim and output dim
        for(int i = 0; i < op_param.dim_num; i++)
        {
            op_param.in_shape[i] = in_dim[i];
        }
        // set the output
        for(int i = 0; i < op_param.out_num; ++i)
        {
            Tensor* out_tensor = node->GetOutputTensor(i);
            auto out_dim = out_tensor->GetShape().GetDim();
            for(int j = 0; j < op_param.dim_num; ++j)
            {
                op_param.output_shape[i].dims[j] = out_dim[j];
            }
            out_data_ptrs[i] = ( int8_t* )get_tensor_mem(out_tensor);
            // set the output quant param
            if(data_type == TENGINE_DT_INT8)
            {
                auto* o_quant = out_tensor->GetQuantParam();
                QuantParam q_param;
                q_param.scale = op_param.out_scale;
                o_quant->resize(0);
                o_quant->push_back(q_param);
            }
        }
    }
    else if(op_param.ismxnet || op_param.isonnx)
    {
        op_param.begin = param->begin;
        op_param.end = param->end;
        op_param.axis = param->axis;
        op_param.dim_num = in_dim.size();
        for(unsigned int idx = 0; idx < in_dim.size(); idx++)
        {
            if(in_dim.size() == 4)
            {
                op_param.in_shape[idx] = in_dim[idx];
            }
            else if(in_dim.size() == 3)
            {
                op_param.in_shape_3[idx] = in_dim[idx];
            }
            else if(in_dim.size() == 2)
            {
                op_param.in_shape_2[idx] = in_dim[idx];
            }
        }
        Tensor* o_tensor = node->GetOutputTensor(0);
        out_data_ptrs[0] = ( int8_t* )get_tensor_mem(o_tensor);
        // Set the int8 output quant param
        if(data_type == TENGINE_DT_INT8)
        {
            auto* o_quant = o_tensor->GetQuantParam();
            QuantParam q_param;
            q_param.scale = op_param.out_scale;
            o_quant->resize(0);
            o_quant->push_back(q_param);
        }
    }
    else    // For tensorflow, there is only one output tensor
    {
        int maxdim = 4;
        int real_dim = op_param.dim_num;
        int dim_idx = 0;
        for(int idx = 0; idx < maxdim; idx++)
        {
            if(maxdim - idx > real_dim)
            {
                op_param.output_shape[0].begins[idx] = 0;
                op_param.output_shape[0].sizes[idx] = 1;
                op_param.in_shape[idx] = 1;
            }
            else
            {
                op_param.output_shape[0].begins[idx] = param->begin_[dim_idx];
                op_param.output_shape[0].sizes[idx] = param->size_[dim_idx];
                op_param.in_shape[idx] = in_dim[dim_idx];
                dim_idx++;
            }
        }
        Tensor* o_tensor = node->GetOutputTensor(0);
        out_data_ptrs[0] = ( int8_t* )get_tensor_mem(o_tensor);
        // Set the int8 output quant param
        if(data_type == TENGINE_DT_INT8)
        {
            auto* o_quant = o_tensor->GetQuantParam();
            QuantParam q_param;
            q_param.scale = op_param.out_scale;
            o_quant->resize(0);
            o_quant->push_back(q_param);
        }
    }
    int ret = kernel_run(input, out_data_ptrs, &op_param);
    if(ret < 0)
        return false;
    return true;
}

bool RefSlice::Postrun(Node* node)
{
    delete[] out_data_ptrs;
    delete[] op_param.output_shape;
    return true;
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefSlice* ops = new RefSlice();

    LOG_DEBUG() << "RefSlice is selected\n";

    return ops;
}

}    // end namespace RefSliceOps

void RegisterRefSlice(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Slice", RefSliceOps::SelectFunc,
                                                  RefSliceOps::default_prio);
}
}    // namespace TEngine
