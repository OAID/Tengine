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
#include "operator/stridedslice.hpp"

#include "kernel/strided_slice/ref_strided_slice_kernel.h"

namespace TEngine {
namespace RefStridedSliceOps {
const int default_prio = 1500;
struct RefStridedSlice : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    RefStridedSlice()
    {
        kernel_run = nullptr;
        InitRegistry();
    }
    struct strided_slice_param op_param;
    ref_strided_slice_t kernel_run;
    KernelRegistry<ref_strided_slice_t> kernel_registry;
};

void RefStridedSlice::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_strided_slice_t )ref_strided_slice_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_strided_slice_t )ref_strided_slice_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_strided_slice_t )ref_strided_slice_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_strided_slice_t )ref_strided_slice_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_strided_slice_t )ref_strided_slice_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_strided_slice_t )ref_strided_slice_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_strided_slice_t )ref_strided_slice_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_strided_slice_t )ref_strided_slice_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefStridedSlice::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
    
    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefStridedSlice::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);

    //    float * input=(float *)get_tensor_mem(input_tensor);
    //    float * output=(float *)get_tensor_mem(output_tensor);

    StridedSlice* slice_op = dynamic_cast<StridedSlice*>(node->GetOp());
    StridedSliceParam* param = slice_op->GetParam();

    const TShape& shape = input_tensor->GetShape();
    const std::vector<int>& in_dim = shape.GetDim();

    const TShape& shape1 = output_tensor->GetShape();
    const std::vector<int> &out_dim_tmp = shape1.GetDim(); 
    std::vector<int> out_dim;
    
    if(in_dim.size() == 2){
        out_dim.push_back(1);
        out_dim.push_back(1);
        out_dim.push_back(out_dim_tmp[0]);
        out_dim.push_back(out_dim_tmp[1]);
    }else if(in_dim.size() == 3){
        out_dim.push_back(out_dim_tmp[0]);
        out_dim.push_back(1);
        out_dim.push_back(out_dim_tmp[1]);
        out_dim.push_back(out_dim_tmp[2]); 
    }else {
        out_dim = out_dim_tmp;
    }
    op_param.in_c=in_dim[3];
    op_param.in_h=in_dim[1];
    op_param.in_w=in_dim[2];
    op_param.batch_num=out_dim[0];
    op_param.out_c=out_dim[3];
    op_param.out_h=out_dim[1];
    op_param.out_w=out_dim[2];
    for (size_t i = 0; i < 4; i++)
    {
        op_param.begin[i]=param->begin[i];
        op_param.stride[i]=param->stride[i];
    }

    void* input = (void* )get_tensor_mem(input_tensor);
    void* output = (void* )get_tensor_mem(output_tensor);
    if(input_tensor->GetDataType() == TENGINE_DT_INT8 || input_tensor->GetDataType() == TENGINE_DT_UINT8)
    {
        auto quant_param = input_tensor->GetQuantParam();
        auto out_quant_param = output_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = (*quant_param)[0].scale;
        out_quant_param->resize(0);
        out_quant_param->push_back(q_param);
    }
    int ret = kernel_run(input,output,&op_param);
    if(ret < 0)
        return false;
    return true;
}

bool RefStridedSlice::Postrun(Node* node)
{
    return true;
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefStridedSlice* ops = new RefStridedSlice();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(exec_attr->graph_layout != TENGINE_LAYOUT_NHWC)
        return nullptr;

    LOG_DEBUG() << "RefStridedSlice is selected\n";

    return ops;
}

}    // end namespace RefStridedSliceOps

void RegisterRefStridedSlice(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "StridedSlice", RefStridedSliceOps::SelectFunc,
                                                  RefStridedSliceOps::default_prio);
}
}    // namespace TEngine
