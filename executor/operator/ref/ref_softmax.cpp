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
 * Author: haitao@openailab.com
 */

#include <vector>
#include <algorithm>

#include "data_type.hpp"
#include "operator/softmax.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel/softmax/ref_softmax_kernel.h"

namespace TEngine {

namespace RefSoftmaxOps {

/* impl ref softmax op */
//
inline static int get_scale_zero(Tensor* itensor, Tensor* otensor, op_data* param)
{
    auto* i_quant = itensor->GetQuantParam();
    auto* o_quant = otensor->GetQuantParam();
    if(i_quant->size() != 1)
    {
        std::cerr << "quant size: input(" << i_quant->size() << ")\n";
        return -1;
    }
    param->i_scale = (*i_quant)[0].scale;
    param->i_zero = (*i_quant)[0].zero_point;

    param->o_scale = (*o_quant)[0].scale;
    param->o_zero = (*o_quant)[0].zero_point;

    return 0;
}
//
struct RefSoftmax : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    float* max_array;
    float* sum_array;

    op_data op_param;

    ref_softmax_kernel_t kernel_run;

    KernelRegistry<ref_softmax_kernel_t> kernel_registry;

    RefSoftmax(void)
    {
        max_array = nullptr;
        sum_array = nullptr;
        kernel_run = nullptr;
        InitRegistry();
    }
};

bool RefSoftmax::Prerun(Node* node)
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

bool RefSoftmax::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const std::vector<int>& dims = input_tensor->GetShape().GetDim();
    Softmax* softmax_op = dynamic_cast<Softmax*>(node->GetOp());
    SoftmaxParam* param_ = softmax_op->GetParam();
    int dim_size = dims.size();
    int axis = param_->axis;
    if(axis > dim_size)
        axis = dim_size - 1;
    int out_size = 1;
    for(int i = 0; i < axis; i++)
    {
        out_size *= dims[i];
    }
    int in_size = 1;
    for(size_t i = axis + 1; i < dims.size(); i++)
    {
        in_size *= dims[i];
    }
    int on_size = dims[axis];

    max_array = ( float* )std::malloc(in_size * sizeof(float));
    sum_array = ( float* )std::malloc(in_size * sizeof(float));

    //
    op_param.out_size = out_size;
    op_param.in_size = in_size;
    op_param.on_size = on_size;

    //
    void* input = ( void* )get_tensor_mem(input_tensor);
    void* output = ( void* )get_tensor_mem(output_tensor);
    //
    /* Get input,kernel,output scale & zero */
    /* Current: one tensor has only one quantparam(scale)*/
    if(input_tensor->GetDataType() == TENGINE_DT_INT8 || input_tensor->GetDataType() == TENGINE_DT_UINT8)
    {
        if(get_scale_zero(input_tensor, output_tensor, &op_param) < 0)
            return false;
    }
    //
    int ret = kernel_run(input, output, max_array, sum_array, &op_param);
    
    std::free(max_array);
    std::free(sum_array);

    if(ret < 0)
        return false;
    else
        return true;
}

void RefSoftmax::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_softmax_kernel_t )ref_softmax_kernel_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_softmax_kernel_t )ref_softmax_kernel_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_softmax_kernel_t )ref_softmax_kernel_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_softmax_kernel_t )ref_softmax_kernel_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_softmax_kernel_t )ref_softmax_kernel_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_softmax_kernel_t )ref_softmax_kernel_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_softmax_kernel_t )ref_softmax_kernel_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_softmax_kernel_t )ref_softmax_kernel_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefSoftmax* ops = new RefSoftmax();

    LOG_DEBUG() << "RefSoftmaxOp is selected\n";

    return ops;
}

}    // namespace RefSoftmaxOps

void RegisterRefSoftmaxOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Softmax", RefSoftmaxOps::SelectFunc, 1000);
}

}    // namespace TEngine
