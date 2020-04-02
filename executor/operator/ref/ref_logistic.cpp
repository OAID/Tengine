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
#include "operator/logistic.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel/logistic/ref_logistic_kernel.h"

namespace TEngine {

namespace RefLogisticOps {

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
    param->scale[0] = (*i_quant)[0].scale;
    if(itensor->GetDataType() == TENGINE_DT_UINT8)
    {
        if(o_quant->size() != 1)
        {
            std::cerr << "output quant size: " << o_quant->size() << "\n";
            return -1;
        }

        param->scale[1] = (*o_quant)[0].scale;
        param->zero_point[1]= (*o_quant)[0].zero_point;

        param->zero_point[0] = (*i_quant)[0].zero_point;
    }
    return 0;
}
//
struct RefLogistic : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    op_data op_param;

    ref_logistic_kernel_t kernel_run;

    KernelRegistry<ref_logistic_kernel_t> kernel_registry;

    RefLogistic(void)
    {

        kernel_run = nullptr;

        InitRegistry();
    }
};

bool RefLogistic::Prerun(Node* node)
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

bool RefLogistic::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    // int element_size = DataType::GetTypeSize(input_tensor->GetDataType());

    int elements = input_tensor->GetShape().GetSize();


    //
    op_param.out_size = elements;

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
    int ret = kernel_run(input, output, &op_param);
    //
    if(input_tensor->GetDataType() == TENGINE_DT_INT8)
    {
        auto* o_quant = output_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = op_param.scale[1];
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }


    if(ret < 0)
        return false;
    else
        return true;
}

void RefLogistic::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_logistic_kernel_t )ref_logistic_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_logistic_kernel_t )ref_logistic_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_logistic_kernel_t )ref_logistic_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_logistic_kernel_t )ref_logistic_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_logistic_kernel_t )ref_logistic_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_logistic_kernel_t )ref_logistic_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_logistic_kernel_t )ref_logistic_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_logistic_kernel_t )ref_logistic_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefLogistic* ops = new RefLogistic();

    LOG_DEBUG() << "RefLogisticOp is selected\n";

    return ops;
}

}    // namespace RefLogisticOps

void RegisterRefLogisticOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Logistic", RefLogisticOps::SelectFunc, 1000);
}

}    // namespace TEngine
