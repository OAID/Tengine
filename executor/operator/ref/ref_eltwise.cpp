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
#include "operator/eltwise.hpp"
#include "kernel/eltwise/ref_eltwise_kernel.h"

namespace TEngine {

namespace RefEltwiseOps {

struct EltwiseOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    eltwise_param op_param;
    ref_eltwise_t kernel_run;

    KernelRegistry<ref_eltwise_t> kernel_registry;

    EltwiseOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};
static int get_scale_zero(Tensor* itensor, Tensor* otensor, eltwise_param* param)
{
    auto* i_quant = itensor->GetQuantParam();
    auto* o_quant = otensor->GetQuantParam();
    if(i_quant->size() != 1)
    {
        LOG_ERROR() << "Input quant size: (" << i_quant->size() << ")\n";
        return -1;
    }
    param->scale[0] = (*i_quant)[0].scale;
    if(itensor->GetDataType() == TENGINE_DT_UINT8)
    {
        if(o_quant->size() != 1)
        {
            LOG_ERROR() << "Output quant size: (" << o_quant->size() << ")\n";
            return -1;
        }

        param->scale[2] = (*o_quant)[0].scale;
        param->zero[2] = (*o_quant)[0].zero_point;

        param->zero[0] = (*i_quant)[0].zero_point;
    }

    return 0;
}
static int get_scale_zero_1(Tensor* itensor, eltwise_param* param)
{
    auto* i_quant = itensor->GetQuantParam();
    if(i_quant->size() != 1)
    {
        LOG_ERROR() << "Input quant size: (" << i_quant->size() << ")\n";
        return -1;
    }
    param->scale[1] = (*i_quant)[0].scale;
    if(itensor->GetDataType() == TENGINE_DT_UINT8)
    {
        param->zero[1] = (*i_quant)[0].zero_point;
    }
    return 0;
}

bool EltwiseOps::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    auto* o_quant = output_tensor->GetQuantParam();
    if (input_tensor->GetDataType() == TENGINE_DT_UINT8 || input_tensor->GetDataType() == TENGINE_DT_INT8)
    {
        if (o_quant->size() == 0)
        {
            return false;
        }
        op_param.scale[2] = (*o_quant)[0].scale;
        op_param.zero[2] = (*o_quant)[0].zero_point;
    }
    op_param.layout = layout;
    if(!kernel_registry.GetKernel(kernel_run, layout, input_tensor->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    // int elem_size=DataType::GetTypeSize(input->GetDataType());

    return true;
}

bool EltwiseOps::Run(Node* node)
{
    Eltwise* elt_op = dynamic_cast<Eltwise*>(node->GetOp());
    EltwiseParam* param = elt_op->GetParam();
    Tensor* input_tensor0 = node->GetInputTensor(0);
    const TShape& ishape = input_tensor0->GetShape();
    void* input0 = get_tensor_mem(input_tensor0);
    Tensor* input_tensor1 = nullptr;
    void* input1 = nullptr;
    // this version only support for input_num=2
    // int input_number=node->GetInputNum();

    // output
    Tensor* output_tensor = node->GetOutputTensor(0);
    if(input_tensor0->GetDataType() == TENGINE_DT_INT8 || input_tensor0->GetDataType() == TENGINE_DT_UINT8)
    {
        if(get_scale_zero(input_tensor0, output_tensor, &op_param) < 0)
            return false;
    }
   
    if(node->GetInputNum() > 1)
    {
        input_tensor1 = node->GetInputTensor(1);
        const TShape& ishape1 = input_tensor1->GetShape();
        input1 = get_tensor_mem(input_tensor1);
        
        auto in_dim1 = ishape1.GetDim();
        for(unsigned int i = 0; i < in_dim1.size(); i++)
        {
            op_param.shape1[i] = in_dim1[i];
        }
        for(int i = in_dim1.size(); i < 4;i++)
        {
            op_param.shape1[i] = 1;
        }
        if(input_tensor1->GetDataType() == TENGINE_DT_INT8 || input_tensor1->GetDataType() == TENGINE_DT_UINT8)
        {
            if(get_scale_zero_1(input_tensor1, &op_param) < 0)
                return false;
        }
    }
    void* output = get_tensor_mem(output_tensor);
    // ToDo The broadcast mode(vector + tensor) need to support,now is only support c channel;
    auto in_dim0 = ishape.GetDim();
    for(unsigned int i = 0; i < in_dim0.size(); i++)
    {
        op_param.shape0[i] = in_dim0[i];
    }
    for(int i = in_dim0.size(); i < 4;i++)
    {
        op_param.shape0[i] = 1;
    }

    op_param.type = param->type;
    op_param.shift = param->shift;
    op_param.power = param->power;
    op_param.pScale = param->scale;
    int ret = kernel_run(input0, input1, output, &op_param);

    if(input_tensor0->GetDataType() == TENGINE_DT_INT8)
    {
        auto* o_quant = output_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = op_param.scale[2];
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }

    if(ret < 0)
        return false;
    else
        return true;
}

void EltwiseOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_eltwise_t )ref_eltwise_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_eltwise_t )ref_eltwise_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_eltwise_t )ref_eltwise_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_eltwise_t )ref_eltwise_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_eltwise_t )ref_eltwise_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_eltwise_t )ref_eltwise_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_eltwise_t )ref_eltwise_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_eltwise_t )ref_eltwise_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    EltwiseOps* ops = new EltwiseOps();

    LOG_DEBUG() << "EltwiseOps RefOp is selected\n";

    return ops;
}

}    // namespace RefEltwiseOps
void RegisterEltwiseOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Eltwise", RefEltwiseOps::SelectFunc, 1000);
}

}    // namespace TEngine
