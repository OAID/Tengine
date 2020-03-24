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
 * Author: haoluo@openailab.com
 */

#include <vector>
#include <math.h>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/l2pool.hpp"
#include "kernel/l2pool/ref_l2pool_kernel.h"

namespace TEngine {

namespace RefL2PoolOps {

struct RefL2Pool : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    struct l2pool_param param;
    ref_l2pool_t kernel_run;
    KernelRegistry<ref_l2pool_t> kernel_registry;

    RefL2Pool(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};
void ConvertPaddingStyleToParameters(float stride_h, float stride_w, 
                                         int in_height, int in_width, int filter_height, int filter_width, PaddingType paddingtype,
                                         int out_height, int out_width,
                                         int* padding_width, int* padding_height)
{
    if(paddingtype == PaddingType::kNone || paddingtype == PaddingType::kValid)
    {
        *padding_width = 0;
        *padding_height = 0;
    }
    else if(paddingtype == PaddingType::kSame)
    {
        *padding_width = (int)(((out_width - 1) * stride_w + filter_width - in_width) / 2);
        *padding_height = (int)(((out_height - 1) * stride_h + filter_height - in_height)/2);
    }

    return;
}
bool RefL2Pool::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;

    Tensor* input = node->GetInputTensor(0);
    auto i_quant = input->GetQuantParam();

    // int weight_out = weight->GetShape().Shape(0);
    // if(weight_out == param.out_number)
    //     param.need_trans = 0;
    // else
    //     param.need_trans = 1;

    Tensor* output = node->GetOutputTensor(0);
    auto o_quant = output->GetQuantParam();

    if(input->GetDataType() == TENGINE_DT_UINT8 || input->GetDataType() == TENGINE_DT_INT8)
    {
        if(i_quant->size() == 0 || o_quant->size() == 0)
        {
            std::cerr << "FC <UINT8> one quant is NONE: <" << i_quant->size() << ","
                      << o_quant->size() << "\n";
            return false;
        }
        param.scale[0] = (*i_quant)[0].scale;
        param.scale[1] = (*o_quant)[0].scale;
        param.zero_point[0] = (*i_quant)[0].zero_point;
        param.zero_point[1] = (*o_quant)[0].zero_point;
    }

    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefL2Pool::Run(Node* node)
{
    if(kernel_run == nullptr)
        return false;

    L2Pool* l2pool_op = dynamic_cast<L2Pool*>(node->GetOp());
    L2PoolParam* param_ = l2pool_op->GetParam();
    
    Tensor* inputTensor = node->GetInputTensor(0);
    const TShape& inputShape = inputTensor->GetShape();
    Tensor* outputTensor = node->GetOutputTensor(0);
    TShape& outputShape = outputTensor->GetShape();
    int input_c = inputShape.GetC();
    int input_h = inputShape.GetH();
    int input_w = inputShape.GetW();
    int input_n = inputShape.GetN();
    int output_h = outputShape.GetH();
    int output_w = outputShape.GetW();  
    int output_c = outputShape.GetC();
    // int input_size = input_c * input_h * input_w;
    // int output_size = output_h * output_w * output_c;
    int padding_w = 0;
    int padding_h = 0;
    
    float* input_data = (float*) get_tensor_mem(inputTensor);
    float* out_data = (float*) get_tensor_mem(outputTensor);
    ConvertPaddingStyleToParameters(param_->stride_h, param_->stride_w, 
                                        input_h, input_w, param_->kernel_h, param_->kernel_w, param_->padding,
                                        output_h, output_w, &padding_w, &padding_h);

    param.inc=input_c;
    param.inh=input_h;
    param.inw=input_w;
    param.inn=input_n;
    param.k_h=param_->kernel_h;
    param.k_w=param_->kernel_w;
    param.outh=output_h;
    param.outw=output_w;
    param.outc= output_c;
    param.pad_h=padding_h;
    param.pad_w=padding_w;
    param.stride_h=param_->stride_h;
    param.stride_w=param_->stride_w;
    



    // if(TENGINE_DT_INT8 == inputTensor->GetDataType())
    // {
    //     auto* out_quant = outputTensor->GetQuantParam();
    //     QuantParam q_param;
    //     q_param.scale = op_param.out_scale;
    //     q_param.zero_point = 0;
    //     out_quant->resize(0);
    //     out_quant->push_back(q_param);
    // }
    if(TENGINE_DT_UINT8 == inputTensor->GetDataType() || TENGINE_DT_INT8 == inputTensor->GetDataType())
    {
        auto* in_quant = inputTensor->GetQuantParam();
        if(in_quant->size())
        {
            param.scale[0]=(*in_quant)[0].scale;
            param.zero_point[0]=(*in_quant)[0].zero_point;
        }
    }
    if(TENGINE_DT_UINT8 == inputTensor->GetDataType())
    {
        auto* out_quant = outputTensor->GetQuantParam();
        if(out_quant->size())
        {
            param.scale[1] = (*out_quant)[0].scale;
            param.zero_point[1] = (*out_quant)[0].zero_point;
        }
    }

    if(kernel_run(input_data, out_data, &param) < 0)
        return false;

    
    if(TENGINE_DT_INT8 == inputTensor->GetDataType())
    {
        auto* out_quant = outputTensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = param.scale[1];
        q_param.zero_point = 0;
        out_quant->resize(0);
        out_quant->push_back(q_param);
    }
    return true;
}

void RefL2Pool::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_l2pool_t )ref_l2pool_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_l2pool_t )ref_l2pool_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_l2pool_t )ref_l2pool_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_l2pool_t )ref_l2pool_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif

#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_l2pool_t )ref_l2pool_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_l2pool_t )ref_l2pool_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_l2pool_t )ref_l2pool_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_l2pool_t )ref_l2pool_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefL2Pool* ops = new RefL2Pool();

    LOG_DEBUG() << "Demo RefL2PoolOpOp is selected\n";

    return ops;
}

}    // namespace RefL2PoolOps

void RegisterRefL2PoolOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "L2Pool", RefL2PoolOps::SelectFunc, 1000);
}

}    // namespace TEngine
