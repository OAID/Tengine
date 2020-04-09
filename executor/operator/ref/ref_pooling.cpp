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

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/pooling.hpp"
#include "kernel/pooling/ref_pooling_kernel.h"

namespace TEngine {

namespace RefPoolingOps {

struct RefPooling : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Reshape(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    struct op_data param;
    ref_pooling_kernel_t kernel_run;
    KernelRegistry<ref_pooling_kernel_t> kernel_registry;

    RefPooling(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool RefPooling::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;
    param.layout = layout;
    Pooling* pooling_op = dynamic_cast<Pooling*>(node->GetOp());
    PoolParam* param_ = pooling_op->GetParam();
    param.kernels[0] = param_->kernel_h;
    param.kernels[1] = param_->kernel_w;
    param.strides[0] = param_->stride_h;
    param.strides[1] = param_->stride_w;
    param.pads[0] = param_->pad_h0;
    param.pads[1] = param_->pad_w0;
    param.method = param_->alg;
    param.caffe_flavor = param_->caffe_flavor;

    Tensor* input = node->GetInputTensor(0);
    param.batch = input->GetShape().GetN();
    param.channel = input->GetShape().GetC();
    param.input[0] = input->GetShape().GetH();
    param.input[1] = input->GetShape().GetW();

    Tensor* output = node->GetOutputTensor(0);
    param.output[0] = output->GetShape().GetH();
    param.output[1] = output->GetShape().GetW();

    if(input->GetDataType() == TENGINE_DT_UINT8)
    {
        auto quant_param = input->GetQuantParam();
        param.zero_point = (*quant_param)[0].zero_point;
    }

    auto i_quant = input->GetQuantParam();
    auto o_quant = output->GetQuantParam();
  #if 1
    if(input->GetDataType() == TENGINE_DT_INT8)
    {
        if(i_quant->size() != 1)
        {
            std::cerr << "Input data_type is INT8 ,and quant param num is not 1 !!!!\n";
            return false;
        }
        o_quant->resize(0);
        o_quant->push_back((*i_quant)[0]);
    }
#endif
    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefPooling::Reshape(Node* node)
{
    Pooling* pooling_op = dynamic_cast<Pooling*>(node->GetOp());
    PoolParam* param_ = pooling_op->GetParam();
    param.kernels[0] = param_->kernel_h;
    param.kernels[1] = param_->kernel_w;

    Tensor* input = node->GetInputTensor(0);
    param.batch = input->GetShape().GetN();
    param.channel = input->GetShape().GetC();
    param.input[0] = input->GetShape().GetH();
    param.input[1] = input->GetShape().GetW();

    Tensor* output = node->GetOutputTensor(0);
    param.output[0] = output->GetShape().GetH();
    param.output[1] = output->GetShape().GetW();
    return true;
}

bool RefPooling::Run(Node* node)
{
    if(kernel_run == nullptr)
        return false;

    Tensor* input = node->GetInputTensor(0);
    Tensor* output = node->GetOutputTensor(0);
    const void* input_data = get_tensor_mem(input);
    void* output_data = get_tensor_mem(output);

    if(kernel_run(input_data, output_data, &param) < 0)
        return false;

    return true;
}

void RefPooling::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_pooling_kernel_t )ref_pooling_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_pooling_kernel_t )ref_pooling_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_pooling_kernel_t )ref_pooling_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_pooling_kernel_t )ref_pooling_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_pooling_kernel_t )ref_pooling_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_pooling_kernel_t )ref_pooling_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_pooling_kernel_t )ref_pooling_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_pooling_kernel_t )ref_pooling_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefPooling* ops = new RefPooling();

    LOG_DEBUG() << "Demo RefPoolingOp is selected\n";

    return ops;
}

}    // namespace RefPoolingOps

void RegisterRefPoolingOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Pooling", RefPoolingOps::SelectFunc, 8000);
}

}    // namespace TEngine
