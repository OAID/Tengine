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
 * Copyright (c) 2017, Open AI Lab
 * Author: haoluo@openailab.com
 */
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <math.h>
#include <cmath>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"

#include "graph.hpp"
#include "operator/convolution.hpp"
#include "kernel/convolution/ref_conv_kernel.h"

namespace TEngine {

namespace RefConvolutionOps {

const int default_prio = 1500;

struct RefConv : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Reshape(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    bool dynamic_shape;
    op_data op_param;

    ref_conv_kernel_t kernel_run;
    KernelRegistry<ref_conv_kernel_t> kernel_registry;
    RefConv(void)
    {
        kernel_run = nullptr;
        InitRegistry();
    }
};
void RefConv::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_conv_kernel_t )ref_conv_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_conv_kernel_t )ref_conv_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_conv_kernel_t )ref_conv_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_conv_kernel_t )ref_conv_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif

#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_conv_kernel_t )ref_conv_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_conv_kernel_t )ref_conv_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_conv_kernel_t )ref_conv_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_conv_kernel_t )ref_conv_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

bool RefConv::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;

    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();

    Tensor* input_tensor = node->GetInputTensor(0);
    op_param.batch = input_tensor->GetShape().GetN();
    op_param.in_shape[0] = input_tensor->GetShape().GetC();
    op_param.in_shape[1] = input_tensor->GetShape().GetH();
    op_param.in_shape[2] = input_tensor->GetShape().GetW();

    Tensor* kernel_tensor = node->GetInputTensor(1);
    op_param.kernels[0] = kernel_tensor->GetShape().GetH();
    op_param.kernels[1] = kernel_tensor->GetShape().GetW();

    Tensor* output_tensor = node->GetOutputTensor(0);
    op_param.out_shape[0] = output_tensor->GetShape().GetC();
    op_param.out_shape[1] = output_tensor->GetShape().GetH();
    op_param.out_shape[2] = output_tensor->GetShape().GetW();

    op_param.strides[0] = param->stride_h;
    op_param.strides[1] = param->stride_w;

    op_param.dilations[1] = param->dilation_h;
    op_param.dilations[0] = param->dilation_w;

    op_param.pads[0] = param->pad_h0;
    op_param.pads[1] = param->pad_w0;
    op_param.group = param->group;
    op_param.activation = param->activation;
    op_param.layout = layout;
    op_param.k_scale = NULL;
    if(kernel_tensor->GetDataType() == TENGINE_DT_INT8 || kernel_tensor->GetDataType() == TENGINE_DT_UINT8)
    {
        auto* k_quant = kernel_tensor->GetQuantParam();
        int size = k_quant->size();
        if(size < 1)
            return false;
        float* scale = ( float* )malloc(sizeof(float) * size);
        for(int i = 0; i < size; i++)
        {
            scale[i] = (*k_quant)[i].scale;
        }
        op_param.k_scale = scale;
        op_param.zero[1] = (*k_quant)[0].zero_point;
        //get the input quant scale and output quant scale
        auto* i_quant = input_tensor->GetQuantParam();
        auto* o_quant = output_tensor->GetQuantParam();
        if(i_quant->size() != 1 || o_quant->size() != 1)
            return false;
        op_param.scale[0] = (*i_quant)[0].scale;
        op_param.zero[0] = (*i_quant)[0].zero_point;
        op_param.scale[1] = (*o_quant)[0].scale;
        op_param.zero[2] = (*o_quant)[0].zero_point;
    }

    if(!kernel_registry.GetKernel(kernel_run, layout, input_tensor->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefConv::Reshape(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    op_param.batch = input_tensor->GetShape().GetN();
    op_param.in_shape[0] = input_tensor->GetShape().GetC();
    op_param.in_shape[1] = input_tensor->GetShape().GetH();
    op_param.in_shape[2] = input_tensor->GetShape().GetW();

    Tensor* output_tensor = node->GetOutputTensor(0);
    op_param.out_shape[0] = output_tensor->GetShape().GetC();
    op_param.out_shape[1] = output_tensor->GetShape().GetH();
    op_param.out_shape[2] = output_tensor->GetShape().GetW();

    return true;
}
bool RefConv::Run(Node* node)
{
    Tensor* i_tensor = node->GetInputTensor(0);
    const void* input = get_tensor_mem(i_tensor);
    Tensor* k_tensor = node->GetInputTensor(1);
    const void* kernel = get_tensor_mem(k_tensor);
    Tensor* b_tensor = node->GetInputTensor(2);
    const void* bias = nullptr;
    if(b_tensor != nullptr)
        bias = get_tensor_mem(b_tensor);
    Tensor* o_tensor = node->GetOutputTensor(0);
    void* output = get_tensor_mem(o_tensor);

    int ret = kernel_run(input, output, kernel, bias, &op_param);
    if(ret < 0)
        return false;
    
    return true;
}

bool RefConv::Postrun(Node* node)
{
    if(op_param.k_scale)
        free(op_param.k_scale);
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    RefConv* ops = new RefConv();

    return ops;
}

}    // namespace RefConvolutionOps

void RegisterRefConv2d(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Convolution", RefConvolutionOps::SelectFunc,
                                                  RefConvolutionOps::default_prio);
}

}    // namespace TEngine
