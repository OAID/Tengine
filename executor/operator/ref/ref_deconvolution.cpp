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
#include "operator/deconvolution.hpp"
#include "kernel/deconvolution/ref_deconv_kernel.h"

namespace TEngine {

namespace RefDeconvolutionOps {

const int default_prio = 1500;

struct RefDeconv : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Reshape(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    int element_size;
    bool dynamic_shape;
    deconv_ref_param op_param;

    ref_deconv_kernel_t kernel_run;
    KernelRegistry<ref_deconv_kernel_t> kernel_registry;
    RefDeconv(void)
    {
        kernel_run = nullptr;
        InitRegistry();
    }
};
void RefDeconv::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_deconv_kernel_t )ref_deconv_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_deconv_kernel_t )ref_deconv_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_deconv_kernel_t )ref_deconv_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_deconv_kernel_t )ref_deconv_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_deconv_kernel_t )ref_deconv_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_deconv_kernel_t )ref_deconv_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_deconv_kernel_t )ref_deconv_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_deconv_kernel_t )ref_deconv_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}
bool RefDeconv::Reshape(Node* node)
{
    return true;
}

bool RefDeconv::Prerun(Node* node)
{
    int layout = exec_attr->graph_layout;

    Deconvolution* deconv_op = dynamic_cast<Deconvolution*>(node->GetOp());
    DeconvParam* param = deconv_op->GetParam();

    Tensor* input_tensor = node->GetInputTensor(0);
    TShape inshape = input_tensor->GetShape();

    if(0 == layout)    // nchw
    {
        op_param.batch = inshape.Shape(0);
        op_param.in_shape[0] = inshape.Shape(1);
        op_param.in_shape[1] = inshape.Shape(2);
        op_param.in_shape[2] = inshape.Shape(3);
    }
    else    // nhwc
    {
        op_param.batch = inshape.Shape(0);
        op_param.in_shape[0] = inshape.Shape(3);
        op_param.in_shape[1] = inshape.Shape(1);
        op_param.in_shape[2] = inshape.Shape(2);
    }

    /* kernel quant param */
    Tensor* kernel_tensor = node->GetInputTensor(1);
    auto* k_quant = kernel_tensor->GetQuantParam();
    if((*k_quant).size() != 0)
    {
        op_param.scale[1] = (*k_quant)[0].scale;
        op_param.zero[1] = (*k_quant)[0].zero_point;
    }

    TShape wshape = kernel_tensor->GetShape();

    if(0 == layout)    // hw
    {
        op_param.kernels[0] = wshape.Shape(2);
        op_param.kernels[1] = wshape.Shape(3);
    }
    else    //
    {
        op_param.kernels[0] = wshape.Shape(1);
        op_param.kernels[1] = wshape.Shape(2);
    }

    /* output quant param */
    Tensor* output_tensor = node->GetOutputTensor(0);
    auto* o_quant = output_tensor->GetQuantParam();
    if((*o_quant).size() != 0)
    {
        op_param.scale[2] = (*o_quant)[0].scale;
        op_param.zero[2] = (*o_quant)[0].zero_point;
    }

    TShape outshape = output_tensor->GetShape();

    if(0 == layout)    // chw
    {
        op_param.out_shape[0] = outshape.Shape(1);
        op_param.out_shape[1] = outshape.Shape(2);
        op_param.out_shape[2] = outshape.Shape(3);
    }
    else
    {
        op_param.out_shape[0] = outshape.Shape(3);
        op_param.out_shape[1] = outshape.Shape(1);
        op_param.out_shape[2] = outshape.Shape(2);
    }

    op_param.strides[0] = param->stride_h;
    op_param.strides[1] = param->stride_w;

    op_param.dilations[1] = param->dilation_h;
    op_param.dilations[0] = param->dilation_w;

    op_param.pads[0] = param->pad_h0;    // pad_h
    op_param.pads[1] = param->pad_w0;    // pad_w

    op_param.group = param->group;
    op_param.activation = param->activation;
    op_param.layout = layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, input_tensor->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefDeconv::Run(Node* node)
{
    // printf("run ref_deconv!!!\n");
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

    /* input quant param */
    auto* in_quant = i_tensor->GetQuantParam();
    if((*in_quant).size() != 0)
    {
        op_param.scale[0] = (*in_quant)[0].scale;
        op_param.zero[0] = (*in_quant)[0].zero_point;
    }

    int ret = kernel_run(input, output, kernel, bias, &op_param);
    if(ret < 0)
        return false;
    if(i_tensor->GetDataType() == TENGINE_DT_INT8)
    {
        auto* o_quant = o_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = op_param.scale[2];
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }
    return true;
}

bool RefDeconv::Postrun(Node* node)
{
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    RefDeconv* ops = new RefDeconv();

    if(node->IsDynamicShape())
        ops->dynamic_shape = true;
    else
        ops->dynamic_shape = false;

    return ops;
}

}    // namespace RefDeconvolutionOps

void RegisterRefDeconv2d(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Deconvolution", RefDeconvolutionOps::SelectFunc,
                                                  RefDeconvolutionOps::default_prio);
}

}    // namespace TEngine
