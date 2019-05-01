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

inline static int get_scale_zero(Tensor* itensor, Tensor* otensor, Tensor* ktensor, op_data* param)
{
    auto* i_quant = itensor->GetQuantParam();
    auto* k_quant = ktensor->GetQuantParam();
    auto* o_quant = otensor->GetQuantParam();
    if( i_quant->size() != 1 || k_quant->size() != 1)
    {
        std::cerr<<"quant size: input("<< i_quant->size()<<"),kernel("<<k_quant->size()<<")\n";
        return -1;
    }
    param->scale[0] = (*i_quant)[0].scale;
    param->scale[1] = (*k_quant)[0].scale;
    if(itensor->GetDataType() == TENGINE_DT_UINT8)
    {
        if( o_quant->size() != 1)
        {
            std::cerr<<"output quant size: "<<o_quant->size()<<"\n";
            return -1;
        }

        param->scale[2] = (*o_quant)[0].scale;
        param->zero[2] = (*o_quant)[0].zero_point;

        param->zero[0] = (*i_quant)[0].zero_point;
        param->zero[1] = (*k_quant)[0].zero_point;
    }
    //printf("scale: %f,%f,%f   --     zero : %d,%d,%d \n",
    //            param->scale[0],param->scale[1],param->scale[2],
    //            param->zero[0],param->zero[1],param->zero[2]);
    return 0;
}

struct RefConv : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Reshape(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    bool dynamic_shape;
    op_data op_param;
    
    ref_conv_kernel_t  kernel_run;
    KernelRegistry<ref_conv_kernel_t>  kernel_registry;
    RefConv(void) 
    {
        kernel_run=nullptr;
        InitRegistry();
    }
};
void RefConv::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register((ref_conv_kernel_t)ref_conv_fp32,TENGINE_LAYOUT_NCHW,TENGINE_DT_FP32);
    kernel_registry.Register((ref_conv_kernel_t)ref_conv_fp32,TENGINE_LAYOUT_NHWC,TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register((ref_conv_kernel_t)ref_conv_fp16,TENGINE_LAYOUT_NCHW,TENGINE_DT_FP16);
    kernel_registry.Register((ref_conv_kernel_t)ref_conv_fp16,TENGINE_LAYOUT_NHWC,TENGINE_DT_FP16);
#endif

#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register((ref_conv_kernel_t)ref_conv_int8,TENGINE_LAYOUT_NCHW,TENGINE_DT_INT8);
    kernel_registry.Register((ref_conv_kernel_t)ref_conv_int8,TENGINE_LAYOUT_NHWC,TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register((ref_conv_kernel_t)ref_conv_uint8,TENGINE_LAYOUT_NCHW,TENGINE_DT_UINT8);
    kernel_registry.Register((ref_conv_kernel_t)ref_conv_uint8,TENGINE_LAYOUT_NHWC,TENGINE_DT_UINT8);
#endif

}

bool RefConv::Prerun(Node* node)
{
    int  layout=exec_attr->graph_layout;
    
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

    if(!kernel_registry.GetKernel(kernel_run,layout,input_tensor->GetDataType()))
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
    //printf("---------------------------- Run ref_conv!!!\n");
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

    /* Get input,kernel,output scale & zero */
    /* Current: one tensor has only one quantparam(scale)*/
    if(i_tensor->GetDataType() == TENGINE_DT_INT8 ||
        i_tensor->GetDataType() == TENGINE_DT_UINT8 )
    {
        if(get_scale_zero(i_tensor, o_tensor, k_tensor, &op_param) < 0)
            return false;
    }

    int ret = kernel_run(input,output,kernel,bias,&op_param);
    if(i_tensor->GetDataType() == TENGINE_DT_INT8)
    {
        auto* o_quant = o_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = op_param.scale[2];
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }
    if(ret<0)
        return false;
    return true;
}

bool RefConv::Postrun(Node* node)
{
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
