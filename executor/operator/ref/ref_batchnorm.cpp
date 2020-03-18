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
 * Copyright (c) 2018, Open AI Lab
 * Author: ruizhang@openailab.com
 */
#include <iostream>
#include <functional>
#include <stdlib.h>
#include "kernel_registry.hpp"
#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/batch_norm.hpp"
#include "kernel/batchnorm/ref_batchnorm_kernel.h"
#include <cmath>

namespace TEngine {

namespace RefBatchNormImpl {

struct RefBatchNormOps : public NodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);
    RefBatchNormOps()
    {
        kernel_run = nullptr;
        InitRegistry();
    }
    struct ref_batchnorm_param op_param;
    ref_batchnorm_kernel_t kernel_run;
    KernelRegistry<ref_batchnorm_kernel_t> kernel_registry;
};

void RefBatchNormOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_batchnorm_kernel_t )ref_batchnorm_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_batchnorm_kernel_t )ref_batchnorm_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_batchnorm_kernel_t )ref_batchnorm_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_batchnorm_kernel_t )ref_batchnorm_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_batchnorm_kernel_t )ref_batchnorm_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_batchnorm_kernel_t )ref_batchnorm_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_batchnorm_kernel_t )ref_batchnorm_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_batchnorm_kernel_t )ref_batchnorm_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}
bool RefBatchNormOps::Prerun(Node* node)
{
    BatchNorm* bn_op = dynamic_cast<BatchNorm*>(node->GetOp());
    BatchNormParam* param = bn_op->GetParam();

    const Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
    const TShape& shape = input_tensor->GetShape();
    int channel_num = shape.GetC();
    float* scale_mean = ( float* )mem_alloc(channel_num * sizeof(float));
    float* scale_var_inv = ( float* )mem_alloc(channel_num * sizeof(float));
    const Tensor* mean_tensor = node->GetInputTensor(3);
    const Tensor* var_tensor = node->GetInputTensor(4);
    const float* mean = ( const float* )get_tensor_mem(mean_tensor);
    const float* var = ( const float* )get_tensor_mem(var_tensor);

    float rescale_factor;
    float eps = param->eps;

    rescale_factor = param->rescale_factor ? 1 / param->rescale_factor : 0;
    for(int c = 0; c < channel_num; c++)
    {
        float tmp = std::sqrt(var[c] * rescale_factor + eps);
        scale_var_inv[c] = (float)(1.f / tmp);
        tmp = rescale_factor * scale_var_inv[c];
        scale_mean[c] = (float)(-mean[c] * tmp);
    }
    float* gamma = NULL;
    float* beta = NULL;
    if(!param->caffe_flavor)
    {
        const Tensor* gamma_tensor = node->GetInputTensor(1);
        const Tensor* beta_tensor = node->GetInputTensor(2);
        gamma = ( float* )get_tensor_mem(gamma_tensor);
        beta = ( float* )get_tensor_mem(beta_tensor);
    }
    int layout = exec_attr->graph_layout;
    op_param.iscaffe = param->caffe_flavor;
    op_param.scale_mean = scale_mean;
    op_param.scale_var_inv = scale_var_inv;
    op_param.gamma = gamma;
    op_param.beta = beta;
    op_param.layout = layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }
    return true;
}

bool RefBatchNormOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    const TShape& shape = input_tensor->GetShape();
    const std::vector<int> dims = shape.GetDim();

    if(TENGINE_LAYOUT_NCHW == op_param.layout)
    {
        if(4 == dims.size())
        {
            op_param.input_n = dims[0];
            op_param.input_c = dims[1];
            op_param.input_h = dims[2];
            op_param.input_w = dims[3];
        }
        else if(3 == dims.size())
        {
            op_param.input_n = dims[0];
            op_param.input_c = dims[1];
            op_param.input_w = dims[2];
            op_param.input_h = 1;
        }
        else if(2 == dims.size())
        {
            op_param.input_n = dims[0];
            op_param.input_c = dims[1];
            op_param.input_w = 1;
            op_param.input_h = 1;
        }
        else
        {
            return false;
        }
    }
    else
    {
        if(4 == dims.size())
        {
            op_param.input_n = dims[0];
            op_param.input_c = dims[3];
            op_param.input_h = dims[1];
            op_param.input_w = dims[2];
        }
        else if(3 == dims.size())
        {
            op_param.input_n = dims[0];
            op_param.input_c = dims[2];
            op_param.input_w = dims[1];
            op_param.input_h = 1;
        }
        else
        {
            return false;
        }
    }

    auto* in_quant = input_tensor->GetQuantParam();
    if(in_quant->size())
    {
        op_param.in_scale = (*in_quant)[0].scale;
        op_param.in_zero = (*in_quant)[0].zero_point;
    }
    uint8_t* input = ( uint8_t* )get_tensor_mem(input_tensor);
    Tensor* output_tensor = node->GetOutputTensor(0);
    uint8_t* out_data = ( uint8_t* )get_tensor_mem(output_tensor);
    const int data_type = input_tensor->GetDataType();
    if(data_type == TENGINE_DT_UINT8)
    {
        auto* o_quant = output_tensor->GetQuantParam();
        op_param.out_scale = (*o_quant)[0].scale;
        op_param.out_zero = (*o_quant)[0].zero_point;
    }
    int ret = kernel_run(input, out_data, &op_param);
    if(ret < 0)
        return false;

    if(data_type == TENGINE_DT_INT8)
    {
        Tensor* o_tensor = node->GetOutputTensor(0);
        auto* o_quant = o_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = op_param.out_scale;
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }

    return true;
}

bool RefBatchNormOps::Postrun(Node* node)
{
    free(op_param.scale_mean);
    free(op_param.scale_var_inv);
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    RefBatchNormOps* ops = new RefBatchNormOps();
    return ops;
}

}    // namespace RefBatchNormImpl

void RegisterRefBatchNormOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "BatchNormalization", RefBatchNormImpl::SelectFunc,
                                                  1000);
}

}    // namespace TEngine
