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

#include "operator/pad.hpp"
#include "kernel/pad/ref_pad_kernel.h"

namespace TEngine {

namespace RefPadOps {

struct RefPad : public MTNodeOps
{
    bool Prerun(Node* node) override;

    bool Run(Node* node) override;
    void InitRegistry(void);

    ref_pad_t kernel_run;
    pad_param param;

    KernelRegistry<ref_pad_t> kernel_registry;
    RefPad(void)
    {
        InitRegistry();
    }
};

bool RefPad::Prerun(Node* node)
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
static int get_scale_zero(Tensor* itensor, Tensor* otensor, pad_param* param)
{
    auto* i_quant = itensor->GetQuantParam();
    auto* o_quant = otensor->GetQuantParam();
    if(i_quant->size() != 1)
        return -1;
    param->scale[0] = (*i_quant)[0].scale;
    if(itensor->GetDataType() == TENGINE_DT_UINT8)
    {
        if(o_quant->size() != 1)
            return -1;

        param->scale[1] = (*o_quant)[0].scale;
        param->zero[1] = (*o_quant)[0].zero_point;

        param->zero[0] = (*i_quant)[0].zero_point;
    }
    return 0;
}

bool RefPad::Run(Node* node)
{
    Pad* pad_op = dynamic_cast<Pad*>(node->GetOp());
    PadParam* op_param = pad_op->GetParam();
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* out_tensor = node->GetOutputTensor(0);
    // int element_size = DataType::GetTypeSize(out_tensor->GetDataType());
    // int out_size= out_tensor->GetTotalSize() / element_size;

    if(input_tensor->GetDataType() == TENGINE_DT_INT8 || input_tensor->GetDataType() == TENGINE_DT_UINT8)
    {
        if(get_scale_zero(input_tensor, out_tensor, &param) < 0)
            return false;
    }
    const TShape& i_shape = input_tensor->GetShape();
    std::vector<int> i_dims = i_shape.GetDim();

    const TShape& o_shape = out_tensor->GetShape();
    std::vector<int> o_dims = o_shape.GetDim();

    int in_n = i_shape.GetN();
    int in_h = i_shape.GetH();
    int in_w = i_shape.GetW();
    int in_c = i_shape.GetC();

    int out_n = o_shape.GetN();
    int out_h = o_shape.GetH();
    int out_w = o_shape.GetW();
    int out_c = o_shape.GetC();

    int in_size = in_n * in_h * in_w * in_c;
    int out_size = out_n * out_h * out_w * out_c;

    param.mode = op_param->mode;
    if(param.mode == 0)
    {
        param.cv_f32 = op_param->value;
        param.cv_f16 = ( __fp16 )fp32_to_fp16(op_param->value);
        param.cv_int8 = op_param->value;
        param.cv_uint8 = op_param->value;
    }
    param.pad_0_h = op_param->pad_0_h;
    param.pad_0_w = op_param->pad_0_w;
    param.pad_1_h = op_param->pad_1_h;
    param.pad_1_w = op_param->pad_1_w;
    param.pad_2_h = op_param->pad_2_h;
    param.pad_2_w = op_param->pad_2_w;
    param.pad_3_h = op_param->pad_3_h;
    param.pad_3_w = op_param->pad_3_w;

    param.in_n = in_n;
    param.in_h = in_h;
    param.in_w = in_w;
    param.in_c = in_c;

    param.out_n = out_n;
    param.out_h = out_h;
    param.out_w = out_w;

    param.in_size = in_size;
    param.out_size = out_size;

    void* in_data = get_tensor_mem(input_tensor);
    void* out_data = get_tensor_mem(out_tensor);

    int ret = kernel_run(in_data, out_data, &param);
    if(input_tensor->GetDataType() == TENGINE_DT_INT8)
    {
        auto* i_quant = input_tensor->GetQuantParam();
        auto* o_quant = out_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = (*i_quant)[0].scale;
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }
    if(ret < 0)
        return false;
    else
        return true;
}

void RefPad::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_pad_t )ref_pad_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_pad_t )ref_pad_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif
#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_pad_t )ref_pad_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_pad_t )ref_pad_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_pad_t )ref_pad_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_pad_t )ref_pad_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif
#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_pad_t )ref_pad_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_pad_t )ref_pad_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefPad* ops = new RefPad();

    LOG_DEBUG() << "Pad RefOp is selected\n";

    return ops;
}

}    // namespace RefPadOps
void RegisterPadOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Pad", RefPadOps::SelectFunc, 1000);
}
}    // namespace TEngine
