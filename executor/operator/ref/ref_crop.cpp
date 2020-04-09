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
 * Author: bingzhang@openailab.com
 */

#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/crop.hpp"

#include "kernel/crop/ref_crop_kernel.h"

namespace TEngine {

namespace RefCropOps {

struct CropOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    //bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    ref_crop_t kernel_run;
    ref_crop_param op_param;
    KernelRegistry<ref_crop_t> kernel_registry;

    CropOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool CropOps::Prerun(Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    int layout = exec_attr->graph_layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        // printf("errorno: %d\n",ENOENT);
        return false;
    }

    return true;
}
/*
bool CropOps::OnBind(Node* node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    return true;
}
*/
bool CropOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    Crop* crop_op = dynamic_cast<Crop*>(node->GetOp());
    CropParam* param = crop_op->GetParam();
    void* in_data = get_tensor_mem(input_tensor);
    void* data = get_tensor_mem(output_tensor);
    TShape& inShape = input_tensor->GetShape();
    std::vector<int> inDims = inShape.GetDim();

    TShape& outShape = output_tensor->GetShape();
    std::vector<int> outDims = outShape.GetDim();

    op_param.num_args=param->num_args;
    op_param.offset_h=param->offset_h;
    op_param.offset_w=param->offset_w;
    op_param.offset_c=param->offset_c;
    op_param.crop_h=param->crop_h;
    op_param.crop_w=param->crop_w;
    op_param.axis=param->axis;
    op_param.flag=param->flag;
    // op_param.iDataN=param->iDataN;
    op_param.iDataH=inDims[2];
    op_param.iDataW=inDims[3];
    op_param.iDataC=inDims[1];
    op_param.oDataN=outDims[0];
    op_param.oDataH=outDims[2];
    op_param.oDataW=outDims[3];
    op_param.oDataC=outDims[1];


    // float scale = 1.f;
    // int zero_point = 0;
    if(input_tensor->GetDataType() == TENGINE_DT_INT8 || input_tensor->GetDataType() == TENGINE_DT_UINT8)
    {
        auto quant_param = input_tensor->GetQuantParam();
        op_param.scale[0] = (*quant_param)[0].scale;
        op_param.zero_point[0] = (*quant_param)[0].zero_point;
        op_param.scale[1]=(*quant_param)[0].scale;
        op_param.zero_point[1] = (*quant_param)[0].zero_point;
        auto out_quant_param = output_tensor->GetQuantParam();
        out_quant_param->resize(0);
        out_quant_param->push_back((*quant_param)[0]);
    }
    int ret = kernel_run(in_data, data,&op_param);

    if(ret < 0)
        return false;
    else
        return true;
}

void CropOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_crop_t )ref_crop_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    // kernel_registry.Register(( ref_crop_t )ref_crop_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_crop_t )ref_crop_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    // kernel_registry.Register(( ref_crop_t )ref_crop_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_crop_t )ref_crop_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    // kernel_registry.Register(( ref_crop_t )ref_crop_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_crop_t )ref_crop_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    // kernel_registry.Register(( ref_crop_t )ref_crop_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    CropOps* ops = new CropOps();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;
    LOG_DEBUG() << "CropOps RefOp is selected\n";

    return ops;
}

}    // namespace RefCropOps
void RegisterRefCropOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Crop", RefCropOps::SelectFunc, 1000);
}
}    // namespace TEngine
