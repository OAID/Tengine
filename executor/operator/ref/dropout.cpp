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

// #include "kernel/Dropout/Dropout_kernel.h"

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/dropout.hpp"

namespace TEngine {

namespace RefDropoutOps {



struct RefDropout : public MTNodeOps
{
    bool Prerun(Node * node) override; 
    bool OnBind(Node * node) override; 
    bool Run(Node * node) override;
    bool Postrun(Node * node) override;
    // void InitRegistry(void);
    // Dropout_param op_param;
    // void * mem;
    // Dropout_t  kernel_run;


    // KernelRegistry<Dropout_t>  kernel_registry;
    
    RefDropout(void) 
    {
    //    mem=nullptr; 
    //    kernel_run=nullptr;

    //    InitRegistry();
    }
};


bool RefDropout::Prerun(Node * node)
{
    // Tensor * input=node->GetInputTensor(0);
    // Tensor* output_tensor = node->GetOutputTensor(0);
    // int  layout=exec_attr->graph_layout;

    // if(input->GetDataType() == TENGINE_DT_INT8 ||
    //     input->GetDataType() == TENGINE_DT_UINT8 )
    // {
    //     if(get_scale_zero(input, output_tensor, &op_param) < 0)
    //         return false;
    // }

      
    // if(!kernel_registry.GetKernel(kernel_run,layout,input->GetDataType()))
    // {
    //     set_tengine_errno(ENOENT);
    //     return false;
    // }

    return true;
}

bool RefDropout::OnBind(Node * node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    return true;
}

bool RefDropout::Run(Node * node)
{
    // Tensor* input_tensor = node->GetInputTensor(0);
    // Tensor* output_tensor = node->GetOutputTensor(0);
    // const TShape& shape = input_tensor->GetShape();
    // void* data = get_tensor_mem(input_tensor);
    // void* out_data = get_tensor_mem(output_tensor);
  
    // int size = shape.GetSize();
    // int ret=kernel_run(data,out_data,size,&op_param);
   
    // if(ret<0)
    //     return false;
    // else
    //     return true;

    Tensor* input = node->GetInputTensor(0);
    Tensor* output = node->GetOutputTensor(0);
    auto i_quant = input->GetQuantParam();
    auto o_quant = output->GetQuantParam();
    if(i_quant->size() != 1)
    {
        LOG_ERROR()<<"input quant param num isnot 1 \n";
        return false;
    }
    o_quant->resize(0);
    o_quant->push_back((*i_quant)[0]);

    return true;
}

bool RefDropout::Postrun(Node * node)
{
    return true;
}

// void RefDropout::InitRegistry(void)
// {
// #ifdef CONFIG_KERNEL_FP32
//     kernel_registry.Register((Dropout_t)Dropout_fp32,TENGINE_LAYOUT_NCHW,TENGINE_DT_FP32);
//     kernel_registry.Register((Dropout_t)Dropout_fp32,TENGINE_LAYOUT_NHWC,TENGINE_DT_FP32);
// #endif

// #ifdef CONFIG_KERNEL_FP16
//     kernel_registry.Register((Dropout_t)Dropout_fp16,TENGINE_LAYOUT_NCHW,TENGINE_DT_FP16);
//     kernel_registry.Register((Dropout_t)Dropout_fp16,TENGINE_LAYOUT_NHWC,TENGINE_DT_FP16);
// #endif
// #ifdef CONFIG_KERNEL_INT8
//     kernel_registry.Register((Dropout_t)Dropout_int8,TENGINE_LAYOUT_NCHW,TENGINE_DT_INT8);
//     kernel_registry.Register((Dropout_t)Dropout_int8,TENGINE_LAYOUT_NHWC,TENGINE_DT_INT8);
// #endif

// #ifdef CONFIG_KERNEL_UINT8
//     kernel_registry.Register((Dropout_t)Dropout_uint8,TENGINE_LAYOUT_NCHW,TENGINE_DT_UINT8);
//     kernel_registry.Register((Dropout_t)Dropout_uint8,TENGINE_LAYOUT_NHWC,TENGINE_DT_UINT8);
// #endif

// }

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefDropout* ops = new RefDropout();

    LOG_DEBUG()<<"ReluOps RefOp is selected\n";

    return ops;
}




}    // namespace RefReluOps
void RegisterDropoutOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Dropout", RefDropoutOps::SelectFunc, 1000);
}
}    // namespace TEngine
