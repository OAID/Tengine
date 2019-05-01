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

#include "kernel/eltwise/eltwise.h"

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/eltwise.hpp"

namespace TEngine {

namespace RefEltwiseOps {



struct EltwiseOps : public MTNodeOps
{
    bool Prerun(Node * node) override; 
    bool Run(Node * node) override; 
    bool Postrun(Node * node) override;
    void InitRegistry(void);

    eltwise_param op_param;
    eltwise_t  kernel_run;

    KernelRegistry<eltwise_t>  kernel_registry;

    EltwiseOps(void) 
    {
       kernel_run=nullptr;

       InitRegistry();
    }
};
static int get_scale_zero(Tensor* itensor,Tensor * otensor,eltwise_param* param)
{
    auto* i_quant = itensor->GetQuantParam();
    auto* o_quant = otensor->GetQuantParam();
    if( i_quant->size() != 1 )
    {
        LOG_ERROR()<<"Input quant size: ("<<i_quant->size()<<")\n";
        return -1;
    }
    param->scale[0] = (*i_quant)[0].scale;
    if(itensor->GetDataType() == TENGINE_DT_UINT8)
    {
        if( o_quant->size() != 1)
        {
            LOG_ERROR()<<"Output quant size: ("<<o_quant->size()<<")\n";
            return -1;
        }

        param->scale[2] = (*o_quant)[0].scale;
        param->zero[2] = (*o_quant)[0].zero_point;

        param->zero[0] = (*i_quant)[0].zero_point;
    }

    return 0;
}
static int get_scale_zero_1(Tensor* itensor,eltwise_param* param)
{
    auto* i_quant = itensor->GetQuantParam();
    if( i_quant->size() != 1 )
    {
        LOG_ERROR()<<"Input quant size: ("<<i_quant->size()<<")\n";
        return -1;
    }
    param->scale[1] = (*i_quant)[0].scale;
    if(itensor->GetDataType() == TENGINE_DT_UINT8)
    {

        param->zero[1] = (*i_quant)[0].zero_point;
    }
    return 0;
}

bool EltwiseOps::Prerun(Node * node)
{
    Tensor * input_tensor=node->GetInputTensor(0);

    int  layout=exec_attr->graph_layout;

    if(!kernel_registry.GetKernel(kernel_run,layout,input_tensor->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }
 
    //int elem_size=DataType::GetTypeSize(input->GetDataType());

    return true;
}


bool EltwiseOps::Run(Node * node)
{
    Tensor* input_tensor0 = node->GetInputTensor(0);
    int element_size = DataType::GetTypeSize(input_tensor0->GetDataType());
    const TShape& ishape = input_tensor0->GetShape();
    void* input0 = get_tensor_mem(input_tensor0);
    Tensor* input_tensor1 = nullptr;
    void* input1 = nullptr;
    int input1_count4 = 0;
    int input_chan_1 = 0;
    int input_hw_1 = 0;
    int input_h_1 = 0;
    int input_w_1 = 0;
    int input_n_1 = 0;
    // this version only support for input_num=2
    // int input_number=node->GetInputNum();

    // output
    Tensor* output_tensor = node->GetOutputTensor(0);
    if(input_tensor0->GetDataType() == TENGINE_DT_INT8 ||input_tensor0->GetDataType() == TENGINE_DT_UINT8 )
    {
        if(get_scale_zero(input_tensor0,output_tensor, &op_param) < 0)
            return false;
    }

    if(node->GetInputNum() > 1)
    {
        input_tensor1 = node->GetInputTensor(1);
        const TShape& ishape1 = input_tensor1->GetShape();
        input1 = get_tensor_mem(input_tensor1);
        input1_count4 = input_tensor1->GetTotalSize() / element_size;
        input_n_1=ishape1.GetN();
        input_chan_1 = ishape1.GetC();
        input_hw_1 = ishape1.GetH() * ishape1.GetW();
        input_h_1=ishape1.GetH();
        input_w_1=ishape1.GetW();

        if(input_tensor1->GetDataType() == TENGINE_DT_INT8 ||
        input_tensor1->GetDataType() == TENGINE_DT_UINT8 )
        {   
            if(get_scale_zero_1(input_tensor1, &op_param) < 0)
                return false;
        }
    }
    int layout = ishape.GetDataLayout();
    void* output = get_tensor_mem(output_tensor);
    Eltwise* eltwise_op = dynamic_cast<Eltwise*>(node->GetOp());
    EltwiseParam* param = eltwise_op->GetParam();
    int input_count4 = input_tensor0->GetTotalSize() / element_size;
    int input_chan = ishape.GetC();
    int input_hw = ishape.GetH() * ishape.GetW();
    int input_h=ishape.GetH();
    int input_w=ishape.GetW();
    int input_n=ishape.GetN();
    //get out_tensor size
    Tensor* output_tensor0 = node->GetOutputTensor(0);
    int out_element_size = DataType::GetTypeSize(output_tensor0->GetDataType());
    int out_size = output_tensor0->GetTotalSize()/out_element_size;
    float * output_buf=(float *)malloc(sizeof(float)*out_size);
    int ret=kernel_run(output, input0, input1, param->type, input_count4,
                     input_chan,input_chan_1,input_hw,input_hw_1, input1_count4,
                     input_h,input_w,input_h_1,input_w_1,input_n,input_n_1,layout,
                     out_size,output_buf,&op_param);
    free(output_buf);

    if(input_tensor1->GetDataType() == TENGINE_DT_INT8 
        || input_tensor0->GetDataType() == TENGINE_DT_INT8)
    {

        auto* o_quant = output_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale =op_param.scale[2];
        o_quant->resize(0);
        o_quant->push_back(q_param);

    }


    if(ret<0)
         return false;
    else
         return true;
}

bool EltwiseOps::Postrun(Node * node)
{
    return true;
}

void EltwiseOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register((eltwise_t)eltwise_fp32,TENGINE_LAYOUT_NCHW,TENGINE_DT_FP32);
    kernel_registry.Register((eltwise_t)eltwise_fp32,TENGINE_LAYOUT_NHWC,TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register((eltwise_t)eltwise_fp16,TENGINE_LAYOUT_NCHW,TENGINE_DT_FP16);
    kernel_registry.Register((eltwise_t)eltwise_fp16,TENGINE_LAYOUT_NHWC,TENGINE_DT_FP16);
#endif
#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register((eltwise_t)eltwise_int8,TENGINE_LAYOUT_NCHW,TENGINE_DT_INT8);
    kernel_registry.Register((eltwise_t)eltwise_int8,TENGINE_LAYOUT_NHWC,TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register((eltwise_t)eltwise_uint8,TENGINE_LAYOUT_NCHW,TENGINE_DT_UINT8);
    kernel_registry.Register((eltwise_t)eltwise_uint8,TENGINE_LAYOUT_NHWC,TENGINE_DT_UINT8);
#endif

}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    EltwiseOps* ops = new EltwiseOps();

    LOG_DEBUG()<<"EltwiseOps RefOp is selected\n";

    return ops;
}




}    // namespace RefEltwiseOps
void RegisterEltwiseOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Eltwise", RefEltwiseOps::SelectFunc, 1000);
}

}    // namespace TEngine