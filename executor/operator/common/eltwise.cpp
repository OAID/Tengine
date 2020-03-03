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
 * Author: chunyinglv@openailab.com
 */
#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/eltwise.hpp"
#include "data_type.hpp"
namespace TEngine {

namespace EltwiseImplCommon {

struct EltwiseOps : public NodeOps
{
    template <typename data_type>
    bool kernel_run(void* output, void* input0, void* input1, int type, const TShape ishape0, const TShape ishape1, const TShape oshape,int input1_count4, int layout)
    {
        data_type* out_ptr = ( data_type* )output;
        data_type* in0 = ( data_type* )input0;
        data_type* in1 = ( data_type* )input1;

        int input_count4 = ishape0.GetSize();
        int input_chan = ishape0.GetC();
        int input_hw = ishape0.GetH() * ishape0.GetW();

        int batch = ishape0.GetN();
        int channel = ishape0.GetC();
        int height = ishape0.GetH();
        int width = ishape0.GetW();

        switch(type)
        {
            case ELT_SUB:
                if(input_count4 == input1_count4)
                {
                    for(int i = 0; i < input_count4; ++i)
                    {
                        *out_ptr++ = (*in0++) - (*in1++);
                    }
                }
                else if(input_chan == input1_count4)
                {
                    if(layout == 0){
                        for(int i = 0; i < input_count4; ++i)
                        {
                            *out_ptr++ = in0[i] - in1[i / input_hw];
                        }
                    } else {
                        for(int b = 0; b < batch; b++){
                            for(int h = 0; h < height; h++){
                                for(int w = 0; w < width; w++){
                                for(int c = 0;  c < channel; c++){
                                    int index = b*channel*height*width + h*width*channel + w*channel +c;
                                out_ptr[index] = in0[index] - in1[c];
                                }
                            }
                            }
                        }
                    }
                }
                else
                    return false;
                break;
            case ELT_SUM:
                if(input1_count4 == 1)
                {
                    for(int i = 0; i < input_count4; ++i)
                    {
                        *out_ptr++ = (*in0++) + in1[0];
                    }
                }
                else if(input_count4 == input1_count4)
                {
                    for(int i = 0; i < input_count4; ++i)
                    {
                        *out_ptr++ = (*in0++) + (*in1++);
                    }
                }
                else if(input_chan == input1_count4)
                {
                    for(int i = 0; i < input_count4; ++i)
                    {
                        *out_ptr++ = in0[i] + in1[i / input_hw];
                    }
                }
                else
                    return false;
                break;
            case ELT_MAX:
                for(int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = std::max(in0[i], in1[i]);
                }
                break;
            case ELT_PROD:
                if(input1_count4 == 1)
                {
                    for(int i = 0; i < input_count4; ++i)
                    {
                        *out_ptr++ = (*in0++) * in1[0];
                    }
                }
                else if(input_count4 == input1_count4)
                {
                    for(int i = 0; i < input_count4; ++i)
                    {
                        *out_ptr++ = in0[i] * in1[i];
                    }
                }
                else if(input_chan == input1_count4)
                {
                    if(layout == 0){
                        for(int i = 0; i < input_count4; ++i)
                        {
                            *out_ptr++ = in0[i] * in1[i / input_hw];
                        }
                    }
                    else {
                        for(int b = 0; b < batch; b++){
                            for(int h = 0; h < height; h++){
                                for(int w = 0; w < width; w++){
                                    for(int c = 0;  c < channel; c++){
                                        int index = b*channel*height*width + h*width*channel + w*channel +c;
                                        out_ptr[index] = in0[index] * in1[c];
                                    }
                                }
                            }
                        }
                    }
                }
                else
                    return false;
                break;
            case ELT_RSQRT:
                for(int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = 1 / sqrt(in0[i]);
                }
                break;
            case ELT_MIN_SCALAR:
                for(int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = std::min((*in0++), in1[0]);
                }
                break;
            case ELT_SUB_SCALAR:
                for(int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = (*in0++) - in1[0];
                }
                break;
            case ELT_PROD_SCALAR:
                for(int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = (*in0++) * in1[0];
                }
                break;
            case ELT_DIV:
                if(input1_count4 == 1)
                {
                    for(int i = 0; i < input_count4; ++i)
                    {
                        *out_ptr++ = in0[i] / in1[0];
                    }
                }   
                else if(input_count4 == input1_count4)
                {
                    for(int i = 0; i < input_count4; ++i)
                    {
                        *out_ptr++ = in0[i] / in1[i];
                    }
                }
                else if(input_count4 == 1)
                {
                    for(int i = 0; i < input1_count4; ++i)
                    {
                        *out_ptr++ = in0[0] / (*in1++);
                    }
                }
                else if(ishape0.GetC() == input1_count4)
                {
                    for(int n = 0; n < ishape0.GetN(); n++)
                    {
                        for(int c = 0; c < ishape0.GetC(); c++)
                        {
                            for(int i = 0; i < input_hw; ++i)
                            {
                                int offset = 0;
                                if(layout == 0)
                                    offset = n * ishape0.GetC() * input_hw + c * input_hw + i;
                                else
                                    offset = n * ishape0.GetC() * input_hw + i * ishape0.GetC() + c;
                                out_ptr[offset] = in0[offset] / in1[c];
                            }
                        }
                    }
                }
                else if(ishape1.GetC() == input_count4)
                {
                    for(int n = 0; n < ishape1.GetN(); n++)
                    {
                        for(int c = 0; c < ishape1.GetC(); c++)
                        {
                            for(int i = 0; i < input_hw; ++i)
                            {
                                int offset = 0;
                                if(layout == 0)
                                    offset = n * ishape1.GetC() * input_hw + c * input_hw + i;
                                else
                                    offset = n * ishape1.GetC() * input_hw + i * ishape1.GetC() + c;

                                out_ptr[offset] = in0[c] / in1[offset];
                            }
                        }
                    }
                }
                else
                {
                    break;
                }
                break;
            case ELT_FLOOR:
                for(int i = 0; i < input_count4; ++i)
                {
                    out_ptr[i] = floor(in0[i]);
                }
                break;
            default:
                break;
        }
        return true;
    }

    bool Run(Node* node)
    {
        // input
        Tensor* input_tensor0 = node->GetInputTensor(0);
        int element_size = DataType::GetTypeSize(input_tensor0->GetDataType());
        const TShape& ishape = input_tensor0->GetShape();
        void* input0 = get_tensor_mem(input_tensor0);

        Tensor* input_tensor1 = nullptr;
        void* input1 = nullptr;
        int input1_count4 = 0;
        const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
        if(node->GetInputNum() > 1)
        {
            input_tensor1 = node->GetInputTensor(1);
            input1 = get_tensor_mem(input_tensor1);
            input1_count4 = input_tensor1->GetTotalSize() / element_size;
        }
        const TShape& ishape1 = (node->GetInputNum() > 1)?input_tensor1->GetShape():ishape;
        // this version only support for input_num=2
        // int input_number=node->GetInputNum();

        // output
        Tensor* output_tensor = node->GetOutputTensor(0);
        const TShape& oshape = output_tensor->GetShape();
        void* output = get_tensor_mem(output_tensor);
        Eltwise* eltwise_op = dynamic_cast<Eltwise*>(node->GetOp());
        EltwiseParam* param = eltwise_op->GetParam();

        bool result = false;
        switch(element_size)
        {
            case 4:
                result = kernel_run<float>(output, input0, input1, param->type, ishape, ishape1, oshape,input1_count4, exec_attr->graph_layout);
                break;
        }

        return result;
    }    // Run

};    // struct EltwiseOps

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    if(data_type != TENGINE_DT_FP32)
        return nullptr;

    EltwiseOps* ops = new EltwiseOps();

    return ops;
}

}    // namespace EltwiseImpl

using namespace EltwiseImplCommon;

void RegisterEltwiseNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "Eltwise", EltwiseImplCommon::SelectFunc, 1000);
}

}    // namespace TEngine
