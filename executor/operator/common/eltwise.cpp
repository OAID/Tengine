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

namespace EltwiseImpl {

struct EltwiseOps : public NodeOps
{
    template <typename data_type>
    bool kernel_run(void* output, void* input0, void* input1, int type, const TShape ishape, int input1_count4)
    {
        data_type* out_ptr = ( data_type* )output;
        data_type* in0 = ( data_type* )input0;
        data_type* in1 = ( data_type* )input1;

        int input_count4 = ishape.GetSize();
        int input_chan = ishape.GetC();
        int input_hw = ishape.GetH() * ishape.GetW();

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
                    for(int i = 0; i < input_count4; ++i)
                    {
                        *out_ptr++ = in0[i] - in1[i / input_hw];
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
                    for(int i = 0; i < input_count4; ++i)
                    {
                        *out_ptr++ = in0[i] * in1[i / input_hw];
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

        if(node->GetInputNum() > 1)
        {
            input_tensor1 = node->GetInputTensor(1);
            input1 = get_tensor_mem(input_tensor1);
            input1_count4 = input_tensor1->GetTotalSize() / element_size;
        }

        // this version only support for input_num=2
        // int input_number=node->GetInputNum();

        // output
        Tensor* output_tensor = node->GetOutputTensor(0);
        void* output = get_tensor_mem(output_tensor);
        Eltwise* eltwise_op = dynamic_cast<Eltwise*>(node->GetOp());
        EltwiseParam* param = eltwise_op->GetParam();

        bool result = false;
        switch(element_size)
        {
            case 4:
                result = kernel_run<float>(output, input0, input1, param->type, ishape, input1_count4);
                break;
#ifdef CONFIG_FLOAT16
            case 2:
                result = kernel_run<__fp16>(output, input0, input1, param->type, ishape, input1_count4);
                break;
#endif
            case 1:
                result = kernel_run<char>(output, input0, input1, param->type, ishape, input1_count4);
                break;
        }

        return result;
    }    // Run

};    // struct EltwiseOps

}    // namespace EltwiseImpl

using namespace EltwiseImpl;

void RegisterEltwiseNodeExec(void)
{
    EltwiseOps* ops = new EltwiseOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "Eltwise", ops);
}

}    // namespace TEngine
