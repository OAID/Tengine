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

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/relu.hpp"
#include "data_type.hpp"

namespace TEngine {

namespace ReLuImpl {

struct ReLuOps : public NodeOps
{
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

    template <typename data_type> void kernel_run(void* input, void* out, int size, float negative_slope)
    {
        data_type* in_data = ( data_type* )input;
        data_type* out_data = ( data_type* )out;
        if(negative_slope == 0)
        {
            for(int i = 0; i < size; i++)
            {
                out_data[i] = MAX(in_data[i], 0);
            }
        }
        else
        {
            for(int i = 0; i < size; i++)
            {
                out_data[i] = MAX(in_data[i], 0.f) + negative_slope * MIN(in_data[i], 0.f);
            }
        }
    }

    bool Run(Node* node) override
    {
        // input tensor and output tensor is the same
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        int element_size = DataType::GetTypeSize(input_tensor->GetDataType());
        const TShape& shape = input_tensor->GetShape();
        int elem_num = shape.GetSize();

        ReLu* relu_op = dynamic_cast<ReLu*>(node->GetOp());
        ReLuParam* param = relu_op->GetParam();
        float* data = ( float* )get_tensor_mem(input_tensor);
        float* data1 = ( float* )get_tensor_mem(output_tensor);

        switch(element_size)
        {
            case 4:
                kernel_run<float>(( void* )data, ( void* )data1, elem_num, param->negative_slope);
                break;
#if CONFIG_KERNEL_FP16 && __ARM_ARCH >= 8
            case 2:
                kernel_run<__fp16>(( void* )data, ( void* )data1, elem_num, ( float )param->negative_slope);
                break;
#endif
            case 1:
                kernel_run<char>(( void* )data, ( void* )data1, elem_num, ( float )param->negative_slope);
                break;
        }
        return true;
    }
};

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;
    ReLuOps* ops = new ReLuOps();

    return ops;
}

}    // namespace ReLuImpl

using namespace ReLuImpl;

void RegisterReLuNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "ReLu", ReLuImpl::SelectFunc, 1000);
}

}    // namespace TEngine
