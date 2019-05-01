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
 * Author: jingyou@openailab.com
 */
#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>
#include <cmath>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/logistic.hpp"
#include "data_type.hpp"

namespace TEngine {

namespace LogisticImpl {

struct LogisticOps : public NodeOps
{
    bool OnBind(Node* node) override
    {
        inplace_t io_map;

        io_map[0] = 0;
        node->SetAttr(ATTR_INPLACE, io_map);

        return true;
    }

    bool Run(Node* node) override
    {
        Tensor* input = node->GetInputTensor(0);
        Tensor* output = node->GetOutputTensor(0);
        int element_size = DataType::GetTypeSize(input->GetDataType());

        int elements = input->GetShape().GetSize();

        if(element_size == 4)
        {
            float* input_ptr = ( float* )get_tensor_mem(input);
            float* output_ptr = ( float* )get_tensor_mem(output);

            for(int i = 0; i < elements; i++)
                output_ptr[i] = 1.f / (1.f + std::exp(-input_ptr[i]));
        }
        else if(element_size == 1)
        {
            uint8_t* input_ptr = ( uint8_t* )get_tensor_mem(input);
            uint8_t* output_ptr = ( uint8_t* )get_tensor_mem(output);
            auto i_quantized = input->GetQuantParam();
            auto o_quantized = output->GetQuantParam();

            float i_scale = (*i_quantized)[0].scale;
            int i_zero = (*i_quantized)[0].zero_point;
            float o_scale = (*o_quantized)[0].scale;
            int o_zero = (*o_quantized)[0].zero_point;

            for(int i = 0; i < elements; i++)
            {
                float real_input = (input_ptr[i] - i_zero) * i_scale;
                float real_output = 1.f / (1.f + std::exp(-real_input));
                output_ptr[i] = std::round(real_output / o_scale) + o_zero;
            }
        }

        return true;
    }
};

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if( (data_type != TENGINE_DT_FP32 && data_type != TENGINE_DT_UINT8)
        || exec_attr->graph_layout != TENGINE_LAYOUT_NHWC)
        return nullptr;

    LogisticOps* ops = new LogisticOps();

    return ops;
}

}    // namespace LogisticImpl

using namespace LogisticImpl;

void RegisterLogisticNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "Logistic", LogisticImpl::SelectFunc, 1000);
}

}    // namespace TEngine
