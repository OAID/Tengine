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
 * Author: haitao@openailab.com
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
#include "operator/concat.hpp"
#include "data_type.hpp"

extern "C" void concat_neon(void* output, void* input, int input_size);

namespace TEngine {

namespace ConcatImpl {

struct ConcatOps : public NodeOps
{
    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        int element_size = DataType::GetTypeSize(input_tensor->GetDataType());
        Tensor* output_tensor = node->GetOutputTensor(0);
        Concat* concat_op = dynamic_cast<Concat*>(node->GetOp());
        ConcatParam* param = concat_op->GetParam();

        std::vector<int> dims = output_tensor->GetShape().GetDim();
        int axis = param->axis;
        int out_size, in_size, on_size;
        out_size = 1;
        for(int i = 0; i < axis; i++)
        {
            out_size *= dims[i];
        }
        in_size = element_size;
        for(size_t i = axis + 1; i < dims.size(); i++)
        {
            in_size *= dims[i];
        }

        uint8_t* output = ( uint8_t* )get_tensor_mem(output_tensor);
        uint8_t* output_ptr = output;
        int input_number = node->GetInputNum();
        int out_axis = dims[axis];
        int offset_concat_axis = 0;
        for(int i = 0; i < input_number; ++i)
        {
            input_tensor = node->GetInputTensor(i);
            uint8_t* input = ( uint8_t* )get_tensor_mem(input_tensor);
            std::vector<int> in_dims = input_tensor->GetShape().GetDim();
            on_size = in_dims[axis];
            for(int n = 0; n < out_size; ++n)
            {
                memcpy(output_ptr + (n * out_axis + offset_concat_axis) * in_size, input + n * on_size * in_size,
                           (on_size * in_size));
            }
            offset_concat_axis += on_size;
        }

        return true;
    }
};

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    if(data_type != TENGINE_DT_FP32)
        return nullptr;

    ConcatOps* ops = new ConcatOps();

    return ops;
}

}    // namespace ConcatImpl

using namespace ConcatImpl;

void RegisterConcatNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "Concat", ConcatImpl::SelectFunc, 1000);
}

}    // namespace TEngine
