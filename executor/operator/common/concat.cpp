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
        auto out_quant = output_tensor->GetQuantParam();
        int out_zero = (*out_quant)[0].zero_point;
        float out_scale = (*out_quant)[0].scale;

        Concat* concat_op = dynamic_cast<Concat*>(node->GetOp());
        ConcatParam* param = concat_op->GetParam();

        std::vector<int> dims = input_tensor->GetShape().GetDim();
        std::vector<int> out_dims = output_tensor->GetShape().GetDim();
        int axis = param->axis;
        int out_size, in_size, on_size;
        out_size = 1;
        for(int i = 0; i < axis; i++)
        {
            out_size *= dims[i];
        }
        in_size = element_size;
        for(size_t i = axis + 1; i < out_dims.size(); i++)
        {
            in_size *= dims[i];
        }

        uint8_t* output = ( uint8_t* )get_tensor_mem(output_tensor);
        uint8_t* output_ptr = output;
        int input_number = node->GetInputNum();
        int offset_concat_axis = 0;
        int out_axis = out_dims[axis];
        for(int i = 0; i < input_number; ++i)
        {
            input_tensor = node->GetInputTensor(i);
            uint8_t* input = ( uint8_t* )get_tensor_mem(input_tensor);
            dims = input_tensor->GetShape().GetDim();
            on_size = dims[axis];
            for(int n = 0; n < out_size; ++n)
            {
                if(element_size == 4)
                    memcpy(output_ptr + (n * out_axis + offset_concat_axis) * in_size, input + n * on_size * in_size,
                           (on_size * in_size));
                else if(element_size == 1)
                {
                    auto quant = input_tensor->GetQuantParam();
                    int zero_point = (*quant)[0].zero_point;
                    float scale = (*quant)[0].scale;
                    uint8_t* output_cur = output_ptr + (n * out_axis + offset_concat_axis) * in_size;
                    uint8_t* input_cur = input + n * on_size * in_size;
                    for(int m = 0; m < on_size; m++)
                        for(int n = 0; n < in_size; n++)
                        {
                            output_cur[m * in_size + n] =
                                std::round((input_cur[m * in_size + n] - zero_point) * scale / out_scale) + out_zero;
                        }
                }
            }
            offset_concat_axis += on_size;
        }

        return true;
    }
};

}    // namespace ConcatImpl

using namespace ConcatImpl;

void RegisterConcatNodeExec(void)
{
    ConcatOps* ops = new ConcatOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "Concat", ops);
}

}    // namespace TEngine
