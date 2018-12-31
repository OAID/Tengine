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

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"

namespace TEngine {

namespace SliceImpl {

struct SliceOps : public NodeOps
{
    bool Run(Node* node)
    {
        // currently, only working on channel C (slice_axis=1)
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor0 = node->GetOutputTensor(0);
        Tensor* output_tensor1 = node->GetOutputTensor(1);

        const std::vector<int>& dims = input_tensor->GetShape().GetDim();

        int hw = dims[2] * dims[3];
        int slice_size = dims[1] / 2 * hw;
        int size = dims[1] * hw;
        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output0 = ( float* )get_tensor_mem(output_tensor0);
        float* output1 = ( float* )get_tensor_mem(output_tensor1);

        for(int i = 0; i < dims[0]; i++)
        {
            float* in0 = input + i * size;
            float* in1 = in0 + slice_size;
            for(int j = 0; j < slice_size; j++)
            {
                output0[j] = in0[j];
                output1[j] = in1[j];
            }
        }
        return true;
    }
};

}    // namespace SliceImpl

using namespace SliceImpl;

void RegisterSliceNodeExec(void)
{
    SliceOps* ops = new SliceOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "Slice", ops);
}

}    // namespace TEngine
