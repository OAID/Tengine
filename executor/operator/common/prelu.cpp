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

#include <functional>
#include <cstring>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
namespace TEngine {

namespace PreluImpl {

struct PreluOps : public NodeOps
{
    bool OnBind(Node* node)
    {
        // set the inplace feature
        inplace_t io_map;

        io_map[0] = 0;

        node->SetAttr(ATTR_INPLACE, io_map);

        return true;
    }

    bool Run(Node* node)
    {
        // inplace implement
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        const TShape& shape = input_tensor->GetShape();
        const std::vector<int> dims = shape.GetDim();

        int channel_size = dims[2] * dims[3];
        // int img_size=channel_size*dims[1];
        float* data = ( float* )get_tensor_mem(input_tensor);
        float* out_data = ( float* )get_tensor_mem(output_tensor);
        const Tensor* slope_tensor = node->GetInputTensor(1);
        float* slope = ( float* )get_tensor_mem(slope_tensor);
        for(int i = 0; i < dims[0]; i++)
        {
            for(int c = 0; c < dims[1]; c++)
            {
                for(int l = 0; l < channel_size; l++)
                {
                    *out_data = MAX(*data, 0) + slope[c] * MIN(*data, 0.f);
                    out_data++;
                    data++;
                }
            }
        }
        return true;
    }
};

}    // namespace PreluImpl

using namespace PreluImpl;

void RegisterPReLUNodeExec(void)
{
    PreluOps* ops = new PreluOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "PReLU", ops);
}

}    // namespace TEngine
