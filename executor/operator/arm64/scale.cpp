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

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/scale.hpp"

namespace TEngine {

namespace ScaleImpl {

extern "C" void scale_neon(float* input, float* gamma, int channel, int channel_size, float* output);
extern "C" void scale_neon_bias(float* input, float* gamma, int channel, int channel_size, float* output, float* bias);

struct ScaleOps : public NodeOps
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
        const Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        const TShape& shape = input_tensor->GetShape();

        const Tensor* gamma_tensor = node->GetInputTensor(1);
        const Tensor* beta_tensor = node->GetInputTensor(2);

        const std::vector<int> dims = shape.GetDim();

        int batch_number = dims[0];
        int channel_num = dims[1];
        int channel_size = dims[2] * dims[3];
        int img_size = channel_num * channel_size;

        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);
        float* gamma = ( float* )get_tensor_mem(gamma_tensor);

        if(beta_tensor == nullptr)
        {
            for(int i = 0; i < batch_number; i++)
            {
                scale_neon(input, gamma, channel_num, channel_size, output);
                input += img_size;
                output += img_size;
            }
        }
        else
        {
            float* beta = ( float* )get_tensor_mem(beta_tensor);

            for(int i = 0; i < batch_number; i++)
            {
                scale_neon_bias(input, gamma, channel_num, channel_size, output, beta);
                input += img_size;
                output += img_size;
            }
        }

        return true;
    }
};

}    // namespace ScaleImpl

using namespace ScaleImpl;

void RegisterScaleNodeExec(void)
{
    ScaleOps* ops = new ScaleOps();

    NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Scale", ops);
}

}    // namespace TEngine
