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
#include "operator/permute.hpp"
#include <math.h>

namespace TEngine {

namespace PermuteImpl {

struct PermuteOps : public NodeOps
{
    void permute_hwc(float* input, float* output, int height, int width, int channel, int _wc, int _hw)
    {
        for(int h = 0; h < height; h++)
        {
            float* output_ptr = output + h * _wc;

            for(int w = 0; w < width; w++)
            {
                for(int c = 0; c < channel; c++)
                {
                    const float* input_ptr = input + c * _hw + h * width;
                    output_ptr[w * channel + c] = input_ptr[w];
                }
            }
        }
    }

    bool Run(Node* node)
    {
        const Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);

        Permute* permute_op = dynamic_cast<Permute*>(node->GetOp());
        PermuteParam* param = permute_op->GetParam();

        const TShape& shape = input_tensor->GetShape();
        const std::vector<int> dims = shape.GetDim();

        int batch_number = dims[0];
        int channel = dims[1];
        int width = dims[3];
        int height = dims[2];
        int _wc = width * channel;
        int _hw = width * height;
        int _chw = channel * _hw;

        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);
        // 0231 [bhwc]
        if((param->order0 == 0) && (param->order1 == 2) && (param->order2 == 3) && (param->order3 == 1))
        {
            for(int b = 0; b < batch_number; b++)
            {
                permute_hwc(input, output, height, width, channel, _wc, _hw);
                input += _chw;
                output += _chw;
            }
        }
        // other case to be support
        return true;
    }
};

}    // namespace PermuteImpl

using namespace PermuteImpl;

void RegisterPermuteNodeExec(void)
{
    PermuteOps* ops = new PermuteOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "Permute", ops);
}

}    // namespace TEngine
