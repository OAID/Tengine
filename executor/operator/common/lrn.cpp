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
#include "operator/lrn.hpp"

namespace TEngine {

namespace LRNImpl {

struct LRNOps : public NodeOps
{
    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);

        LRN* lrn_op = dynamic_cast<LRN*>(node->GetOp());
        LRNParam* param = lrn_op->GetParam();

        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);

        float* square = ( float* )(std::malloc(input_tensor->GetTotalSize()));

        const TShape& shape = input_tensor->GetShape();
        const std::vector<int>& dims = shape.GetDim();

        int n = dims[0];
        int c = dims[1];
        int h = dims[2];
        int w = dims[3];

        int img_size = c * h * w;
        int channel_size = h * w;
        float alpha = param->alpha;
        float beta = param->beta;
        float bias = param->k;
        int local_size = param->local_size;

        float* accum_square = ( float* )(std::malloc(channel_size * sizeof(float)));

        for(int i = 0; i < n; i++)
        {
            /* get square value */

            float* img_base = input + i * img_size;

            for(int j = 0; j < img_size; j++)
                square[j] = img_base[j] * img_base[j] + bias;

            if(param->norm_region == LRN_ACROSS_CHANNELS)
            {
                float alpha_over_size = alpha / local_size;

                for(int j = 0; j < c; j++)
                {
                    int c_start = j - local_size / 2;
                    int c_end = j + local_size / 2;

                    std::memset(accum_square, 0x0, channel_size * sizeof(float));

                    for(int l = c_start; l <= c_end; l++)
                    {
                        if(l < 0 || l >= c)
                            continue;

                        for(int n = 0; n < channel_size; n++)
                        {
                            accum_square[n] += square[l * channel_size + n];
                        }
                    }

                    /* get the output */

                    for(int n = 0; n < channel_size; n++)
                    {
                        int offset = i * img_size + j * channel_size + n;
                        output[offset] = input[offset] * std::pow(1.0f + alpha_over_size * accum_square[n], -beta);
                    }
                }
            }
            else
            {
                std::cout << "LRN: IN CHANNEL, TO BE IMPLEMENTED\n";
            }
        }

        std::free(square);
        std::free(accum_square);

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

    LRNOps* ops = new LRNOps();

    return ops;
}

}    // namespace LRNImpl

using namespace LRNImpl;

void RegisterLRN_NodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "LRN", LRNImpl::SelectFunc, 1000);
}

}    // namespace TEngine