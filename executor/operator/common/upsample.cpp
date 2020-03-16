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
 * Copyright (c) 2019, Open AI Lab
 * Author: ruizhang@openailab.com
 */
#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/upsample.hpp"

namespace TEngine {

namespace UpsampleImpl {

struct UpsampleOps : public NodeOps
{
    bool Run(Node* node) override
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);

        Upsample* upsample_op = dynamic_cast<Upsample*>(node->GetOp());
        UpsampleParam* param = upsample_op->GetParam();
        std::vector<int> dims = input_tensor->GetShape().GetDim();
        std::vector<int> out_dims = output_tensor->GetShape().GetDim();
        int scale = param->scale;
        int batch = out_dims[0];
        int channel = out_dims[1];
        int out_h = out_dims[2];
        int out_w = out_dims[3];
        int input_h = dims[2];
        int input_w = dims[3];

        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);

        for(int n = 0; n < batch; ++n)
        {
            for(int c = 0; c < channel; c++)
            {
                for(int h = 0; h < out_h; h++)
                {
                    for(int w = 0; w < out_w; w++)
                    {
                        int in_w = w / scale;
                        int in_h = h / scale;
                        int out_idx = n * channel * out_h * out_w + c * out_h * out_w + h * out_w + w;
                        int in_idx = n * channel * input_h * input_w + c * input_w * input_h + in_h * input_w + in_w;
                        output[out_idx] = input[in_idx];
                    }
                }
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
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    UpsampleOps* ops = new UpsampleOps();

    return ops;
}

}    // namespace UpsampleImpl

using namespace UpsampleImpl;

void RegisterUpsampleNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "Upsample", UpsampleImpl::SelectFunc, 1000);
}

}    // namespace TEngine
