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
#include "operator/normalize.hpp"
#include <math.h>

namespace TEngine {

namespace NormalizeImpl {

struct NormalizeOps : public NodeOps
{
    bool Prerun(Node* node)
    {
        const Tensor* input_tensor = node->GetInputTensor(0);
        const TShape& shape = input_tensor->GetShape();
        const std::vector<int> dims = shape.GetDim();

        float* buffer = ( float* )std::malloc(sizeof(float) * dims[2] * dims[3]);
        (*node)["buffer"] = buffer;

        return true;
    }

    void norm_channel(float* input, float* output, float* buffer, float* scale, int hw, int channel)
    {
        for(int j = 0; j < hw; j++)
        {
            buffer[j] = 0;
            for(int i = 0; i < channel; i++)
            {
                float data = *(input + i * hw + j);
                buffer[j] += (data * data);
            }
            buffer[j] = 1.f / sqrt(buffer[j]);
        }

        float* out_ptr = output;
        for(int j = 0; j < hw; j++)
        {
            for(int i = 0; i < channel; i++)
            {
                float data = *(input + i * hw + j);
                *(out_ptr + i * hw + j) = data * buffer[j] * scale[i];
            }
        }
    }

    bool Run(Node* node)
    {
        const Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        const Tensor* scale_tensor = node->GetInputTensor(1);

        Normalize* normalize_op = dynamic_cast<Normalize*>(node->GetOp());
        NormalizeParam* param_ = normalize_op->GetParam();

        const TShape& shape = input_tensor->GetShape();
        const std::vector<int> dims = shape.GetDim();

        int batch_number = dims[0];
        int channel_num = dims[1];
        int channel_size = dims[2] * dims[3];
        int img_size = channel_num * channel_size;

        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);
        float* scale = ( float* )get_tensor_mem(scale_tensor);
        float* buffer = any_cast<float*>(node->GetAttr("buffer"));

        if(param_->channel_shared == 0 && param_->across_spatial == 0)
            for(int i = 0; i < batch_number; i++)
            {
                norm_channel(input, output, buffer, scale, channel_size, channel_num);
                input += img_size;
                output += img_size;
            }
        // other case to be support
        return true;
    }

    bool Postrun(Node* node)
    {
        float* addr;

        addr = any_cast<float*>(node->GetAttr("buffer"));
        std::free(addr);
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

    NormalizeOps* ops = new NormalizeOps();

    return ops;
}

}    // namespace NormalizeImpl

using namespace NormalizeImpl;

void RegisterNormalizeNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "Normalize", NormalizeImpl::SelectFunc, 1000);
}

}    // namespace TEngine
