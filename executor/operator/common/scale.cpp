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
#include <stdlib.h>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/scale.hpp"
#include "data_type.hpp"

namespace TEngine {

namespace ScaleImpl {

template <typename data_type>
void kernel_run(void* in_data, void* out_data, void* gamma_data, void* beta_data, const TShape shape)
{
    const std::vector<int> dims = shape.GetDim();
    int batch_number = dims[0];
    int channel_num = dims[1];
    int channel_size = dims[2] * dims[3];
    int img_size = channel_num * channel_size;

    data_type* input = ( data_type* )in_data;
    data_type* gamma = ( data_type* )gamma_data;
    data_type* beta = ( data_type* )beta_data;
    data_type* output = ( data_type* )out_data;

    for(int i = 0; i < batch_number; i++)
    {
        for(int c = 0; c < channel_num; c++)
        {
            int offset = i * img_size + c * channel_size;
            for(int l = 0; l < channel_size; l++)
            {
                if(beta != nullptr)
                    output[offset + l] = input[offset + l] * gamma[c] + beta[c];
                else
                    output[offset + l] = input[offset + l] * gamma[c];
            }
        }
    }
}

struct ScaleOps : public NodeOps
{
    bool Run(Node* node)
    {
        const Tensor* input_tensor = node->GetInputTensor(0);
        const Tensor* gamma_tensor = node->GetInputTensor(1);
        const Tensor* beta_tensor = node->GetInputTensor(2);
        Tensor* output_tensor = node->GetOutputTensor(0);
        const TShape& shape = input_tensor->GetShape();

        int element_size = DataType::GetTypeSize(input_tensor->GetDataType());

        void* input = get_tensor_mem(input_tensor);
        void* gamma = get_tensor_mem(gamma_tensor);
        void* output = get_tensor_mem(output_tensor);
        void* beta = nullptr;

        if(beta_tensor != nullptr)
        {
            beta = get_tensor_mem(beta_tensor);
        }

        switch(element_size)
        {
            case 4:
                kernel_run<float>(input, output, gamma, beta, shape);
                break;
#ifdef CONFIG_FLOAT16
            case 2:
                kernel_run<__fp16>(input, output, gamma, beta, shape);
                break;
#endif
            case 1:
                kernel_run<char>(input, output, gamma, beta, shape);
                break;
        }

        return true;
    }
};

}    // namespace ScaleImpl

using namespace ScaleImpl;

void RegisterScale_NodeExec(void)
{
    ScaleOps* ops = new ScaleOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "Scale", ops);
}

}    // namespace TEngine
