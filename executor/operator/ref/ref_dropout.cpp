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
 * Author: zpluo@openailab.com
 */

#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/dropout.hpp"

namespace TEngine {

namespace RefDropoutOps {

struct RefDropout : public MTNodeOps
{
    bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    bool Prerun(Node* node) override;
};

bool RefDropout::OnBind(Node* node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    return true;
}

bool RefDropout::Prerun(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    auto i_quant = input_tensor->GetQuantParam();
    auto o_quant = output_tensor->GetQuantParam();

    if(i_quant->size() == 1)
    {
        o_quant->resize(0);
        o_quant->push_back((*i_quant)[0]);
    }
    return true;
}
bool RefDropout::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    void* input_data = get_tensor_mem(input_tensor);
    void* output_data = get_tensor_mem(output_tensor);
    
    if(input_data == output_data)
        return true;

    int data_size = input_tensor->GetTotalSize();
    memcpy(output_data, input_data, data_size);

    return true;
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefDropout* ops = new RefDropout();

    LOG_DEBUG() << "ReluOps RefOp is selected\n";

    return ops;
}

}    // namespace RefDropoutOps
void RegisterDropoutOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Dropout", RefDropoutOps::SelectFunc, 1000);
}
}    // namespace TEngine
