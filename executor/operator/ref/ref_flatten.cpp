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
#include "operator/flatten.hpp"

namespace TEngine {

namespace RefFlattenOps {

struct RefFlatten : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;

    RefFlatten(void) {}
};

bool RefFlatten::Prerun(Node* node)
{
    Tensor* output_tensor = node->GetOutputTensor(0);
    int data_type = -1;
    Tensor* input_tensor = node->GetInputTensor(0);
    data_type = input_tensor->GetDataType();
    auto* in_quant = input_tensor->GetQuantParam();
    float scale = 0;
    if((*in_quant).size() != 0)
    {
        scale = (*in_quant)[0].scale;
    }
    else
    {
        scale = 1;
    }

    if(data_type == TENGINE_DT_INT8)
    {
        auto* o_quant = output_tensor->GetQuantParam();
        QuantParam q_param;
        q_param.scale = scale;
        o_quant->resize(0);
        o_quant->push_back(q_param);
    }

    return true;
}

bool RefFlatten::OnBind(Node* node)
{
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);
    return true;
}

bool RefFlatten::Run(Node* node)
{
    return true;
}

bool RefFlatten::Postrun(Node* node)
{
    return true;
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefFlatten* ops = new RefFlatten();

    LOG_DEBUG() << "ReluOps RefOp is selected\n";

    return ops;
}

}    // namespace RefFlattenOps
void RegisterFlattenOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Flatten", RefFlattenOps::SelectFunc, 1000);
}
}    // namespace TEngine
