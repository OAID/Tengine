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
 * Author: zpluo@openailab.com
 */
#include <iostream>
#include <functional>
#include <stdlib.h>
#include "kernel_registry.hpp"
#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "tengine_errno.hpp"
#include "operator/copy.hpp"
#include <cmath>

namespace TEngine {

namespace RefCopyImpl {
// const int default_prio = 1500;
struct RefCopyOps : public NodeOps
{
    bool Prerun(Node* node) override;
    bool OnBind(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
};

bool RefCopyOps::OnBind(Node* node)
{
    // set the inplace feature
    inplace_t io_map;

    io_map[0] = 0;

    node->SetAttr(ATTR_INPLACE, io_map);

    return true;
}

bool RefCopyOps::Prerun(Node* node)
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

bool RefCopyOps::Run(Node* node)
{
    return true;
}

bool RefCopyOps::Postrun(Node* node)
{

    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    RefCopyOps* ops = new RefCopyOps();
    return ops;
}

}    // namespace RefCopyImpl

using namespace RefCopyImpl;

void RegisterRefCopyOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Copy", RefCopyImpl::SelectFunc, 1000);
}

}    // namespace TEngine
