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
#include "operator/reshape.hpp"

namespace TEngine {

namespace ReshapeImpl {

struct ReshapeOps : public NodeOps
{
    bool OnBind(Node* node)
    {
        inplace_t io_map;

        io_map[0] = 0;
        node->SetAttr(ATTR_INPLACE, io_map);

        return true;
    }

    bool Run(Node* node)
    {
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

    ReshapeOps* ops = new ReshapeOps();

    return ops;
}

}    // namespace ReshapeImpl

using namespace ReshapeImpl;

void RegisterReshapeNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "Reshape", ReshapeImpl::SelectFunc, 1000);
}

}    // namespace TEngine
