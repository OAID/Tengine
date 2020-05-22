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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: qtang@openailab.com
 */

#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/fully_connected.hpp"
#include "fully_connected_x86.h"
#include <math.h>

namespace TEngine {

namespace FCImpl {

struct FcBlasOps : public NodeOps
{
    bool Run(Node* node) override;
};

bool FcBlasOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    Tensor* weight_tensor = node->GetInputTensor(1);
    bool has_bias = node->GetInputNum() > 2 ? true : false;

    float* input = ( float* )get_tensor_mem(input_tensor);
    float* output = ( float* )get_tensor_mem(output_tensor);
    float* weight = ( float* )get_tensor_mem(weight_tensor);

    Tensor* bias_tensor;
    float* bias = nullptr;

    if(has_bias)
    {
        bias_tensor = node->GetInputTensor(2);
        bias = ( float* )get_tensor_mem(bias_tensor);
    }

    const TShape& shape = input_tensor->GetShape();
    const std::vector<int> in_dims = shape.GetDim();
    const TShape& shape1 = output_tensor->GetShape();
    const std::vector<int> out_dims = shape1.GetDim();

    int batch_number = in_dims[0];
    int inc = in_dims[1];
    int inh = 1;
    int inw = 1;

    if(in_dims.size() > 2)
        inh = in_dims[2];

    if(in_dims.size() > 3)
        inw = in_dims[3];

    int in_size = inc * inh * inw;
    int outc = out_dims[1];

    innerproduct(batch_number, inc, inh, inw, outc, weight, input, output, bias);

    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    FcBlasOps* ops = new FcBlasOps();

    return ops;
}
}    // namespace FCImpl

void RegisterFcNodeExec_x86(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("x86", "FullyConnected", FCImpl::SelectFunc, 500))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio \n";
}

}    // namespace TEngine
