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
 * Author: bingzhang@openailab.com
 */
#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>
#include <complex>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "data_type.hpp"
#include "operator/power.hpp"

namespace TEngine {

namespace PowerImpl {

struct PowerOps : public NodeOps
{
    bool Run(Node* node)
    {
        // Get data Info from input tensor
        Tensor* input_tensor = node->GetInputTensor(0);
        TShape& inShape = input_tensor->GetShape();
        float* input = ( float* )get_tensor_mem(input_tensor);
        std::vector<int> inDims = inShape.GetDim();
        int iDataH = inDims[2];
        int iDataW = inDims[3];
        int iDataC = inDims[1];
        int iDataN = inDims[0];

        // Get data info from output tensor
        Tensor* output_tensor = node->GetOutputTensor(0);
        float* output = ( float* )get_tensor_mem(output_tensor);

        // Get param infor from Crop param
        Power* pr_op = dynamic_cast<Power*>(node->GetOp());
        PowerParam* param_ = pr_op->GetParam();

        for(int n = 0; n < iDataN; n++)
        {
            for(int c = 0; c < iDataC; c++)
            {
                for(int h = 0; h < iDataH; h++)
                {
                    for(int w = 0; w < iDataW; w++)
                    {
                        int size = n * iDataC * iDataH * iDataW + c * iDataH * iDataW + h * iDataW + w;
                        output[size] = pow((param_->shift + param_->scale * input[size]), param_->power);
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

    PowerOps* ops = new PowerOps();

    return ops;
}
}    // namespace PowerImpl

void RegisterPower_NodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "Power", PowerImpl::SelectFunc, 1000);
}
}    // namespace TEngine
