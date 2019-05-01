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
#include "operator/reorg.hpp"

namespace TEngine {

namespace ReorgImpl {

struct ReorgOps : public NodeOps
{
    void reorg(float* x, int w, int h, int c, int batch, int stride, float* out)
    {
        int b, i, j, k;
        int out_c = c / (stride * stride);

        for(b = 0; b < batch; ++b)
        {
            for(k = 0; k < c; ++k)
            {
                for(j = 0; j < h; ++j)
                {
                    for(i = 0; i < w; ++i)
                    {
                        int in_index = i + w * (j + h * (k + c * b));
                        int c2 = k % out_c;
                        int offset = k / out_c;
                        int w2 = i * stride + offset % stride;
                        int h2 = j * stride + offset / stride;
                        int out_index = w2 + w * stride * (h2 + h * stride * (c2 + out_c * b));
                        out[in_index] = x[out_index];
                    }
                }
            }
        }
    }

    bool Run(Node* node)
    {
        const Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        const std::vector<int>& dims = input_tensor->GetShape().GetDim();

        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);
        Reorg* reorg_op = dynamic_cast<Reorg*>(node->GetOp());
        ReorgParam* param = reorg_op->GetParam();

        reorg(input, dims[3], dims[2], dims[1], dims[0], param->stride, output);

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

    ReorgOps* ops = new ReorgOps();

    return ops;
}

}    // namespace ReorgImpl

using namespace ReorgImpl;

void RegisterReorgNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "Reorg", ReorgImpl::SelectFunc, 1000);
}

}    // namespace TEngine
