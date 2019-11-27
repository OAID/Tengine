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
        int layout = exec_attr->graph_layout;
        int model_layout = exec_attr->model_layout;
        if(model_layout == layout)
            return true;

        if(model_layout == TENGINE_LAYOUT_NHWC && layout == TENGINE_LAYOUT_NCHW)
        {
            Tensor* input_tensor = node->GetInputTensor(0);
            Tensor* output_tensor = node->GetOutputTensor(0);
            TShape& shape = input_tensor->GetShape();
            int input_n = shape.GetN();
            int input_c = shape.GetC();
            int input_h = shape.GetH();
            int input_w = shape.GetW();
            int hw = input_h * input_w;
            if(hw == 1)
                return true;
            int out_size = input_n * input_c * hw;
            float* input_org = ( float* )get_tensor_mem(input_tensor);
            float* output = ( float* )malloc(out_size * sizeof(float));
            float* output_org = ( float* )get_tensor_mem(output_tensor);
            for(int n = 0; n < input_n; n++)
                for(int c = 0; c < input_c; c++)
                    for(int h = 0; h < input_h; h++)
                        for(int w = 0; w < input_w; w++)
                        {
                            int in_index = n * input_c * input_h * input_w + c * input_h * input_w + h * input_w + w;
                            int out_index = n * input_h * input_w * input_c + h * input_w * input_c + w * input_c + c;
                            output[out_index] = input_org[in_index];
                        }
            memcpy(output_org, output, out_size * sizeof(float));
            free(output);
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
