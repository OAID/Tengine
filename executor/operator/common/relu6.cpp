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
 * Author: jingyou@openailab.com
 */
#include <iostream>
#include <functional>
#include <cstring>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "data_type.hpp"

namespace TEngine {

namespace ReLu6Impl {

struct ReLu6Ops : public NodeOps
{
    bool OnBind(Node* node) override
    {
        // set the inplace feature
        inplace_t io_map;

        io_map[0] = 0;

        node->SetAttr(ATTR_INPLACE, io_map);

        return true;
    }

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
    template <typename data_type> void kernel_run(void* data, int size)
    {
        data_type* out_data = ( data_type* )data;

        for(int i = 0; i < size; i++)
        {
            out_data[i] = MIN(MAX(out_data[i], 0), 6);
        }
    }

    bool Run(Node* node) override
    {
        // input tensor and output tensor is the same
        Tensor* input_tensor = node->GetInputTensor(0);
        int element_size = DataType::GetTypeSize(input_tensor->GetDataType());
        const TShape& shape = input_tensor->GetShape();
        int elem_num = shape.GetSize();
        void* data = get_tensor_mem(input_tensor);

        switch(element_size)
        {
            case 4:
                kernel_run<float>(data, elem_num);
                break;
#ifdef CONFIG_FLOAT16
            case 2:
                kernel_run<__fp16>(data, elem_num);
                break;
#endif
            case 1:
                kernel_run<char>(data, elem_num);
                break;
        }

        return true;
    }
};

}    // namespace ReLu6Impl

using namespace ReLu6Impl;

void RegisterReLu6NodeExec(void)
{
    ReLu6Ops* ops = new ReLu6Ops();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "ReLu6", ops);
}

}    // namespace TEngine
