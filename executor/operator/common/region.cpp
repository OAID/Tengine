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
#include <math.h>
#include <stdlib.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/region.hpp"

int entry_index(int batch, int location, int entry, int hw, int chw, int classes)
{
    int coords = 4;
    int n = location / hw;
    int loc = location % hw;
    return batch * chw + n * hw * (coords + classes + 1) + entry * hw + loc;
}
static inline float logistic_activate(float x)
{
    return 1. / (1. + exp(-x));
}

void logit_activate_array(float* x, const int n)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        x[i] = logistic_activate(x[i]);
    }
}

void softmax(float* input, int n, int stride, float* output)
{
    int i;
    float sum = 0;
    float largest = input[0];
    for(i = 0; i < n; ++i)
    {
        if(input[i * stride] > largest)
            largest = input[i * stride];
    }
    for(i = 0; i < n; ++i)
    {
        float e = exp(input[i * stride] - largest);
        sum += e;
        output[i * stride] = e;
    }
    for(i = 0; i < n; ++i)
    {
        output[i * stride] /= sum;
    }
}

void softmax_cpu(float* input, int n, int batch, int batch_offset, int groups, int stride, float* output)
{
    int g, b;
    for(b = 0; b < batch; ++b)
    {
        for(g = 0; g < groups; ++g)
        {
            softmax(input + b * batch_offset + g, n, stride, output + b * batch_offset + g);
        }
    }
}

namespace TEngine {

namespace RegionImpl {

struct RegionOps : public NodeOps
{
    bool Run(Node* node)
    {
        const Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);

        const std::vector<int>& dims = input_tensor->GetShape().GetDim();

        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);

        Region* reorg_op = dynamic_cast<Region*>(node->GetOp());
        RegionParam* param = reorg_op->GetParam();

        int hw = dims[2] * dims[3];
        int chw = dims[1] * hw;
        int nchw = dims[0] * chw;
        int num_box = param->num_box;
        int num_class = param->num_classes;
        int coords = param->coords;
        memcpy(output, input, nchw * sizeof(float));

        for(int b = 0; b < dims[0]; b++)
        {
            for(int n = 0; n < num_box; n++)
            {
                int index = entry_index(b, n * hw, 0, hw, chw, num_class);
                logit_activate_array(output + index, 2 * hw);
                index = entry_index(b, n * hw, coords, hw, chw, num_class);
                logit_activate_array(output + index, hw);
                index = entry_index(b, n * hw, coords + 1, hw, chw, num_class);
            }
        }

        int index = entry_index(0, 0, coords + 1, hw, chw, num_class);
        softmax_cpu(input + index, num_class, dims[0] * num_box, chw / num_box, hw, hw, output + index);

        return true;
    }
};

}    // namespace RegionImpl

using namespace RegionImpl;

void RegisterRegionNodeExec(void)
{
    RegionOps* ops = new RegionOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "Region", ops);
}

}    // namespace TEngine
