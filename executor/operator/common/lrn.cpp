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
 * Author: haitao@openailab.com
 */
#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>
#include <cmath>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/lrn.hpp"

namespace TEngine {

namespace LRNImpl {

void lrn_kernel(int i, int id, void* data, const float* input, float* output, float* square,
        int h, int w, int channel, int local_size, float alpha_over_size, float beta)
{
    int step = ((int*)data)[0];
    int channel_size = h * w;
    float* accum_square = ( float* )(std::malloc(channel_size * sizeof(float)));

    int start_c = step * id;
    int end_c = step * id + step;
    for(int c = start_c; c < end_c; c++)
    {
        int c_start = c - local_size / 2;
        int c_end = c + local_size / 2;

        std::memset(accum_square, 0x0, channel_size * sizeof(float));

        for(int l = c_start; l <= c_end; l++)
        {
            if(l < 0 || l >= channel)
                continue;

            for(int n = 0; n < channel_size; n++)
            {
                accum_square[n] += square[l * channel_size + n];
            }
        }
        /* get the output */
        const float* cur_input = input + c * channel_size;
        float* cur_output = output + c * channel_size;
        for(int n = 0; n < channel_size; n++)
        {
            *cur_output++ = *cur_input++ * std::pow(1.0f + alpha_over_size * accum_square[n], -beta);
        }

    }

    std::free(accum_square);
}
struct LRNOps : public NodeOps
{
    LRNOps()
    {
        name_ = "com_lrn_fp32";
    }
    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);

        LRN* lrn_op = dynamic_cast<LRN*>(node->GetOp());
        LRNParam* param = lrn_op->GetParam();

        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);

        const TShape& shape = input_tensor->GetShape();
        const std::vector<int>& dims = shape.GetDim();

        int n = dims[0];
        int c = dims[1];
        int h = dims[2];
        int w = dims[3];

        int img_size = c * h * w;
        float alpha = param->alpha;
        float beta = param->beta;
        float bias = param->k;
        int local_size = param->local_size;
        float alpha_over_size = alpha / local_size;

        float* square = ( float* )(std::malloc(img_size * sizeof(float)));
        int cpu_number = cpu_info->GetCPUNumber();
        int num_task = c < cpu_number ? c : cpu_number;
        int step = c / num_task;

        for(int i = 0; i < n; i++)
        {
            /* get square value */

            float* in_base = input + i * img_size;
            float* out_base = output + i * img_size;

            if(param->norm_region != LRN_ACROSS_CHANNELS)
            {
                LOG_ERROR()<<"LRN Only support ACORSS_CHANNEL\n";
                return false;
            }
            else
            {
                for(int j = 0; j < img_size; j++)
                    square[j] = in_base[j] * in_base[j] + bias;
            }
            if(num_task == 1)
            {
                lrn_kernel(0, 0, &c, in_base, out_base, square, h, w, c, local_size, alpha_over_size, beta);
            }
            else
            {
                MULTI_THREAD_START(num_task, step, id, param)
                    lrn_kernel(0, id, param, in_base, out_base, square, h, w, c, local_size, alpha_over_size, beta);
                MULTI_THREAD_END();
            }
            if(num_task * step != c)
            {
                int offset = num_task * step;
                int remain_num = c - offset;
                in_base += offset * h * w;
                out_base += offset * h * w;
                lrn_kernel(0, 0, &remain_num, in_base, out_base, square, h, w, c, local_size, alpha_over_size, beta);
            }
        }

        std::free(square);
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

    LRNOps* ops = new LRNOps();

    return ops;
}

}    // namespace LRNImpl

using namespace LRNImpl;

void RegisterLRN_NodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "LRN", LRNImpl::SelectFunc, 1000);
}

}    // namespace TEngine
