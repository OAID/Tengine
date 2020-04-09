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

#include <functional>
#include <cstring>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include <arm_neon.h>
#include "operator/relu.hpp"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
namespace TEngine {

namespace ReluFP32Impl32 {

const int default_prio = 300;

inline bool relu_kernel(const int i, const int id, const void* data, const float* input,float* output, const float slope)
{
    float32x4_t _zero = vdupq_n_f32(0.f);
    int step = ((int*)data)[0];
    const float* cur_input = input + id * step;
    float* cur_output = output + id * step;
    if(slope == 0)
    {
        for(int l=0; l< (step & -4); l += 4)
        {
            float32x4_t _p = vld1q_f32(cur_input);
            _p = vmaxq_f32(_p, _zero);
            vst1q_f32(cur_output, _p);
            cur_input += 4;
            cur_output += 4;

        }
        for(int i = step & ~3; i < step; i++)
        {
            *cur_output++ = MAX(*cur_input++, 0.f);
        }
    }
    else
    {
        float32x4_t _slope = vdupq_n_f32(slope);
        for(int l=0; l< (step & -4); l += 4)
        {
            float32x4_t _p = vld1q_f32(cur_input);
            // ri = ai <= bi ? 1...1:0...0
            uint32x4_t _lemask = vcleq_f32(_p, _zero);
            float32x4_t _ps = vmulq_f32(_p, _slope);
            // bitwise select 
            _p = vbslq_f32(_lemask, _ps, _p);
            vst1q_f32(cur_output, _p);
            cur_input += 4;
            cur_output += 4;
        }
        for(int i = step & ~3; i < step; i++)
        {
            *cur_output++ = MAX(cur_input[0], 0.f) + slope * MIN(cur_input[0], 0.f);
            cur_input ++;
        }
    }
    return true;
}

struct ReluOps : public NodeOps
{
    ReluOps()
    {
        name_ = "arm_relu_fp32";
    }

    bool OnBind(Node* node)
    {
        // set the inplace feature
        inplace_t io_map;

        io_map[0] = 0;

        node->SetAttr(ATTR_INPLACE, io_map);

        return true;
    }

    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        const TShape& shape = output_tensor->GetShape();
        int elem_num = shape.GetSize();
        ReLu* relu_op = dynamic_cast<ReLu*>(node->GetOp());
        ReLuParam* param = relu_op->GetParam();
        float* data = ( float* )get_tensor_mem(input_tensor);
        float* out_data = ( float* )get_tensor_mem(output_tensor);
        float negativeslope = param->negative_slope;

        int cpu_number = cpu_info->GetCPUNumber();
        int block = elem_num >> 8;
        block = block > 0 ? block : 1;
        int num_task = cpu_number < block ? cpu_number : block;
        int step = elem_num / num_task;

        if(num_task == 1)
            relu_kernel( 0, 0, &step, data, out_data, negativeslope);
        else
        {
            MULTI_THREAD_START(num_task, step, p_id, p_param)
                    relu_kernel(0, p_id, p_param, data, out_data, negativeslope);
            MULTI_THREAD_END();
        }
        if(num_task * step != elem_num)
        {
            int offset = num_task * step;
            int remain_num = elem_num - offset;
            relu_kernel(0, 0, &remain_num, data + offset, out_data + offset, negativeslope);
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

    ReluOps* ops = new ReluOps();

    return ops;
}

}    // namespace PreluImpl

using namespace ReluFP32Impl32;

void RegisterReluFP32(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm32", "ReLu", ReluFP32Impl32::SelectFunc, ReluFP32Impl32::default_prio);
}

}    // namespace TEngine
