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
 * Copyright (c) 2019, Open AI Lab
 * Author: ddzhao@openailab.com
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
#include "operator/elu.hpp"
#include "data_type.hpp"
#include "neon_mathfun.h"
#include "arm_neon.h"
#include "compiler_fp16.h"

namespace TEngine {

namespace EluFP32Impl64 {

const int default_prio = 300;
void elu_kernel(int i, int id, void* data, const float* input, float* output, float alpha)
{
    int elem_num = ((int*)data)[0];
    float32x4_t _one = vdupq_n_f32(1.f);
    float32x4_t _zero = vdupq_n_f32(0.f);
    float32x4_t _alpha = vdupq_n_f32(alpha);
    const float* cur_input = input + id * elem_num;
    float* cur_output = output + id * elem_num;
    for(int i = 0; i < (elem_num & -4); i += 4)
    {
        float32x4_t _p = vld1q_f32(cur_input);
        uint32x4_t _lemask = vcleq_f32(_p, _zero);

        float32x4_t _nps = exp_ps(_p);
        _nps = vsubq_f32(_nps, _one);
        _nps = vmulq_f32(_nps, _alpha);

        _p = vbslq_f32(_lemask, _nps, _p);
        vst1q_f32(cur_output, _p);
        cur_input += 4;
        cur_output += 4;
    }
    for(int i = elem_num & ~3; i < elem_num; i++)
    {
        if (*cur_input < 0.f)
            *cur_output = (exp(*cur_input) - 1.f) * alpha;
        else
            *cur_output = *cur_input;
        cur_input ++;
        cur_output ++;
    }
}
struct EluOps : public NodeOps
{
    EluOps()
    {
        name_ = "arm_elu_fp32";
    }

    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        Elu* Elu_op = dynamic_cast<Elu*>(node->GetOp());
        EluParam* param_ = Elu_op->GetParam();
        float alpha = param_->alpha;

        int elem_num = input_tensor->GetShape().GetSize();

        float* data = ( float* )get_tensor_mem(input_tensor);
        float* out_data = ( float* )get_tensor_mem(output_tensor);

        int cpu_number = cpu_info->GetCPUNumber();
        int block = elem_num >> 8;
        block = block > 0 ? block : 1;
        int num_task = cpu_number < block ? cpu_number : block;
        int step = elem_num / num_task;

        if(num_task == 1)
            elu_kernel( 0, 0, &step, data, out_data, alpha);
        else
        {
            MULTI_THREAD_START(num_task, step, p_id, p_param)
                    elu_kernel(0, p_id, p_param, data, out_data, alpha);
            MULTI_THREAD_END();
        }
        if(num_task * step != elem_num)
        {
            int offset = num_task * step;
            int remain_num = elem_num - offset;
            elu_kernel(0, 0, &remain_num, data + offset, out_data + offset, alpha);
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

    EluOps* ops = new EluOps();

    return ops;
}

}    // namespace EluFP32Impl64

using namespace EluFP32Impl64;

void RegisterEluFP32(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Elu", EluFP32Impl64::SelectFunc, EluFP32Impl64::default_prio);
}

}    // namespace TEngine
