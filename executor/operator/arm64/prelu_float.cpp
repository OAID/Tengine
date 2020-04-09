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

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
namespace TEngine {

namespace PReluFP32Impl64 {

const int default_prio = 300;

struct PreluOps : public NodeOps
{
    PreluOps()
    {
        name_ = "arm_prelu_fp32";
    }
    bool OnBind(Node* node);
    bool Run(Node* node);
    bool Direct_Prelu(float *input,float *output,float* slope,int channel, int height ,int width);

};
bool PreluOps::OnBind(Node* node)
{
    // set the inplace feature
    inplace_t io_map;
    io_map[0] = 0;
    node->SetAttr(ATTR_INPLACE, io_map);

    return true;
}

void prelu_kernel(int i, int id, void* data, const float *input,float *output,float* slope,int channel_size)
{
    int step = ((int*)data)[0];
    float32x4_t _zero = vdupq_n_f32(0.f);
    for(int c = 0; c < step; c++)
    {
        int cur_c = id * step + c;
        const float* cur_input = input + cur_c * channel_size;
        float* cur_output = output +  cur_c * channel_size;
        float32x4_t _slope = vdupq_n_f32(slope[cur_c]);
        for(int l=0; l< (channel_size & -4); l += 4)
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
        for(int l = channel_size & ~3; l < channel_size; l++)
        {
            *cur_output = MAX(cur_input[0], 0.f) + slope[cur_c] * MIN(cur_input[0], 0.f);
            cur_input ++;
        }
    }
}

bool PreluOps::Run(Node* node)
{
    // inplace implement
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const TShape& shape = input_tensor->GetShape();
    const std::vector<int> dims = shape.GetDim();
    int channel_size = dims[2] * dims[3];
    int channel_num = dims[1];
    int img_size = channel_size * channel_num;
    const float* input = ( float* )get_tensor_mem(input_tensor);
    float* output = ( float* )get_tensor_mem(output_tensor);
    const Tensor* slope_tensor = node->GetInputTensor(1);
    float* slope = ( float* )get_tensor_mem(slope_tensor);

    int cpu_number = cpu_info->GetCPUNumber();
    int block = channel_num ;
    block = block > 0 ? block : 1;
    int num_task = cpu_number < block ? cpu_number : block;
    int step = channel_num / num_task;

    for(int n = 0; n < dims[0]; n++)
    {
        const float *input_data = input + n * img_size;
        float *out_data = output + n * img_size;

        if(num_task == 1)
            prelu_kernel( 0, 0, &step, input_data, out_data, slope, channel_size);
        else
        {
            MULTI_THREAD_START(num_task, step, p_id, p_param)
                    prelu_kernel(0, p_id, p_param, input_data, out_data, slope, channel_size);
            MULTI_THREAD_END();
        }
        if(num_task * step != channel_num)
        {
            int offset = num_task * step;
            int remain_num = channel_num - offset;
            input_data += offset * channel_size;
            out_data += offset * channel_size;
            prelu_kernel(0, 0, &remain_num, input_data, out_data, slope + offset, channel_size);
        }
    }
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    PreluOps* ops = new PreluOps();

    return ops;
}

}    // namespace PreluImpl

using namespace PReluFP32Impl64;

void RegisterPReluFP32(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm64", "PReLU", PReluFP32Impl64::SelectFunc, PReluFP32Impl64::default_prio);
}

}    // namespace TEngine
