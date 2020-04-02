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

#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "math.h"
#include <sys/time.h>
#include <arm_neon.h>

namespace TEngine {

namespace SigmoidFP32Impl32 {

const int default_prio = 300;

float exp10_f32(float x)
{
    x = 1.0 + x * 0.0009765625f;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    return x;
}

inline float32x4_t vexpq10_f32(float32x4_t x)
{
    x = vmlaq_n_f32(vdupq_n_f32(1.0f), x, 0.0009765625f);    // n = 10
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    return x;
}

struct SigmoidOps : public MTNodeOps
{
    bool OnBind(Node* node)
    {
        inplace_t io_map;

        io_map[0] = 0;

        node->SetAttr(ATTR_INPLACE, io_map);
        return true;
    }

    bool Prerun(Node* node)
    {
        return true;
    }

    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        const TShape& shape = input_tensor->GetShape();
        int elem_num = shape.GetSize();
        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);
        /// void* data = get_tensor_mem(input_tensor);
        /// int ret = kernel_run(data, elem_num, &op_param);
#define SIGMOID_MAX(a, b) ((a) > (b) ? (a) : (b))
#define SIGMOID_MIN(a, b) ((a) < (b) ? (a) : (b))
        /*
        for(int i = 0; i < elem_num; i++)
        {
            input[i] = SIGMOID_MIN(input[i], 30.0f);
            input[i] = SIGMOID_MAX(input[i], -30.0f);
            std::cout<< input[i] << " ";
            float tmp_exp = exp8mul(-input[i]);
            output[i] = 1 / (1 + tmp_exp);
            std::cout<< tmp_exp << " " <<output[i] << "\n";
        }
        */

        float32x4_t min = vdupq_n_f32(-30.0f);
        float32x4_t max = vdupq_n_f32(30.0f);
        float32x4_t tmp_vec = vdupq_n_f32(1);
        for(int i = 0; i < (elem_num & -4); i += 4)
        {
            float32x4_t _input = vld1q_f32(input + i);
            _input = vmaxq_f32(_input, min);
            _input = vminq_f32(_input, max);
            float32x4_t tmp_exp = vaddq_f32(tmp_vec, vexpq10_f32(vmulq_n_f32(_input, -1.0f)));
            /// float32x4_t tmp_exp = vaddq_f32(tmp_vec, vexpq_f32(vmulq_n_f32(_input, -1.0f)));
            float32x4_t out = vrecpeq_f32(tmp_exp);
            out = vmulq_f32(vrecpsq_f32(tmp_exp, out), out);
            out = vmulq_f32(vrecpsq_f32(tmp_exp, out), out);
            vst1q_f32(output, out);
            output += 4;
        }
        for(int i = elem_num & ~3; i < elem_num; i++)
        {
            input[i] = SIGMOID_MIN(input[i], 30.0f);
            input[i] = SIGMOID_MAX(input[i], -30.0f);
            float tmp_exp = exp10_f32(-input[i]);
            *output++ = 1 / (1 + tmp_exp);
        }

        return true;
    }
};

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    SigmoidOps* ops = new SigmoidOps();
    return ops;
}

}    // namespace SigmoidFP32Impl32
using namespace SigmoidFP32Impl32;

void RegisterSigmoidFP32(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm32", "Sigmoid", SigmoidFP32Impl32::SelectFunc,
                                                  SigmoidFP32Impl32::default_prio);
}

}    // namespace TEngine
