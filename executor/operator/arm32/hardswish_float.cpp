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
#include "operator/hardswish.hpp"
#include "data_type.hpp"
#include "neon_mathfun.h"
#include "arm_neon.h"
#include "compiler_fp16.h"

namespace TEngine {

namespace HardswishFP32Impl32 {

const int default_prio = 300;

struct HardswishOps : public NodeOps
{
    HardswishOps()
    {
        name_ = "arm_hardswish_fp32";
    }
    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        Hardswish* Hardswish_op = dynamic_cast<Hardswish*>(node->GetOp());
        HardswishParam* param_ = Hardswish_op->GetParam();
        float alpha = param_->alpha;
        float beta = param_->beta;
        float lower = -beta / alpha;
        float upper = (1.f / alpha) + lower;
        
        int elem_num = input_tensor->GetShape().GetSize();

        float* data = ( float* )get_tensor_mem(input_tensor);
        float* out_data = ( float* )get_tensor_mem(output_tensor);

        float32x4_t _zero = vdupq_n_f32(0.f);
        float32x4_t _one = vdupq_n_f32(1.f);
        for(int i = 0; i < (elem_num & -4); i += 4)
        {
            float32x4_t _p = vld1q_f32(data+i);
            float32x4_t _ans = vdupq_n_f32(beta);
            _ans = vmlaq_n_f32(_ans, _p, alpha);
            _ans = vmaxq_f32(_ans, _zero);
            _ans = vminq_f32(_ans, _one);
            _ans = vmulq_f32(_ans, _p);
            vst1q_f32(out_data+i, _ans);
        }
        for(int i = elem_num & ~3; i < elem_num; i++)
        {
            if (data[i] < lower)
                out_data[i] = 0.f;
            else if (data[i] > upper) out_data[i] = data[i];
            else
                out_data[i] = data[i] * (data[i] * alpha + beta);
        }
/*      
        for(int i = 0; i < elem_num; i++)
        {
            if (data[i] < lower)
                out_data[i] = 0.f;
            else if (data[i] > upper) out_data[i] = data[i];
            else
                out_data[i] = data[i] * (data[i] * alpha + beta);
        }
*/

        return true;

    }
};

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(data_type!=TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    HardswishOps* ops = new HardswishOps();

    return ops;
}

}    // namespace HardswishFP32Impl32

using namespace HardswishFP32Impl32;

void RegisterHardswishFP32(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm32", "Hardswish", HardswishFP32Impl32::SelectFunc, HardswishFP32Impl32::default_prio);
}

}    // namespace TEngine
