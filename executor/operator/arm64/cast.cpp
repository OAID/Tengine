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
#include "operator/cast.hpp"
#include "data_type.hpp"
#include "neon_mathfun.h"
#include "arm_neon.h"
#include "compiler_fp16.h"

namespace TEngine {

namespace CastFP32Impl64 {

const int default_prio = 300;

struct CastOps : public NodeOps
{
    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        Cast* Cast_op = dynamic_cast<Cast*>(node->GetOp());
        CastParam* param_ = Cast_op->GetParam();
        int type_from = param_->type_from;
        int type_to = param_->type_to;

        int elem_num = input_tensor->GetShape().GetSize();

        if (type_from == 1 && type_to == 2)
        {
            float* data = ( float* )get_tensor_mem(input_tensor);
            __fp16* out_data = ( __fp16* )get_tensor_mem(output_tensor);
            for(int i = 0; i < (elem_num & -4); i += 4)
            {
                float32x4_t x = vld1q_f32(data + i);
                float16x4_t _p = vcvt_f16_f32(x);
                vst1_f16(out_data + i, _p);
            }
            for(int i = elem_num & ~3; i < elem_num; i++)
            {
                out_data[i] = fp32_to_fp16(data[i]);
            }
            
        }

        if (type_from == 2 && type_to == 1)
        {
            __fp16* data = ( __fp16* )get_tensor_mem(input_tensor);
            float* out_data = ( float* )get_tensor_mem(output_tensor);
            for(int i = 0; i < (elem_num & -4); i += 4)
            {
                float16x4_t x = vld1_f16(data + i);
                float32x4_t _p = vcvt_f32_f16(x);
                vst1q_f32(out_data + i, _p);
            }
            for(int i = elem_num & ~3; i < elem_num; i++)
            {
                out_data[i] = fp16_to_fp32(data[i]);
            }
        }

        return true;

    }
};

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    //Tensor* input = node->GetInputTensor(0);
    //const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    CastOps* ops = new CastOps();

    return ops;
}

}    // namespace CastFP32Impl64

using namespace CastFP32Impl64;

void RegisterCastFP32(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Cast", CastFP32Impl64::SelectFunc, CastFP32Impl64::default_prio);
}

}    // namespace TEngine
