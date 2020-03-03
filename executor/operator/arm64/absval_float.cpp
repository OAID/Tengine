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
#include <operator/absval.hpp>

namespace TEngine {

namespace AbsvalFP32Impl64 {

const int default_prio = 300;

struct AbsvalOps : public MTNodeOps
{
    AbsvalOps()
    {
        name_ = "arm_abs_fp32";
    }

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
        const TShape& shape = input_tensor->GetShape();
        
        float* data = ( float* )get_tensor_mem(input_tensor);
        
        int channel_num = shape.GetC();
        int batch_number = shape.GetN();
        int channel_size = shape.GetW() * shape.GetH();
        
        for(int c = 0; c < channel_num * batch_number; c++)
        {
            for(int i = 0; i < (channel_size & -4); i += 4)
            {
                float32x4_t _p = vld1q_f32(data);
                _p = vabsq_f32(_p);
                vst1q_f32(data, _p);

                data += 4;
            }
            for(int i = channel_size & ~3; i < channel_size; i++)
            {
                if (*data < 0)
                    *data = -*data;
                data++;
            }
        }
        
        /*
        int elem_num = shape.GetSize();
        for(int i = 0; i < (elem_num & -4); i += 4)
        {
	        float32x4_t _p = vld1q_f32(data);
            _p = vabsq_f32(_p);
            vst1q_f32(data, _p);

            data += 4;
        }
        for(int i = elem_num & ~3; i < elem_num; i++)
        {
	        if (*data < 0)
                *data = -*data;
            data++;
        }
        */

        return true;
    }
};

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;
    AbsvalOps* ops = new AbsvalOps();

    return ops;
}

}    // namespace AbsvalFP32Impl64
using namespace AbsvalFP32Impl64;
void RegisterAbsvalFP32(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Absval", AbsvalFP32Impl64::SelectFunc,
                                                  AbsvalFP32Impl64::default_prio);
}

}    // namespace TEngine
