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
 * Author: zpluo@openailab.com
 */

#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"

#include "operator/reducel2.hpp"
#include "kernel/reducel2/ref_reducel2_kernel.h"

namespace TEngine {

namespace RefReduceL2Ops {

struct RefReduceL2 : public MTNodeOps
{
    bool Prerun(Node* node) override;

    bool Run(Node* node) override;
    void InitRegistry(void);

    ref_reducel2_t kernel_run;
    reducel2_param param;

    KernelRegistry<ref_reducel2_t> kernel_registry;
    RefReduceL2(void)
    {
        InitRegistry();
    }
};

bool RefReduceL2::Prerun(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);

    int layout = exec_attr->graph_layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, input_tensor->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefReduceL2::Run(Node* node)
{
    ReduceL2* reducel2_op = dynamic_cast<ReduceL2*>(node->GetOp());
    ReduceL2Param* op_param = reducel2_op->GetParam();

    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* out_tensor = node->GetOutputTensor(0);

    const TShape& i_shape = input_tensor->GetShape();
    auto in_dim = i_shape.GetDim();
    void* in_data = get_tensor_mem(input_tensor);
    void* out_data = get_tensor_mem(out_tensor);

    if(op_param->axis < 0)
        param.axis = op_param->axis + (int)in_dim.size();
    else
        param.axis = op_param->axis;   
 
    for(unsigned int i = 0; i < in_dim.size();i++)
    {
        param.dims[i] = in_dim[i];
    }
    for(unsigned int i = in_dim.size();i < 4;i++)
    {
        param.dims[i] = 1;
    }
    int ret = kernel_run(in_data, out_data, &param);
    
    if(ret < 0)
        return false;
    else
        return true;
}

void RefReduceL2::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_reducel2_t )ref_reducel2_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefReduceL2* ops = new RefReduceL2();

    LOG_DEBUG() << "ReduceL2 RefOp is selected\n";

    return ops;
}
}
using namespace RefReduceL2Ops;
void RegisterRefReduceL2Ops(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "ReduceL2", RefReduceL2Ops::SelectFunc, 1000);
}
}    // namespace TEngine
