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
 * Author: jinmingzhang@openailab.com
 */

#include <iostream>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "graph.hpp"
#include "operator/topk.hpp"

#include "kernel/topk/ref_topk_kernel.h"
namespace TEngine {
namespace RefTopkOps {
const int default_prio = 1500;
struct RefTopk : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    RefTopk()
    {
        kernel_run = nullptr;
        InitRegistry();
    }

    struct topk_param op_param;
    ref_topk_t kernel_run;
    void** output_data;
    KernelRegistry<ref_topk_t> kernel_registry;
};

void RefTopk::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_topk_t )ref_topk_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    // kernel_registry.Register(( ref_topk_t )ref_topk_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

}


bool RefTopk::Prerun(Node* node)
{

    int out_nums = node->GetOutputNum();
    output_data = new void*[out_nums];

    int layout = exec_attr->graph_layout;
    TopK* topk_op = dynamic_cast<TopK*>(node->GetOp());
    TopKParam* param_ = topk_op->GetParam();

    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
    auto in_dims = input_tensor->GetShape().GetDim();
    int dim_dize = in_dims.size();
    int num_rows = 1;
    for(int i = 0; i < dim_dize - 1; ++i)
    {
        num_rows *= in_dims.at(i);
    }

    op_param.k = param_->k;
    op_param.row_size = in_dims.back();
    op_param.num_rows = num_rows;

    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefTopk::Run(Node* node)
{
    Tensor* i_tensor = node->GetInputTensor(0);
    void* input = get_tensor_mem(i_tensor);

    auto dims = i_tensor->GetShape().GetDim();
    for(int ii = 0; ii < 2; ++ii)
    {
        Tensor* o_tensor = node->GetOutputTensor(ii);
        output_data[ii] = get_tensor_mem(o_tensor);
    }
 
    int ret = kernel_run(input, output_data[0], ( int* )output_data[1], &op_param);
    if(ret < 0)
        return false;


    return true;
}

bool RefTopk::Postrun(Node* node)
{
    delete[] output_data;
    return true;
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefTopk* ops = new RefTopk();

    LOG_DEBUG() << "Ref Topk is selected\n";

    return ops;
}

}    // end namespace RefTopkOps

void RegisterRefTopKOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "TopK", RefTopkOps::SelectFunc,
                                                  RefTopkOps::default_prio);
}
}    // namespace TEngine
