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
 * Author: bzhang@openailab.com
 */
#include <iostream>
#include <functional>
#include <stdlib.h>
#include "kernel_registry.hpp"
#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "tengine_errno.hpp"
#include "operator/transpose.hpp"
#include "kernel/transpose/ref_transpose_kernel.h"
#include <vector>
#include <cmath>

namespace TEngine {

namespace RefTransposeImpl {
// const int default_prio = 1500;
struct RefTransposeOps : public NodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);
    RefTransposeOps()
    {
        kernel_run = nullptr;
        InitRegistry();
    }
    struct ref_transpose_param op_param;
    ref_transpose_n_kernel_t kernel_run;
    KernelRegistry<ref_transpose_n_kernel_t> kernel_registry;
};

void RefTransposeOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_transpose_n_kernel_t )ref_transpose_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_transpose_n_kernel_t )ref_transpose_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

}

bool RefTransposeOps::Prerun(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    int data_type = input_tensor->GetDataType();
    int layout = exec_attr->graph_layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, data_type))
    {
        set_tengine_errno(ENOENT);
        return false;
    }
    return true;
}

bool RefTransposeOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);

    Tensor* output_tensor = node->GetOutputTensor(0);
    void* out_data = ( void* )get_tensor_mem(output_tensor);
    void* in_data = ( void* )get_tensor_mem(input_tensor);
    Transpose* transpose = dynamic_cast<Transpose*>(node->GetOp());
    TransposeParam* param_ = transpose->GetParam();
    int tr_size = param_->tr_shape.size();
    op_param.permute.clear();
    for(int i = 0; i < tr_size; i++){
        op_param.permute.push_back(param_->tr_shape[i]);
    }
    /*
    op_param.permute[0] = param_->dim_0;
    op_param.permute[1] = param_->dim_1;
    op_param.permute[2] = param_->dim_2;
    op_param.permute[3] = param_->dim_3;
    */
    std::vector<int>& in_dims = input_tensor->GetShape().GetDim();
    op_param.dims=in_dims.size();
    //printf("dims: %d \n", op_param.dims);
    //printf("in_dims : %d %d\n", (int)in_dims.size(), in_dims[0]);
    /*
    for(int i= 0; i < 4; i++){
        if(op_param.permute[i] == -2){
            op_param.dims++;
        }
    }
    */
    
    /*
    if(in_dims.size() == 4){
        for(int i = 0; i < 4; i++){
            op_param.in_dims[i] = in_dims[i];
        }
    } else if(in_dims.size() == 3){
        for(int i = 0; i < 3; i++){
            op_param.in_dims[i] = in_dims[i];
        }
        op_param.in_dims[3] = 1;
    } else if(in_dims.size() == 2){
        for(int i = 0; i < 2; i++){
            op_param.in_dims[i] = in_dims[i];
        }
        op_param.in_dims[2] = 1;
        op_param.in_dims[3] = 1;
    } 
    */
   op_param.in_dims.clear();
    for(int i = 0; i < (int)in_dims.size(); i++){
        op_param.in_dims.push_back(in_dims[i]);
    }

    int ret = kernel_run(in_data, out_data, &op_param);
    if(ret < 0)
        return false;

    return true;
}

bool RefTransposeOps::Postrun(Node* node)
{
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    RefTransposeOps* ops = new RefTransposeOps();
    return ops;
}

}    // namespace RefTransposeImpl

using namespace RefTransposeImpl;

void RegisterRefTransposeOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Transpose", RefTransposeImpl::SelectFunc, 1000);
}

}    // namespace TEngine
