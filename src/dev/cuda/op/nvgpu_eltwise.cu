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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: hhchen@openailab.com
 */


#include "cuda_executor.hpp"

extern "C"
{
#include "tengine_op.h"
#include "eltwise_param.h"
}

__global__ void eltwise_sum(float *y, float *x0, float *x1, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        y[idx] = x0[idx] + x1[idx];
    }
}

void eltwisesum_gpu_kernel(struct ir_graph* ir_graph, struct ir_node* ir_node, dict_uint2voidx  gpu_addr_map)
{
    struct ir_tensor* input_tensor0 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* input_tensor1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    eltwise_param* param = (eltwise_param*)ir_node->op.param_mem;

    /* init grid and block */
    int bs = 1024;
    int s = ceil((output_tensor->elem_num + bs - 1.) / bs);
    dim3 grid = dim3(s);

    switch (param->type)
    {
        case ELT_SUM:
            eltwise_sum<<<grid, bs>>>((float*)gpu_addr_map[output_tensor->idx], (float*)gpu_addr_map[input_tensor0->idx], (float*)gpu_addr_map[input_tensor1->idx], output_tensor->elem_num);
            break;
        default:
            break;
    }
}

void CUDAEngine::AddEltwiseNode(struct ir_graph* ir_graph, struct ir_node* ir_node)
{
    TLOG_INFO("Tengine GPU: Support OP(%d) OP_RELU.\n", ir_node->idx);
    eltwisesum_gpu_kernel(ir_graph, ir_node, this->gpu_addr_map);
    this->ops.push_back(std::bind(&eltwisesum_gpu_kernel, ir_graph, ir_node, this->gpu_addr_map));
}
