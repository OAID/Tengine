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
#include "eltwise_param.h"

#include "graph/tensor.h"
#include "operator/op.h"
#include "utility/log.h"
}

__global__ void eltwise_sum(float *y, float *x0, float *x1, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        y[idx] = x0[idx] + x1[idx];
    }
}

void eltwisesum_gpu_kernel(struct graph* ir_graph, struct node* ir_node, dict_uint2voidx  gpu_addr_map)
{
    struct tensor* input_tensor0 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* input_tensor1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    eltwise_param* param = (eltwise_param*)ir_node->op.param_mem;

    /* init grid and block */
    int bs = 1024;
    int s = ceil((output_tensor->elem_num + bs - 1.) / bs);
    dim3 grid = dim3(s);

    switch (param->type)
    {
        case ELT_SUM:
            eltwise_sum<<<grid, bs>>>((float*)gpu_addr_map[output_tensor->index], (float*)gpu_addr_map[input_tensor0->index], (float*)gpu_addr_map[input_tensor1->index], output_tensor->elem_num);
            break;
        default:
            break;
    }
}

void CUDAEngine::AddEltwiseNode(struct graph* ir_graph, struct node* ir_node)
{
    TLOG_INFO("Tengine GPU: Support OP(%d) OP_ELTWISE.\n", ir_node->index);
    eltwisesum_gpu_kernel(ir_graph, ir_node, this->gpu_addr_map);
    this->ops.push_back(std::bind(&eltwisesum_gpu_kernel, ir_graph, ir_node, this->gpu_addr_map));
}
