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
#include "concat_param.h"

#include "graph/tensor.h"
#include "operator/op.h"
#include "utility/log.h"
}

void concat_gpu_kernel(struct graph* ir_graph, struct node* ir_node, dict_uint2voidx  gpu_addr_map)
{
    concat_param* param = (concat_param*)ir_node->op.param_mem;
    int axis = param->axis;

    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    int elem_num_axis_sum = output_tensor->dims[0];
    for (int i = 1; i < axis; i++)
    {
        elem_num_axis_sum *= output_tensor->dims[i];
    }
    elem_num_axis_sum = output_tensor->elem_num / elem_num_axis_sum;

    struct tensor* input_tensor;
    /* init grid and block */
    int sum_size = 0;
    for (int i = 0; i < ir_node->input_num; i++)
    {
        input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);
        int bs = 1024;
        int s = ceil((input_tensor->elem_num + bs - 1.) / bs);
        dim3 grid = dim3(s);

        int elem_num_peraxis = 1;
        for (int i = 3; i > axis; i--)
        {
            elem_num_peraxis *= input_tensor->dims[i];
        }
        int elem_num_axis = elem_num_peraxis * input_tensor->dims[axis];

        for (int p = 0; p < input_tensor->dims[axis-1]; p++)
        {
            cudaMemcpy((float*)gpu_addr_map[output_tensor->index] + sum_size + p * elem_num_axis_sum, (float*)gpu_addr_map[input_tensor->index] + p * elem_num_axis,
                       elem_num_axis * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        int size = 1;
        for (int j = 0; j < axis; j++)
        {
            size *= input_tensor->dims[j];
        }
        size = input_tensor->elem_num / size;
        sum_size += size;
    }
}

void CUDAEngine::AddConcatNode(struct graph* ir_graph, struct node* ir_node)
{
    TLOG_INFO("Tengine GPU: Support OP(%d) OP_CONCAT.\n", ir_node->index);
    concat_gpu_kernel(ir_graph, ir_node, this->gpu_addr_map);
    this->ops.push_back(std::bind(&concat_gpu_kernel, ir_graph, ir_node, this->gpu_addr_map));
}
