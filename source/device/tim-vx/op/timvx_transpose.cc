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
 * Copyright (c) 2021, Open AI Lab
 * Author: hhchen@openailab.com
 */

#include "timvx_executor.hpp"

extern "C"
{
#include "operator/op.h"
#include "transpose_param.h"
}


bool VXEngine::AddTransposeNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct transpose_param* param = (struct transpose_param*)ir_node->op.param_mem;

    std::vector<uint32_t> perm(output_tensor->dim_num);
    if (output_tensor->dim_num == 5)
    {
        std::map<uint32_t, uint32_t> lookup;
        for (int j = 0; j < 5; j++)
        {
            lookup[param->tr_shape[j]] = j;
        }
        perm[0] = param->tr_shape[lookup[(uint32_t )(4 - param->tr_shape[4])]];
        perm[1] = param->tr_shape[lookup[(uint32_t )(4 - param->tr_shape[3])]];
        perm[2] = param->tr_shape[lookup[(uint32_t )(4 - param->tr_shape[2])]];
        perm[3] = param->tr_shape[lookup[(uint32_t )(4 - param->tr_shape[1])]];
        perm[4] = param->tr_shape[lookup[(uint32_t )(4 - param->tr_shape[0])]];
    }
    else if (output_tensor->dim_num == 4)
    {
        std::map<uint32_t, uint32_t> lookup;
        for (int j = 0; j < 4; j++)
        {
            lookup[param->tr_shape[j]] = j;
        }
        perm[0] = param->tr_shape[lookup[(uint32_t )(3 - param->tr_shape[3])]];
        perm[1] = param->tr_shape[lookup[(uint32_t )(3 - param->tr_shape[2])]];
        perm[2] = param->tr_shape[lookup[(uint32_t )(3 - param->tr_shape[1])]];
        perm[3] = param->tr_shape[lookup[(uint32_t )(3 - param->tr_shape[0])]];
    }
    else if (output_tensor->dim_num == 3)
    {
        perm[0] = (uint32_t )param->tr_shape[2];
        perm[1] = (uint32_t )param->tr_shape[1];
        perm[2] = (uint32_t )param->tr_shape[0];
    }

    auto transpose = graph->CreateOperation<tim::vx::ops::Transpose>(perm);
    vx_node_map[ir_node->index] = transpose;

    (*transpose)
        .BindInputs({ this->vx_tensor_map[input_tensor->index] })
        .BindOutputs({ this->vx_tensor_map[output_tensor->index] });

    return true;
}

