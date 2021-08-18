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
#include "reduction_param.h"
}

bool VXEngine::AddReduceNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct reduction_param* param = (struct reduction_param*)ir_node->op.param_mem;

    int32_t reduceAxes = 0;
    if (input_tensor->dim_num == 5)
    {
        reduceAxes = 3;
    }
    else
    {
        if (param->dim_0 != -2)
            reduceAxes = 3;
        else if (param->dim_1 == -2)
            reduceAxes = 2;
        else if (param->dim_2 == -2)
            reduceAxes = 1;
        else
            reduceAxes = 0;
    }

    std::vector<int32_t> axis;
    axis.push_back(reduceAxes);

    switch(param->type)
    {
        case (0):
        {
            auto reduce = graph->CreateOperation<tim::vx::ops::ReduceSum>(axis, param->keepdim);
            vx_node_map[ir_node->index] = reduce;

            (*reduce)
                    .BindInputs({ this->vx_tensor_map[input_tensor->index] })
                    .BindOutputs({ this->vx_tensor_map[output_tensor->index] });

            break;
        }
        default:
            TLOG_ERR("This pad type is not supported yet: Pad_type(%d) Tensor_name(%s) tensor_index(%d) tensor_data_type(%d) .\n",param->type, input_tensor->name, input_tensor->index, input_tensor->data_type);
            break;
    }

    return true;
}
