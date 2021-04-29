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
#include "eltwise_param.h"
}


bool VXEngine::AddEltwiseNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;

    std::vector<std::shared_ptr<tim::vx::Tensor> > add_in_tensor(ir_node->input_num);
    for (int i = 0; i < ir_node->input_num; i++)
    {
        struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);
        add_in_tensor[i] = this->vx_tensor_map[input_tensor->index];
    }
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    eltwise_param* param = (eltwise_param*)ir_node->op.param_mem;

    switch (param->type)
    {
        case ELT_SUM:
        {
            auto eltsum = graph->CreateOperation<tim::vx::ops::Add>();
            (*eltsum)
                .BindInputs(add_in_tensor)
                .BindOutputs({ this->vx_tensor_map[output_tensor->index] });
            break;
        }
        case ELT_SUB:
        {
            auto eltsub = graph->CreateOperation<tim::vx::ops::Sub>();
            (*eltsub)
                .BindInputs(add_in_tensor)
                .BindOutputs({ this->vx_tensor_map[output_tensor->index] });
            break;
        }
        default:
            break;
    }

    return 0;
}

