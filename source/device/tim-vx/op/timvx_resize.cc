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
#include "resize_param.h"
}


bool VXEngine::AddResizeNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;

    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct resize_param* param = (struct resize_param*)ir_node->op.param_mem;

    tim::vx::ResizeType resize_type;
    if (param->type == 0)
    {
        resize_type = tim::vx::ResizeType::NEAREST_NEIGHBOR;
    }
    else if(param->type == 1)
    {
        resize_type = tim::vx::ResizeType::BILINEAR;
    }
    else
    {
        TLOG_ERR("Tengine: VX does not support resize type(%d).\n", (int)resize_type);
    }

    std::vector<std::shared_ptr<tim::vx::Tensor> > add_in_tensor(ir_node->input_num);
    for (int i = 0; i < ir_node->input_num; i++)
    {
        struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);
        add_in_tensor[i] = this->vx_tensor_map[input_tensor->index];
    }

    auto resize = graph->CreateOperation<tim::vx::ops::Resize>(resize_type, 0.0f, false, false, output_tensor->dims[2], output_tensor->dims[3]);
    vx_node_map[ir_node->index] = resize;

    (*resize)
        .BindInputs(add_in_tensor)
        .BindOutputs({ this->vx_tensor_map[output_tensor->index] });

    return true;
}
