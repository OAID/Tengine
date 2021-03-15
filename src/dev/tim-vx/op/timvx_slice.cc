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
#include "tengine_op.h"
#include "slice_param.h"
}


bool VXEngine::AddSliceNode(struct ir_node* ir_node)
{
    TLOG_INFO("Tengine TIM-VX: Support OP(%d) OP_SLICE.\n", ir_node->idx);
    struct ir_graph* ir_graph = ir_node->graph;

    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct slice_param* param = (struct slice_param*)ir_node->op.param_mem;

    uint32_t axis = output_tensor->dim_num - param->axis;

    std::vector<int32_t> start;
    for (int i = output_tensor->dim_num - 1; i >= 0; i--)
    {
        if (axis == i)
            start.push_back(param->begin);
        else
            start.push_back(0);
    }

    std::vector<int32_t> length;
    for (int i = output_tensor->dim_num - 1; i >= 0; i--)
    {
        if (axis == i)
            length.push_back(param->end - param->begin);
        else
            length.push_back(-1);
    }

    auto slice = this->graph->CreateOperation<tim::vx::ops::Slice>(output_tensor->dim_num, start, length);
    (*slice).BindInput( this->vx_tensor_map[input_tensor->idx] )
        .BindOutput({ this->vx_tensor_map[output_tensor->idx] });

    return true;
}
