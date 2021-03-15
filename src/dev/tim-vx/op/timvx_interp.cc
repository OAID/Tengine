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
#include "interp_param.h"
}


bool VXEngine::AddInterpNode(struct ir_node* ir_node)
{
    TLOG_INFO("Tengine TIM-VX: Support OP(%d) OP_INTERP.\n", ir_node->idx);
    struct ir_graph* ir_graph = ir_node->graph;

    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct interp_param* param = (struct interp_param*)ir_node->op.param_mem;

    tim::vx::ResizeType resize_type;
    if (param->resize_type == 1)
        resize_type = tim::vx::ResizeType::NEAREST_NEIGHBOR;
    else if(param->resize_type == 2)
        resize_type = tim::vx::ResizeType::BILINEAR;
    else
        fprintf(stderr, " Not support this resize type(%d)\n",resize_type);

    auto resize = graph->CreateOperation<tim::vx::ops::Resize>(resize_type, 0.0f, false, false, param->output_height, param->output_width);
    (*resize)
        .BindInputs({ this->vx_tensor_map[input_tensor->idx] })
        .BindOutputs({ this->vx_tensor_map[output_tensor->idx] });

    return true;
}
