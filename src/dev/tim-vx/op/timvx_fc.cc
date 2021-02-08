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
#include "convolution_param.h"
}


bool VXEngine::AddFullyConnectionNode(struct ir_node* ir_node)
{
    TLOG_INFO("Tengine TIM-VX: Support OP(%d) OP_FC.\n", ir_node->idx);
    struct ir_graph* ir_graph = ir_node->graph;

    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    auto fc = graph->CreateOperation<tim::vx::ops::FullyConnected>(
        2, weight_tensor->dims[0]);

    if (ir_node->input_num > 2)
    {
        TLOG_INFO("Log:Use Bias\n");
        struct ir_tensor* bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        (*fc)
            .BindInputs({this->vx_tensor_map[input_tensor->idx], this->vx_tensor_map[weight_tensor->idx], this->vx_tensor_map[bias_tensor->idx]})
            .BindOutputs({ this->vx_tensor_map[output_tensor->idx] });
    }
    else
    {
        (*fc)
            .BindInputs({ this->vx_tensor_map[input_tensor->idx], this->vx_tensor_map[weight_tensor->idx] })
            .BindOutputs({ this->vx_tensor_map[output_tensor->idx] });
    }

    return true;
}
