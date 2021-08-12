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
#include "crop_param.h"
}

bool VXEngine::AddCropNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct crop_param* param = (struct crop_param*)ir_node->op.param_mem;

    std::vector<int32_t> start;
    std::vector<int32_t> end;

    if (param->num_args == 1)
    {
        int offsetH = (input_tensor->dims[2] - param->crop_h) / 2;
        int offsetW = (input_tensor->dims[3] - param->crop_w) / 2;

        start.push_back(offsetW);
        start.push_back(offsetH);
        start.push_back(0);
        start.push_back(0);

        end.push_back(output_tensor->dims[3]);
        end.push_back(output_tensor->dims[2]);
        end.push_back(output_tensor->dims[1]);
        end.push_back(output_tensor->dims[0]);
    }
    else if (param->num_args == 2)
    {
        int offsetH = input_tensor->dims[2] ;
        int offsetW = input_tensor->dims[3];

        start.push_back(offsetW);
        start.push_back(offsetH);
        start.push_back(0);
        start.push_back(0);

        end.push_back(output_tensor->dims[3]);
        end.push_back(output_tensor->dims[2]);
        end.push_back(output_tensor->dims[1]);
        end.push_back(output_tensor->dims[0]);
    }

    std::vector<int32_t> stride;
    stride.push_back(1);
    stride.push_back(1);
    stride.push_back(1);
    stride.push_back(1);

    auto crop = this->graph->CreateOperation<tim::vx::ops::StridedSlice>(start, end, stride,
                                                                          15, 15, 15);
    vx_node_map[ir_node->index] = crop;
    (*crop)
        .BindInputs({ this->vx_tensor_map[input_tensor->index] })
        .BindOutputs({ this->vx_tensor_map[output_tensor->index] });

    return true;
}

