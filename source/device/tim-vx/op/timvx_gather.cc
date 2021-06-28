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
#include <malloc.h>
#include "operator/op.h"
#include "gather_param.h"
}


bool VXEngine::AddGatherNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct gather_param* param = (struct gather_param*)ir_node->op.param_mem;

    auto gather = graph->CreateOperation<tim::vx::ops::Gather>(3 - param->axis);
    vx_node_map[ir_node->index] = gather;

    auto Dims = (unsigned int*)input_tensor->dims;
    tim::vx::ShapeType vx_shape;
    vx_shape.push_back(param->indices_num);
    tim::vx::Quantization none_quant(tim::vx::QuantType::NONE, 1, 0);
    tim::vx::TensorSpec vx_spec(tim::vx::DataType::INT32, vx_shape,
                                tim::vx::TensorAttribute::CONSTANT, none_quant);

    int* data_indices = (int*)malloc(param->indices_num * sizeof(int) ) ;
    for (int i = 0; i < param->indices_num; i++)
        data_indices[i] = i;
    std::shared_ptr<tim::vx::Tensor> vx_tensor = this->graph->CreateTensor(vx_spec, data_indices);
    this->vx_tensor_map[ir_node->input_tensors[0] + ir_graph->tensor_num] = vx_tensor;

    (*gather)
        .BindInputs({ this->vx_tensor_map[input_tensor->index], vx_tensor })
        .BindOutputs({ this->vx_tensor_map[output_tensor->index] });

    return true;
}
