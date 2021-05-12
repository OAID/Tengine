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
 * Author: lswang@openailab.com
 */

#include "../trt_executor.hpp"

EXPORT_BEGIN
#include "softmax_param.h"
EXPORT_FINISH


bool TensorRTEngine::AddSoftmaxNode(struct graph* ir_graph, struct node* node)
{
    struct tensor* softmax_input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    if (nullptr != softmax_input && !check_if_input_in_map(softmax_input->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for Softmax(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    nvinfer1::ITensor * trt_tensor = tensor_real_map[tensor_swap_map[softmax_input->index]];

    nvinfer1::ISoftMaxLayer * layer = this->network->addSoftMax(*trt_tensor);
    if (nullptr == layer)
    {
        fprintf(stderr, "Tengine: Add Softmax(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    struct softmax_param* param = (struct softmax_param*) node->op.param_mem;
    uint32_t axes = 1u << (param->axis - static_cast<int>(this->network->hasImplicitBatchDimension()));
    layer->setAxes(axes);

    layer->setName(node->name);

    this->layer_map[node->index] = layer;

    // TODO: set the softmax axis, current the tensor rt default policy works
    trt_tensor = layer->getOutput(0);

    struct tensor* softmax_output  = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    this->SetRange(softmax_output, trt_tensor);

    this->tensor_real_map[softmax_output->index] = trt_tensor;
    this->tensor_swap_map[softmax_output->index] = softmax_output->index;

    return true;
}
