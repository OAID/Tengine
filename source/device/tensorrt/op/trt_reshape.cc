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

#include <NvInferRuntime.h>


bool TensorRTEngine::AddReshapeNode(struct graph* ir_graph, struct node* node)
{
    struct tensor* reshape_input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* reshape_output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    if (nullptr == reshape_input || nullptr == reshape_output)
    {
        fprintf(stderr, "Tengine: Get input & output for Reshape(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    if (!check_if_input_in_map(reshape_input->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for Reshape(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    nvinfer1::ITensor* trt_tensor = tensor_real_map[tensor_swap_map[reshape_input->index]];

    nvinfer1::IShuffleLayer* layer = this->network->addShuffle(*trt_tensor);
    if (nullptr == layer)
    {
        fprintf(stderr, "Tengine: Add Reshape(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    layer->setName(node->name);

    nvinfer1::Dims dims{};
    dims.nbDims = reshape_output->dim_num;

    for (int i = 0; i < dims.nbDims; i++)
        dims.d[i] = reshape_output->dims[i];

    layer->setReshapeDimensions(dims);

    this->layer_map[node->index] = layer;

    nvinfer1::ITensor* reshape_output_tensor = layer->getOutput(0);

    this->SetRange(reshape_output, reshape_output_tensor);

    this->tensor_real_map[node->output_tensors[0]] = reshape_output_tensor;
    this->tensor_swap_map[node->output_tensors[0]] = node->output_tensors[0];

    return true;
}
