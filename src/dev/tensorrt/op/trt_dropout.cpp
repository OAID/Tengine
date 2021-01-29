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

#include "trt_executor.hpp"

extern "C"
{
#include "tengine_op.h"
}


bool TensorRTEngine::AddDropoutNode(struct ir_graph* ir_graph, struct ir_node* node)
{
    struct ir_tensor* drop_input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct ir_tensor* drop_output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    if (nullptr != drop_input && nullptr != drop_output && !check_if_input_in_map(drop_input->idx, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input & output for Dropout(id: %d, name: %s) layer failed.\n", node->idx, node->name);
        return false;
    }

    nvinfer1::ITensor * drop_input_tensor = tensor_real_map[tensor_swap_map[drop_input->idx]];

    nvinfer1::IShuffleLayer* layer = this->network->addShuffle(*drop_input_tensor);
    if (nullptr == layer)
    {
        fprintf(stderr, "Tengine: Add Reshape(id: %d, name: %s) layer failed.\n", node->idx, node->name);
        return false;
    }

    layer->setName(node->name);

    nvinfer1::Dims dims{};
    dims.nbDims = drop_output->dim_num;

    for (int i = 0; i < dims.nbDims; i++)
        dims.d[i] = drop_output->dims[i];

    layer->setReshapeDimensions(dims);
    layer->setZeroIsPlaceholder(false);

    this->layer_map[node->idx] = layer;

    nvinfer1::ITensor* drop_output_tensor = layer->getOutput(0);

    this->SetRange(drop_output, drop_output_tensor);

    tensor_real_map[drop_output->idx] = drop_output_tensor;
    tensor_swap_map[drop_output->idx] = drop_output->idx;

    return true;
}
