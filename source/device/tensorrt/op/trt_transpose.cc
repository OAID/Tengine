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

EXPORT_BEGIN
#include "transpose_param.h"
EXPORT_FINISH


bool TensorRTEngine::AddTranspose(struct graph *ir_graph, struct node *node)
{
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    if (nullptr == input || nullptr == output)
    {
        fprintf(stderr, "Tengine: Get input & output for Transpose(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    if (!check_if_input_in_map(input->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for Transpose(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    auto transpose_param = (struct transpose_param*)node->op.param_mem;
    if (nullptr == transpose_param || transpose_param->tr_shape_size <= 0)
    {
        fprintf(stderr, "Tengine: TensorRT get transpose param failed.\n");
        return false;
    }

    nvinfer1::ITensor* input_tensor = tensor_real_map[tensor_swap_map[input->index]];

    nvinfer1::IShuffleLayer* layer = this->network->addShuffle(*input_tensor);
    if (nullptr == layer)
    {
        fprintf(stderr, "Tengine: Add Transpose(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    layer->setName(node->name);

    nvinfer1::Dims dims{};
    dims.nbDims = output->dim_num;

    for (int i = 0; i < dims.nbDims; i++)
        dims.d[i] = output->dims[i];

    nvinfer1::Permutation order = { 0 };
    for (int i = 0; i < transpose_param->tr_shape_size; i++)
    {
        order.order[i] = transpose_param->tr_shape[i];
    }
    for (int i = transpose_param->tr_shape_size; i < nvinfer1::Dims::MAX_DIMS; i++)
    {
        order.order[i] = 0;
    }

    layer->setZeroIsPlaceholder(false);

    layer->setReshapeDimensions(dims);
    layer->setFirstTranspose(order);

    this->layer_map[node->index] = layer;

    nvinfer1::ITensor* output_tensor = layer->getOutput(0);

    this->SetRange(output, output_tensor);

    this->tensor_real_map[node->output_tensors[0]] = output_tensor;
    this->tensor_swap_map[node->output_tensors[0]] = node->output_tensors[0];

    return true;
}
