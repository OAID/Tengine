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
#include "permute_param.h"
EXPORT_FINISH

#include <NvInferRuntime.h>


bool TensorRTEngine::AddPermuteNode(struct graph* ir_graph, struct node* node)
{
    struct tensor* permute_input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* permute_output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    if (nullptr == permute_input || nullptr == permute_output)
    {
        fprintf(stderr, "Tengine: Get input & output for Permute(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    if (!check_if_input_in_map(permute_input->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for Permute(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    nvinfer1::ITensor* trt_tensor = tensor_real_map[tensor_swap_map[permute_input->index]];

    permute_param* param = (permute_param*)node->op.param_mem;
    if (0 != param->order0)
    {
        fprintf(stderr, "Tengine: TensorRT does not support permute in N (batch) dimension(now is %d), and order index must be within the tensor dimensions.\n", param->order0);
        return false;
    }

    nvinfer1::IShuffleLayer* layer = this->network->addShuffle(*trt_tensor);
    if (nullptr == layer)
    {
        fprintf(stderr, "Tengine: Add Permute(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    layer->setName(node->name);

    nvinfer1::Dims dims{};
    dims.nbDims = permute_output->dim_num;

    for (int i = 0; i < dims.nbDims; i++)
        dims.d[i] = permute_output->dims[i];

    nvinfer1::Permutation order;
    order.order[0] = param->order0;
    order.order[1] = param->order1;
    order.order[2] = param->order2;
    order.order[3] = param->order3;

    layer->setZeroIsPlaceholder(false);

    layer->setReshapeDimensions(dims);
    layer->setFirstTranspose(order);

    this->layer_map[node->index] = layer;

    nvinfer1::ITensor* permute_output_tensor = layer->getOutput(0);

    this->SetRange(permute_output, permute_output_tensor);

    this->tensor_real_map[node->output_tensors[0]] = permute_output_tensor;
    this->tensor_swap_map[node->output_tensors[0]] = node->output_tensors[0];

    return true;
}
