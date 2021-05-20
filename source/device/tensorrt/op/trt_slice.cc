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
#include "slice_param.h"
EXPORT_FINISH

#include <NvInferRuntime.h>


bool TensorRTEngine::AddSliceNode(struct graph* ir_graph, struct node* node)
{
    struct tensor* slice_input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* slice_output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    if (nullptr == slice_input || nullptr == slice_output)
    {
        fprintf(stderr, "Tengine: Get input & output for Slice(id: %d, name: %s) layer failed.\n", node->index,
                node->name);
        return false;
    }

    if (!check_if_input_in_map(slice_input->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for Slice(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    auto* param = ( struct slice_param* )node->op.param_mem;

    nvinfer1::Dims start, size, stride;

    start.nbDims = slice_input->dim_num;
    size.nbDims = slice_input->dim_num;
    stride.nbDims = slice_input->dim_num;

    if (0 != param->axis)
    {
        for (uint8_t i = 0; i < slice_input->dim_num; i++)
        {
            start.d[i] = 0;
            size.d[i] = slice_output->dims[i];
            stride.d[i] = 1;
        }

        start.d[param->axis] = param->begin;
        size.d[param->axis] = param->end - param->begin;
    }
    else
    {
        fprintf(stderr, "Tengine: Slice(id: %d, name: %s) in batch is not supported for now.\n", node->index, node->name);
        return false;
    }

    nvinfer1::ITensor* trt_tensor = tensor_real_map[tensor_swap_map[slice_input->index]];

    nvinfer1::ISliceLayer* layer = this->network->addSlice(*trt_tensor, start, size, stride);
    if (nullptr == layer)
    {
        fprintf(stderr, "Tengine: Add Slice(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    layer->setName(node->name);

    this->layer_map[node->index] = layer;

    nvinfer1::ITensor* slice_output_tensor = layer->getOutput(0);

    this->SetRange(slice_output, slice_output_tensor);

    this->tensor_real_map[node->output_tensors[0]] = slice_output_tensor;
    this->tensor_swap_map[node->output_tensors[0]] = node->output_tensors[0];

    return true;
}
