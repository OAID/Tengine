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
#include "pad_param.h"
EXPORT_FINISH

#include <NvInferRuntime.h>


bool TensorRTEngine::AddPadNode(struct graph* ir_graph, struct node* node)
{
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    if (nullptr == input_tensor || nullptr == output_tensor)
    {
        fprintf(stderr, "Tengine: Get input & output for Pad(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    if (!check_if_input_in_map(input_tensor->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for Pad(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    nvinfer1::ITensor* trt_tensor = tensor_real_map[tensor_swap_map[input_tensor->index]];

    pad_param* param = (pad_param*)node->op.param_mem;

    nvinfer1::DimsHW dims_pre{};
    dims_pre.d[0] = param->pad_2_h;
    dims_pre.d[1] = param->pad_2_w;

    nvinfer1::DimsHW dims_post{};
    dims_post.d[0] = param->pad_3_h;
    dims_post.d[1] = param->pad_3_w;

    nvinfer1::IPaddingLayer* layer = this->network->addPadding(*trt_tensor, dims_pre, dims_post);
    if (nullptr == layer)
    {
        fprintf(stderr, "Tengine: Add Pad(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    layer->setName(node->name);

    this->layer_map[node->index] = layer;

    nvinfer1::ITensor* trt_output_tensor = layer->getOutput(0);

    this->SetRange(ir_graph, node->output_tensors[0], trt_output_tensor);

    this->tensor_real_map[node->output_tensors[0]] = trt_output_tensor;
    this->tensor_swap_map[node->output_tensors[0]] = node->output_tensors[0];

    return true;
}
