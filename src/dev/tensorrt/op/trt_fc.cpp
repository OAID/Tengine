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

#include <NvInferRuntime.h>


bool TensorRTEngine::AddFullyConnectedNode(struct ir_graph* ir_graph, struct ir_node* node)
{
    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct ir_tensor* weight_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[1]);;
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);;

    if (nullptr == input_tensor || nullptr == weight_tensor || nullptr == output_tensor)
    {
        fprintf(stderr, "Tengine: Get input & output for FullyConnected(id: %d, name: %s) layer failed.\n", node->idx, node->name);
        return false;
    }

    /* input: NxK weight: MxK  output: NxM */
    int M = output_tensor->dims[1];
    int K = weight_tensor->dims[1];

    nvinfer1::Weights kernel{nvinfer1::DataType::kFLOAT, nullptr, 0};

    if (0 != get_type(weight_tensor->data_type, kernel.type))
    {
        fprintf(stderr, "Tengine: Tensor weight type(%d) cannot supported.\n", weight_tensor->data_type);
        return false;
    }

    kernel.values = weight_tensor->data;
    kernel.count = M * K;

    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};

    if(2 < node->input_num)
    {
        struct ir_tensor* bias_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[2]);
        get_type(bias_tensor->data_type, bias.type);
        bias.values = bias_tensor->data;
        bias.count = M;
    }

    if (!check_if_input_in_map(input_tensor->idx, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for FullyConnected(id: %d, name: %s) layer failed.\n", node->idx, node->name);
        return false;
    }

    nvinfer1::ITensor * trt_tensor = tensor_real_map[tensor_swap_map[input_tensor->idx]];

    nvinfer1::IFullyConnectedLayer* layer = this->network->addFullyConnected(*trt_tensor, M, kernel, bias);
    if (nullptr == layer)
    {
        fprintf(stderr, "Tengine: Add FullyConnected(id: %d, name: %s) layer failed.\n", node->idx, node->name);
        return false;
    }

    layer->setName(node->name);

    this->layer_map[node->idx] = layer;

    trt_tensor = layer->getOutput(0);

    this->SetRange(output_tensor, trt_tensor);

    tensor_real_map[output_tensor->idx] = trt_tensor;
    tensor_swap_map[output_tensor->idx] = output_tensor->idx;

    return true;
}
