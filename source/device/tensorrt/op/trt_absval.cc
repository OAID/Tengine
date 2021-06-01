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


bool TensorRTEngine::AddAbsVal(struct graph* ir_graph, struct node* node)
{
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    if (nullptr == input || nullptr == output)
    {
        fprintf(stderr, "Tengine: Get input & output for AbsVal(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    if (!check_if_input_in_map(input->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for AbsVal(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    nvinfer1::ITensor* input_tensor = tensor_real_map[tensor_swap_map[input->index]];

    nvinfer1::IUnaryLayer* layer = this->network->addUnary(*input_tensor, nvinfer1::UnaryOperation::kABS);
    layer->setName(node->name);

    this->layer_map[node->index] = layer;

    auto layer_output = layer->getOutput(0);

    this->SetRange(output, layer_output);

    this->tensor_real_map[output->index] = layer_output;
    this->tensor_swap_map[output->index] = output->index;

    return true;
}
