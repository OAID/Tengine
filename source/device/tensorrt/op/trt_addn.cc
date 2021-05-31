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


bool TensorRTEngine::AddAddN(struct graph* ir_graph, struct node* node)
{
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    if (nullptr == output)
    {
        fprintf(stderr, "Tengine: Get output for AddN(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    std::vector<nvinfer1::ITensor*> input_tensors(node->input_num);

    for (int i = 0; i < node->input_num; i++)
    {
        struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[i]);
        if (nullptr == input)
        {
            fprintf(stderr, "Tengine: Get input(%d) for AddN(id: %d, name: %s) layer failed.\n", i, node->index, node->name);
            return false;
        }

        if (!check_if_input_in_map(input->index, this->tensor_swap_map))
        {
            fprintf(stderr, "Tengine: Query input for AddN(id: %d, name: %s) layer failed.\n", node->index, node->name);
            return false;
        }

        nvinfer1::ITensor* input_tensor = tensor_real_map[tensor_swap_map[input->index]];
        input_tensors[i] = input_tensor;
    }

    if (input_tensors.size() == 1)
    {
        fprintf(stderr, "Tengine: Only 1 input for AddN(id: %d, name: %s) is not allowed.\n", node->index, node->name);
        return false;
    }

    int count = 0;
    nvinfer1::ITensor* temp_result = nullptr;
    nvinfer1::IElementWiseLayer* layer = nullptr;
    while (!input_tensors.empty())
    {
        if (nullptr == temp_result)
        {
            auto input_a = input_tensors[input_tensors.size() - 1];
            auto input_b = input_tensors[input_tensors.size() - 2];

            layer = this->network->addElementWise(*input_a, *input_b, nvinfer1::ElementWiseOperation::kSUM);
            std::string layer_name = std::string(node->name) + "_" + std::to_string(count);
            layer->setName(layer_name.c_str());

            temp_result = layer->getOutput(0);

            input_tensors.pop_back();
            input_tensors.pop_back();
        }
        else
        {
            auto input = input_tensors[input_tensors.size() - 1];

            layer = this->network->addElementWise(*input, *temp_result, nvinfer1::ElementWiseOperation::kSUM);
            std::string layer_name = std::string(node->name) + "_" + std::to_string(count);
            layer->setName(layer_name.c_str());

            temp_result = layer->getOutput(0);

            input_tensors.pop_back();
        }

        count++;
    }

    layer->setName(node->name);
    this->layer_map[node->index] = layer;

    this->SetRange(output, temp_result);

    this->tensor_real_map[output->index] = temp_result;
    this->tensor_swap_map[output->index] = output->index;

    return true;
}
