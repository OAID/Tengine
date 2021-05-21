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

bool TensorRTEngine::AddHardSwishNode(struct graph* ir_graph, struct node* node)
{
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    if (nullptr == input || nullptr == output)
    {
        fprintf(stderr, "Tengine: Get input & output for HardSwish(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    if (!check_if_input_in_map(input->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for HardSwish(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    nvinfer1::ITensor* input_tensor = tensor_real_map[tensor_swap_map[input->index]];

    float* param_buffer = (float*)sys_malloc(3 * sizeof(float));
    this->host_buffer.push_back(param_buffer);

    param_buffer[0] = 1.f / 6.f, param_buffer[1] = 0.5f, param_buffer[2] = 1.f;
    nvinfer1::Weights lambda_scale{nvinfer1::DataType::kFLOAT, &(param_buffer[0]), 1};
    nvinfer1::Weights lambda_shift{nvinfer1::DataType::kFLOAT, &(param_buffer[1]), 1};
    nvinfer1::Weights lambda_power{nvinfer1::DataType::kFLOAT, &(param_buffer[2]), 1};

    nvinfer1::IScaleLayer* scale_layer = this->network->addScale(*input_tensor, nvinfer1::ScaleMode::kUNIFORM, lambda_shift, lambda_scale, lambda_power);
    std::string scale_layer_name = std::string(node->name) + "_scale";
    scale_layer->setName(scale_layer_name.c_str());

    auto scale_layer_output = scale_layer->getOutput(0);

    nvinfer1::IActivationLayer* relu1_layer = this->network->addActivation(*scale_layer_output, nvinfer1::ActivationType::kCLIP);
    relu1_layer->setAlpha(0.f);
    relu1_layer->setBeta(1.f);

    std::string relu1_layer_name = std::string(node->name) + "_relu1";
    relu1_layer->setName(relu1_layer_name.c_str());

    auto relu1_output = relu1_layer->getOutput(0);

    nvinfer1::IElementWiseLayer* product_Layer = this->network->addElementWise(*input_tensor, *relu1_output, nvinfer1::ElementWiseOperation::kPROD);
    product_Layer->setName(node->name);

    auto product_output = product_Layer->getOutput(0);

    this->SetRange(output, product_output);

    this->tensor_real_map[output->index] = product_output;
    this->tensor_swap_map[output->index] = output->index;

    return true;
}
