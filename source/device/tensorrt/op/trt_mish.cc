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


/*
 * y = x * tanh ( ln(1 + e^x) )
 *   = x * ( (1 + e^x)^2 - 1 ) / ( (1 + e^x)^2 + 1 )
 */
bool TensorRTEngine::AddMishNode(struct graph* ir_graph, struct node* node)
{
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    if (nullptr == input || nullptr == output)
    {
        fprintf(stderr, "Tengine: Get input & output for Mish(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    if (!check_if_input_in_map(input->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for Mish(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    nvinfer1::ITensor* input_tensor = tensor_real_map[tensor_swap_map[input->index]];

    // get e^x
    nvinfer1::IUnaryLayer* ex_layer = this->network->addUnary(*input_tensor, nvinfer1::UnaryOperation::kEXP);

    std::string ex_layer_name = std::string(node->name) + "_ex";
    ex_layer->setName(ex_layer_name.c_str());

    auto ex_output = ex_layer->getOutput(0);

    float* param_buffer = (float*)sys_malloc(3 * sizeof(float));
    this->host_buffer.push_back(param_buffer);

    param_buffer[0] = 1.f, param_buffer[1] = -1.f, param_buffer[2] = 2.f;

    // get (1 + e^x)^2
    nvinfer1::Weights ex_pos_1_param{nvinfer1::DataType::kFLOAT, &param_buffer[0], 1};
    nvinfer1::Weights ex_2_param{nvinfer1::DataType::kFLOAT, &param_buffer[2], 1};
    nvinfer1::IScaleLayer* ex_scaled_layer = this->network->addScale(*ex_output, nvinfer1::ScaleMode::kUNIFORM, ex_pos_1_param, ex_pos_1_param, ex_2_param);

    std::string ex_scaled_layer_name = std::string(node->name) + "_scale";
    ex_scaled_layer->setName(ex_scaled_layer_name.c_str());

    auto ex_scaled_output = ex_scaled_layer->getOutput(0);

    // get (1 + e^x)^2 + 1, (1 + e^x)^2 - 1
    nvinfer1::Weights ex_neg_1_param{nvinfer1::DataType::kFLOAT, &param_buffer[1], 1};
    nvinfer1::IScaleLayer* numerator_layer = this->network->addScale(*ex_scaled_output, nvinfer1::ScaleMode::kUNIFORM, ex_pos_1_param, ex_pos_1_param, ex_pos_1_param);
    nvinfer1::IScaleLayer* denominator_layer = this->network->addScale(*ex_scaled_output, nvinfer1::ScaleMode::kUNIFORM, ex_pos_1_param, ex_neg_1_param, ex_pos_1_param);

    std::string numerator_layer_name = std::string(node->name) + "_numerator";
    std::string denominator_layer_name = std::string(node->name) + "_denominator";
    numerator_layer->setName(numerator_layer_name.c_str());
    denominator_layer->setName(denominator_layer_name.c_str());

    auto numerator_output = numerator_layer->getOutput(0);
    auto denominator_output = denominator_layer->getOutput(0);

    // get { (1 + e^x)^2 + 1 } / { (1 + e^x)^2 - 1 }
    nvinfer1::IElementWiseLayer* fraction_layer = this->network->addElementWise(*numerator_output, *denominator_output, nvinfer1::ElementWiseOperation::kDIV);

    std::string fraction_layer_name = std::string(node->name) + "_fraction";
    fraction_layer->setName(fraction_layer_name.c_str());

    auto fraction_output = fraction_layer->getOutput(0);

    // get x * { (1 + e^x)^2 + 1 } / { (1 + e^x)^2 - 1 }
    nvinfer1::IElementWiseLayer* product_layer = this->network->addElementWise(*input_tensor, *fraction_output, nvinfer1::ElementWiseOperation::kPROD);

    std::string product_layer_name = std::string(node->name) + "_product";
    product_layer->setName(product_layer_name.c_str());

    auto product_output = product_layer->getOutput(0);

    this->layer_map[node->index] = product_layer;

    this->SetRange(output, product_output);

    this->tensor_real_map[output->index] = product_output;
    this->tensor_swap_map[output->index] = output->index;

    return true;
}
