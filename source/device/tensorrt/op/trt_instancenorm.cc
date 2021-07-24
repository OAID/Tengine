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
#include "instancenorm_param.h"
EXPORT_FINISH

bool TensorRTEngine::AddInstanceNormNode(struct graph* ir_graph, struct node* node)
{
    struct tensor* input_tensor  = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* gamma_tensor  = get_ir_graph_tensor(ir_graph, node->input_tensors[1]);
    struct tensor* beta_tensor   = get_ir_graph_tensor(ir_graph, node->input_tensors[2]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    if (nullptr == input_tensor || nullptr == output_tensor)
    {
        fprintf(stderr, "Tengine: Get input & output for HardSwish(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    if (!check_if_input_in_map(input_tensor->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for HardSwish(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    nvinfer1::ITensor* input_trt_tensor = tensor_real_map[tensor_swap_map[input_tensor->index]];
///////////////////
    nvinfer1::Weights kernel{nvinfer1::DataType::kFLOAT, nullptr, 0};
    int HW = input_tensor->dims[2] * input_tensor->dims[3];
    int CHW = input_tensor->dims[1] * HW;
    float* weight_buffer = (float*)sys_malloc(CHW * sizeof(float) );
    this->host_buffer.push_back(weight_buffer);
    for (int i = 0; i < CHW; i++)
    {
        weight_buffer[i] = 1.f/HW;
    }
    kernel.values = (void*)weight_buffer;
    kernel.count = CHW;

    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::DimsHW  kernel_size{ input_tensor->dims[2], input_tensor->dims[3] };

    nvinfer1::IConvolutionLayer* layer_mean = this->network->addConvolution(*input_trt_tensor, input_tensor->dims[1], kernel_size, kernel, bias);

    layer_mean->setStride(nvinfer1::DimsHW(1, 1));
    layer_mean->setPrePadding(nvinfer1::DimsHW(0, 0));
    layer_mean->setPostPadding(nvinfer1::DimsHW(0, 0));
    layer_mean->setDilation(nvinfer1::DimsHW(1, 1));
    layer_mean->setNbGroups(input_tensor->dims[1]);

    std::string layer_mean_name = std::string(node->name) + "_mean";
    layer_mean->setName(layer_mean_name.c_str());
    auto tensor_mean = layer_mean->getOutput(0);
///////////////////
    nvinfer1::IElementWiseLayer* layer_sub = this->network->addElementWise(*input_trt_tensor, *tensor_mean, nvinfer1::ElementWiseOperation::kSUB);
    std::string layer_sub_name = std::string(node->name) + "_sub";
    layer_sub->setName(layer_sub_name.c_str());
    auto tensor_sub = layer_sub->getOutput(0);
///////////////////
    nvinfer1::IElementWiseLayer* layer_prob = this->network->addElementWise(*tensor_sub, *tensor_sub, nvinfer1::ElementWiseOperation::kPROD);
    std::string layer_prob_name = std::string(node->name) + "_prob";
    layer_prob->setName(layer_prob_name.c_str());
    auto tensor_prob = layer_prob->getOutput(0);

    float* param_buffer = (float*)sys_malloc(3 * sizeof(float));
    this->host_buffer.push_back(param_buffer);
///////////////////
    nvinfer1::IConvolutionLayer* layer_prob_sum = this->network->addConvolution(*tensor_prob, input_tensor->dims[1], kernel_size, kernel, bias);

    layer_prob_sum->setStride(nvinfer1::DimsHW(1, 1));
    layer_prob_sum->setPrePadding(nvinfer1::DimsHW(0, 0));
    layer_prob_sum->setPostPadding(nvinfer1::DimsHW(0, 0));
    layer_prob_sum->setDilation(nvinfer1::DimsHW(1, 1));
    layer_prob_sum->setNbGroups(input_tensor->dims[1]);

    std::string layer_prob_sum_name = std::string(node->name) + "_prob_sum";
    layer_prob_sum->setName(layer_prob_sum_name.c_str());
    auto tensor_prob_sum = layer_prob_sum->getOutput(0);
///////////////////
    auto param = (struct instancenorm_Param*)node->op.param_mem;

    nvinfer1::Weights add_eps{nvinfer1::DataType::kFLOAT, nullptr, 0};

    float* add_eps_buffer = (float*)sys_malloc(input_tensor->dims[0] * input_tensor->dims[1] * sizeof(float) );
    this->host_buffer.push_back(add_eps_buffer);

    for (int i = 0; i < input_tensor->dims[0] * input_tensor->dims[1]; i++)
        add_eps_buffer[i] = param->eps;

    add_eps.values = add_eps_buffer;
    add_eps.count = input_tensor->dims[0] * input_tensor->dims[1];

    nvinfer1::Dims4 dim4_eps(input_tensor->dims[0], input_tensor->dims[1], 1, 1);
    nvinfer1::IConstantLayer* layer_eps = this->network->addConstant(dim4_eps, add_eps);
    std::string layer_eps_name = std::string(node->name) + "_scale_eps";
    layer_eps->setName(layer_eps_name.c_str());
    nvinfer1::ITensor * trt_eps_tensor = layer_eps->getOutput(0);

    nvinfer1::IElementWiseLayer* layer_scale = this->network->addElementWise(*tensor_prob_sum, *trt_eps_tensor, nvinfer1::ElementWiseOperation::kSUM);
    std::string layer_scale_name = std::string(node->name) + "_scale";
    layer_scale->setName(layer_scale_name.c_str());
    auto tensor_scale = layer_scale->getOutput(0);
///////////////////
    nvinfer1::IUnaryLayer* layer_sqrt = this->network->addUnary(*tensor_scale, nvinfer1::UnaryOperation::kSQRT);
    std::string layer_sqrt_name = std::string(node->name) + "_sqrt";
    layer_sqrt->setName(layer_sqrt_name.c_str());
    auto tensor_sqrt = layer_sqrt->getOutput(0);
///////////////////
    float* gamma_data = (float*)get_tensor_buffer(gamma_tensor);

    nvinfer1::Weights weight_gamma{nvinfer1::DataType::kFLOAT, nullptr, 0};

    weight_gamma.values = gamma_data;
    weight_gamma.count = gamma_tensor->elem_num;
    weight_gamma.type = nvinfer1::DataType::kFLOAT;

    nvinfer1::Dims4 dim4(1, gamma_tensor->dims[0], 1, 1);
    nvinfer1::IConstantLayer* layer_input1 = this->network->addConstant(dim4, weight_gamma);
    std::string layer_gamma = std::string(gamma_tensor->name) + "_gamma";
    layer_input1->setName(layer_gamma.c_str());

    nvinfer1::ITensor * trt_input1_tensor = layer_input1->getOutput(0);

    nvinfer1::IElementWiseLayer* layer_scale_a = this->network->addElementWise(*trt_input1_tensor, *tensor_sqrt, nvinfer1::ElementWiseOperation::kDIV);
    std::string layer_scale_a_name = std::string(node->name) + "_scale_a";
    layer_scale_a->setName(layer_scale_a_name.c_str());
    auto tensor_scale_a = layer_scale_a->getOutput(0);
///////////////////
    nvinfer1::IElementWiseLayer* layer_scale_b_mul = this->network->addElementWise(*tensor_mean, *tensor_scale_a, nvinfer1::ElementWiseOperation::kPROD);
    std::string layer_scale_b_mul_name = std::string(node->name) + "_scale_b_mul";
    layer_scale_b_mul->setName(layer_scale_b_mul_name.c_str());
    auto tensor_scale_b_mul_pos = layer_scale_b_mul->getOutput(0);
///////////////////
    param_buffer[0] = -1.f, param_buffer[1] = 0.0f, param_buffer[2] = 1.f;
    nvinfer1::Weights lambda_scale_neg{nvinfer1::DataType::kFLOAT, &(param_buffer[0]), 1};
    nvinfer1::Weights lambda_shift_neg{nvinfer1::DataType::kFLOAT, &(param_buffer[1]), 1};
    nvinfer1::Weights lambda_power_neg{nvinfer1::DataType::kFLOAT, &(param_buffer[2]), 1};

    nvinfer1::IScaleLayer* layer_scale_neg = this->network->addScale(*tensor_scale_b_mul_pos, nvinfer1::ScaleMode::kUNIFORM, lambda_shift_neg, lambda_scale_neg, lambda_power_neg);
    std::string layer_scale_neg_name = std::string(node->name) + "_scale_neg";
    layer_scale_neg->setName(layer_scale_neg_name.c_str());
    auto tensor_scale_b_mul = layer_scale_neg->getOutput(0);
///////////////////
    float* beta_data  = (float*)get_tensor_buffer(beta_tensor);

    nvinfer1::Weights bias_beta{nvinfer1::DataType::kFLOAT, nullptr, 0};

    bias_beta.values = beta_data;
    bias_beta.count = beta_tensor->elem_num;
    bias_beta.type = nvinfer1::DataType::kFLOAT;

    nvinfer1::Dims4 dim4_beta(1, beta_tensor->dims[0], 1, 1);
    nvinfer1::IConstantLayer* layer_input2 = this->network->addConstant(dim4_beta, bias_beta);
    std::string layer_beta = std::string(beta_tensor->name) + "_beta";
    layer_input2->setName(layer_beta.c_str());

    nvinfer1::ITensor * trt_input2_tensor = layer_input2->getOutput(0);

    nvinfer1::IElementWiseLayer* layer_add_b = this->network->addElementWise(*tensor_scale_b_mul, *trt_input2_tensor, nvinfer1::ElementWiseOperation::kSUM);
    std::string layer_add_b_name = std::string(node->name) + "_add_b";
    layer_add_b->setName(layer_add_b_name.c_str());
    auto tensor_add_b = layer_add_b->getOutput(0);
///////////////////
    nvinfer1::IElementWiseLayer* layer_prob_scale_a = this->network->addElementWise(*input_trt_tensor, *tensor_scale_a, nvinfer1::ElementWiseOperation::kPROD);
    std::string layer_prob_scale_a_name = std::string(node->name) + "_mul_scale_a";
    layer_prob_scale_a->setName(layer_prob_scale_a_name.c_str());
    auto tensor_prob_scale_a = layer_prob_scale_a->getOutput(0);
///////////////////
    nvinfer1::IElementWiseLayer* layer_sum_add_b = this->network->addElementWise(*tensor_prob_scale_a, *tensor_add_b, nvinfer1::ElementWiseOperation::kSUM);
    std::string layer_sum_add_b_name = std::string(node->name) + "_sum_add_b";
    layer_sum_add_b->setName(layer_sum_add_b_name.c_str());
    auto tensor_sum_add_b = layer_sum_add_b->getOutput(0);
///////////////////
    this->layer_map[node->index] = layer_sum_add_b;

    this->SetRange(output_tensor, tensor_sum_add_b);

    this->tensor_real_map[output_tensor->index] = tensor_sum_add_b;
    this->tensor_swap_map[output_tensor->index] = output_tensor->index;

    return true;
}
