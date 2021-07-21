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
#include <math.h>

EXPORT_BEGIN
#include "batchnorm_param.h"
EXPORT_FINISH

bool TensorRTEngine::AddBatchNormNode(struct graph* ir_graph, struct node* node)
{
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* in_gamma_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[1]);
    struct tensor* in_beta_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[2]);
    struct tensor* in_mean_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[3]);
    struct tensor* in_var_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[4]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    if (nullptr == input_tensor || nullptr == output_tensor)
    {
        fprintf(stderr, "Tengine: Get input & output for BatchNorm(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    if (!check_if_input_in_map(input_tensor->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for BatchNorm(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    nvinfer1::ITensor* input_trt_tensor = tensor_real_map[tensor_swap_map[input_tensor->index]];

    auto param = (struct batchnorm_param*)node->op.param_mem;

    float* in_gamma = (float*)get_tensor_buffer(in_gamma_tensor);
    float* in_beta  = (float*)get_tensor_buffer(in_beta_tensor);
    float* in_mean  = (float*)get_tensor_buffer(in_mean_tensor);
    float* in_var   = (float*)get_tensor_buffer(in_var_tensor);

    float* in_scale = (float*)sys_malloc(in_gamma_tensor->elem_num * sizeof(float));
    float* in_shift = (float*)sys_malloc(in_gamma_tensor->elem_num * sizeof(float));
    this->host_buffer.push_back(in_scale);
    this->host_buffer.push_back(in_shift);

    int channel_num = input_tensor->dims[1];

    float* scale_mean = ( float* )sys_malloc(channel_num * sizeof(float));
    float* scale_var_inv = ( float* )sys_malloc(channel_num * sizeof(float));
    this->host_buffer.push_back(scale_mean);
    this->host_buffer.push_back(scale_var_inv);

    struct batchnorm_param* batchnorm_param = ( struct batchnorm_param* )node->op.param_mem;
    float rescale_factor = batchnorm_param->rescale_factor ? 1 / batchnorm_param->rescale_factor : 0;

    for (int c = 0; c < channel_num; c++)
    {
        float tmp = sqrtf(in_var[c] * rescale_factor + batchnorm_param->eps);
        scale_var_inv[c] = ( float )(1.f / tmp);
        tmp = rescale_factor * scale_var_inv[c];
        scale_mean[c] = ( float )(-in_mean[c] * tmp);
    }

    for (int i = 0; i < in_gamma_tensor->elem_num; i++)
    {
        in_scale[i] = in_gamma[i] * scale_var_inv[i];
        in_shift[i] = in_beta[i] + in_gamma[i] * scale_mean[i];
    }

////////////
    nvinfer1::Weights weight_gamma{nvinfer1::DataType::kFLOAT, nullptr, 0};

    weight_gamma.values = in_scale;
    weight_gamma.count = in_gamma_tensor->elem_num;
    weight_gamma.type = nvinfer1::DataType::kFLOAT;

    nvinfer1::Dims4 dim4(1, in_gamma_tensor->dims[0], 1, 1);
    nvinfer1::IConstantLayer* layer_input1 = this->network->addConstant(dim4, weight_gamma);
    std::string layer_gamma = std::string(in_gamma_tensor->name) + "_gamma";
    layer_input1->setName(layer_gamma.c_str());

    nvinfer1::ITensor * trt_input1_tensor = layer_input1->getOutput(0);

    nvinfer1::IElementWiseLayer* layer_scale_a = this->network->addElementWise(*input_trt_tensor, *trt_input1_tensor, nvinfer1::ElementWiseOperation::kPROD);
    std::string layer_scale_a_name = std::string(node->name) + "_scale_a";
    layer_scale_a->setName(layer_scale_a_name.c_str());
    auto tensor_scale_a = layer_scale_a->getOutput(0);
///////////
    nvinfer1::Weights bias_beta{nvinfer1::DataType::kFLOAT, nullptr, 0};

    bias_beta.values = in_shift;
    bias_beta.count = in_beta_tensor->elem_num;
    bias_beta.type = nvinfer1::DataType::kFLOAT;

    nvinfer1::Dims4 dim4_beta(1, in_beta_tensor->dims[0], 1, 1);
    nvinfer1::IConstantLayer* layer_input2 = this->network->addConstant(dim4_beta, bias_beta);
    std::string layer_beta = std::string(in_beta_tensor->name) + "_beta";
    layer_input2->setName(layer_beta.c_str());

    nvinfer1::ITensor * trt_input2_tensor = layer_input2->getOutput(0);

    nvinfer1::IElementWiseLayer* layer_add_b = this->network->addElementWise(*tensor_scale_a, *trt_input2_tensor, nvinfer1::ElementWiseOperation::kSUM);
    std::string layer_add_b_name = std::string(node->name) + "_add_b";
    layer_add_b->setName(layer_add_b_name.c_str());
    auto tensor_add_b = layer_add_b->getOutput(0);
///////////
    this->layer_map[node->index] = layer_add_b;

    this->SetRange(output_tensor, tensor_add_b);

    this->tensor_real_map[output_tensor->index] = tensor_add_b;
    this->tensor_swap_map[output_tensor->index] = output_tensor->index;

    return true;
}
