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
#include "relu_param.h"
#include "convolution_param.h"
#include "clip_param.h"
EXPORT_FINISH

#include <cmath>
#include <NvInferRuntime.h>


bool TensorRTEngine::addReLUNode(struct graph *ir_graph, struct node *node)
{
    int op_type = OP_RELU;

    struct tensor* relu_input = nullptr;

    bool need_change_name = false;

    if (OP_CONV == node->op.type)
    {
        relu_input = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

        if (nullptr == relu_input)
        {
            fprintf(stderr, "Tengine: Get input for ReLU(id: %d, name: %s) layer failed.\n", node->index, node->name);
            return false;
        }

        auto param = (struct conv_param*)node->op.param_mem;

        switch (param->activation)
        {
            case 0:
                op_type = OP_RELU;
                break;
            case 1:
                op_type = OP_RELU1;
                break;
            case 6:
                op_type = OP_CLIP;
                break;
            default:
                fprintf(stderr, "Tengine: Unsupported RelU type(%d).\n", param->activation);
                return false;
        }

        need_change_name = true;
    }
    else
    {
        relu_input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);

        switch (node->op.type)
        {
            case OP_RELU:
                op_type = OP_RELU;
                break;
            case OP_RELU1:
                op_type = OP_RELU1;
                break;
            case OP_RELU6:
                op_type = OP_RELU6;
                break;
            case OP_CLIP:
                op_type = OP_CLIP;
                break;
            default:
                fprintf(stderr, "Tengine: Unsupported RelU type(%d).\n", node->op.type);
                return false;
        }
    }

    nvinfer1::ITensor* trt_tensor = tensor_real_map[relu_input->index];

    nvinfer1::IActivationLayer * layer = nullptr;

    if (OP_RELU == op_type)
    {
        if (!need_change_name)
        {
            layer = this->network->addActivation(*trt_tensor,nvinfer1::ActivationType::kLEAKY_RELU);
            if (nullptr == layer)
            {
                fprintf(stderr, "Tengine: Add ReLU(id: %d, name: %s) layer failed.\n", node->index, node->name);
                return false;
            }

            layer->setName(node->name);

            auto* param = (relu_param*)node->op.param_mem;
            if (nullptr != param && std::fabs(param->negative_slope) > 0.000001)
            {
                layer->setAlpha(param->negative_slope);
            }
        }
        else
        {
            layer = this->network->addActivation(*trt_tensor,nvinfer1::ActivationType::kRELU);
            if (nullptr == layer)
            {
                fprintf(stderr, "Tengine: Add ReLU(id: %d, name: %s) layer failed.\n", node->index, node->name);
                return false;
            }
        }

        this->layer_map[node->index] = layer;
    }
    else
    {
        layer = this->network->addActivation(*trt_tensor,nvinfer1::ActivationType::kCLIP);
        if (nullptr == layer)
        {
            fprintf(stderr, "Tengine: Add ReLU(id: %d, name: %s) layer failed.\n", node->index, node->name);
            return false;
        }

        if (!need_change_name)
        {
            layer->setName(node->name);
        }

        this->layer_map[node->index] = layer;

        if (OP_RELU1 == op_type)
        {
            layer->setAlpha(0);
            layer->setBeta(1);
        }
        if (OP_RELU6 == op_type)
        {
            layer->setAlpha(0);
            layer->setBeta(6);
        }
        if (OP_CLIP == op_type)
        {
            auto clip_param = (struct clip_param*)node->op.param_mem;

            layer->setAlpha(clip_param->min);
            layer->setBeta(clip_param->max);
        }
    }

    trt_tensor = layer->getOutput(0);

    /*if (need_change_name)
    {
        struct tensor* node_output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

        std::string tensor_name = std::string(node_output->name) + std::to_string(node_output->index);
        trt_tensor->setName(tensor_name.c_str());
    }*/

    this->SetRange(ir_graph, node->output_tensors[0], trt_tensor);

    if (OP_CONV == node->op.type)
    {
        this->tensor_real_map[relu_input->index + tensor_swap_count] = trt_tensor;
        this->tensor_swap_map[relu_input->index] = relu_input->index + tensor_swap_count;
    }
    else
    {
        struct tensor* relu_output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

        this->tensor_real_map[relu_output->index] = trt_tensor;
        this->tensor_swap_map[relu_output->index] = relu_output->index;
    }

    return true;
}
