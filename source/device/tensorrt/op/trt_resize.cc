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
#include "resize_param.h"
EXPORT_FINISH

#include <NvInferRuntime.h>


bool TensorRTEngine::AddResizeNode(struct graph* ir_graph, struct node* node)
{
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    if (nullptr == input_tensor || nullptr == output_tensor)
    {
        fprintf(stderr, "Tengine: Get input & output for Resize(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    if (!check_if_input_in_map(input_tensor->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for Resize(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    struct resize_param* param = (struct resize_param*)node->op.param_mem;

    nvinfer1::ResizeMode resizeMode = nvinfer1::ResizeMode::kLINEAR;
    if (param->type == 0)
    {
        resizeMode = nvinfer1::ResizeMode::kNEAREST;
    }
    else if(param->type == 1)
    {
        resizeMode = nvinfer1::ResizeMode::kLINEAR;
    }

    nvinfer1::ITensor* interp_input_tensor = tensor_real_map[tensor_swap_map[input_tensor->index]];

    nvinfer1::IResizeLayer* layer = this->network->addResize(*interp_input_tensor);
    if (nullptr == layer)
    {
        fprintf(stderr, "Tengine: Add Resize(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    std::vector<float> scale_factors(input_tensor->dim_num, 1.0f);
    scale_factors[2] = param->scale_h;
    scale_factors[3] = param->scale_w;
    layer->setScales(scale_factors.data(), input_tensor->dim_num);

    layer->setResizeMode(resizeMode);

    layer->setName(node->name);

    this->layer_map[node->index] = layer;

    nvinfer1::ITensor* interp_output_tensor = layer->getOutput(0);

    this->SetRange(output_tensor, interp_output_tensor);

    this->tensor_real_map[node->output_tensors[0]] = interp_output_tensor;
    this->tensor_swap_map[node->output_tensors[0]] = node->output_tensors[0];

    return true;
}
