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
#include "upsample_param.h"
EXPORT_FINISH

#include <NvInferRuntime.h>


bool TensorRTEngine::AddUpSampleNode(struct graph* ir_graph, struct node* node)
{
    struct tensor* upsample_input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* upsample_output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    if (nullptr == upsample_input || nullptr == upsample_output)
    {
        fprintf(stderr, "Tengine: Get input & output for Interp(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    if (!check_if_input_in_map(upsample_input->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for Interp(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    nvinfer1::ITensor* upsample_input_tensor = tensor_real_map[tensor_swap_map[upsample_input->index]];

    nvinfer1::IResizeLayer* layer = this->network->addResize(*upsample_input_tensor);
    if (nullptr == layer)
    {
        fprintf(stderr, "Tengine: Add Flatten(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    auto* param = (struct upsample_param*)node->op.param_mem;

    std::vector<float> scale_factors(upsample_input->dim_num, 1.0f);
    scale_factors[2] = param->scale;
    scale_factors[3] = param->scale;
    layer->setScales(scale_factors.data(), upsample_input->dim_num);

    nvinfer1::ResizeMode resizeMode = nvinfer1::ResizeMode::kLINEAR;
    layer->setResizeMode(resizeMode);

    layer->setName(node->name);

    this->layer_map[node->index] = layer;

    nvinfer1::ITensor* upsample_output_tensor = layer->getOutput(0);

    this->SetRange(upsample_output, upsample_output_tensor);

    this->tensor_real_map[node->output_tensors[0]] = upsample_output_tensor;
    this->tensor_swap_map[node->output_tensors[0]] = node->output_tensors[0];

    return true;
}
