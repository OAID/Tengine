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
#include "concat_param.h"
EXPORT_FINISH

#include <NvInferRuntime.h>


bool TensorRTEngine::AddConcatNode(struct graph* ir_graph, struct node* node)
{
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    std::vector<nvinfer1::ITensor *> input_trt_tensor_list;

    for (uint8_t i = 0; i < node->input_num; i++)
    {
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[i]);
        if (nullptr != ir_tensor && !check_if_input_in_map(ir_tensor->index, this->tensor_swap_map))
        {
            fprintf(stderr, "Tengine: Query input for Concat(id: %d, name: %s) layer failed.\n", node->index, node->name);
            return false;
        }
        input_trt_tensor_list.push_back(tensor_real_map[tensor_swap_map[node->input_tensors[i]]]);
    }

    nvinfer1::IConcatenationLayer* layer = this->network->addConcatenation(input_trt_tensor_list.data(), input_trt_tensor_list.size());
    if (nullptr == layer)
    {
        fprintf(stderr, "Tengine: Add Concat(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    layer->setName(node->name);

    this->layer_map[node->index] = layer;

    /* add output tensor into map */
    nvinfer1::ITensor* concat_tensor = layer->getOutput(0);

    this->SetRange(output_tensor, concat_tensor);

    concat_param* param = (concat_param*)node->op.param_mem;

    layer->setAxis(param->axis);

    this->tensor_real_map[output_tensor->index] = concat_tensor;
    this->tensor_swap_map[output_tensor->index] = output_tensor->index;

    return true;
}
