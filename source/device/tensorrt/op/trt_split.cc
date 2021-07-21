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
#include "split_param.h"
#include "utility/vector.h"
EXPORT_FINISH

#include <NvInferRuntime.h>


bool TensorRTEngine::AddSplitNode(struct graph* ir_graph, struct node* node)
{
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);

    if (nullptr == input_tensor)
    {
        fprintf(stderr, "Tengine: Get input & output for Split(id: %d, name: %s) layer failed.\n", node->index,
                node->name);
        return false;
    }

    if (!check_if_input_in_map(input_tensor->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for Split(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    auto* param = ( struct split_param* )node->op.param_mem;

    std::vector<uint32_t> slices;
    uint32_t tmp_num = 0;
    slices.push_back( tmp_num );
    for (int p = 0; p < node->output_num; p++)
    {
        uint32_t split_slice = ((uint32_t*)get_vector_data(param->split_sizes_, p))[0];
        tmp_num += split_slice;
        slices.push_back( tmp_num );
    }

    for (int p = 0; p < node->output_num; p++)
    {
        struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, node->output_tensors[p]);

        nvinfer1::Dims start, size, stride;

        start.nbDims = input_tensor->dim_num;
        size.nbDims = input_tensor->dim_num;
        stride.nbDims = input_tensor->dim_num;

        if (0 != param->axis)
        {
            for (uint8_t i = 0; i < input_tensor->dim_num; i++)
            {
                start.d[i] = 0;
                size.d[i] = output_tensor->dims[i];
                stride.d[i] = 1;
            }

            start.d[param->axis] = slices[p];
            size.d[param->axis] = slices[p+1] - slices[p];
        }
        else
        {
            fprintf(stderr, "Tengine: Split(id: %d, name: %s) in batch is not supported for now.\n", node->index, node->name);
            return false;
        }

        nvinfer1::ITensor* trt_tensor = tensor_real_map[tensor_swap_map[input_tensor->index]];

        nvinfer1::ISliceLayer* layer = this->network->addSlice(*trt_tensor, start, size, stride);
        if (nullptr == layer)
        {
            fprintf(stderr, "Tengine: Add Split(id: %d, name: %s) layer failed.\n", node->index, node->name);
            return false;
        }

        std::string layer_name = std::string(node->name) + std::to_string(p);
        layer->setName(layer_name.c_str());

        this->layer_map[node->index + p * ir_graph->node_num] = layer;

        nvinfer1::ITensor* slice_output_tensor = layer->getOutput(0);

        this->SetRange(output_tensor, slice_output_tensor);

        this->tensor_real_map[node->output_tensors[p]] = slice_output_tensor;
        this->tensor_swap_map[node->output_tensors[p]] = node->output_tensors[p];
    }

    return true;
}
