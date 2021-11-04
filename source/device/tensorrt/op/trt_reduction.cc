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
#include "reduction_param.h"
EXPORT_FINISH

#include <NvInferRuntime.h>


bool TensorRTEngine::AddReductionNode(struct graph* ir_graph, struct node* node)
{
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    if (nullptr == input_tensor || nullptr == output_tensor)
    {
        fprintf(stderr, "Tengine: Get input & output for Reduction(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    if (!check_if_input_in_map(input_tensor->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for Reduction(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    nvinfer1::ITensor* trt_tensor = tensor_real_map[tensor_swap_map[input_tensor->index]];

    reduction_param* param = (reduction_param*)node->op.param_mem;
    nvinfer1::ReduceOperation trt_op_type;
    switch (param->type)
    {
        case 0:
            trt_op_type = nvinfer1::ReduceOperation::kSUM;
            break;
        case 1:
            trt_op_type = nvinfer1::ReduceOperation::kAVG;
            break;
        default:
            fprintf(stderr, "Tengine: Reduction(id: %d) type(%d) was not supported.\n", node->index, param->type);
            return false;
    }

    uint32_t reduceAxes = 0;
    if (input_tensor->dim_num == 5)
    {
        reduceAxes = 2;
    }
    else
    {
        if (param->dim_0 != -2)
            reduceAxes = 1;
        else if (param->dim_1 == -2)
            reduceAxes = 2;
        else if (param->dim_2 == -2)
            reduceAxes = 3;
        else
            reduceAxes = 4;
    }

    if (input_tensor->dim_num == 5)
    {
        /* input reshape */
        nvinfer1::IShuffleLayer* layer_reshape_in = this->network->addShuffle(*trt_tensor);
        std::string layer_reshape_in_name = std::string(node->name) + "_reshape_in";
        layer_reshape_in->setName(layer_reshape_in_name.c_str());

        nvinfer1::Dims dims_in{};
        dims_in.nbDims = 4;
        for (int i = 0; i < 4; i++)
            dims_in.d[i] = 1;
        for (int i = 0; i < 3; i++)
            dims_in.d[i] = input_tensor->dims[i];
        dims_in.d[3] = input_tensor->dims[3] * input_tensor->dims[4];
        layer_reshape_in->setReshapeDimensions(dims_in);

        auto tensor_reshape_in = layer_reshape_in->getOutput(0);

        /* reduce */
        nvinfer1::IReduceLayer* layer_reduce = this->network->addReduce(*tensor_reshape_in, trt_op_type, reduceAxes, 0);
        std::string layer_reduce_name = std::string(node->name) + "_reduce";
        layer_reduce->setName(layer_reduce_name.c_str());

        auto tensor_reduce = layer_reduce->getOutput(0);

        /* output reshape */
        nvinfer1::IShuffleLayer* layer_reshape_out = this->network->addShuffle(*tensor_reduce);
        std::string layer_reshape_out_name = std::string(node->name) + "_reshape_out";
        layer_reshape_out->setName(layer_reshape_out_name.c_str());

        nvinfer1::Dims dims_out{};
        dims_out.nbDims = output_tensor->dim_num;
        for (int i = 0; i < output_tensor->dim_num; i++)
            dims_out.d[i] = 1;
        for (int i = 0; i < output_tensor->dim_num; i++)
            dims_out.d[i] = output_tensor->dims[i];
        layer_reshape_out->setReshapeDimensions(dims_out);

        auto tensor_reshape_out = layer_reshape_out->getOutput(0);

        this->layer_map[node->index] = layer_reshape_out;

        /* tensor map */
        this->SetRange(ir_graph, node->output_tensors[0], tensor_reshape_out);

        this->tensor_real_map[node->output_tensors[0]] = tensor_reshape_out;
        this->tensor_swap_map[node->output_tensors[0]] = node->output_tensors[0];
    }
    else
    {
        nvinfer1::IReduceLayer* layer = this->network->addReduce(*trt_tensor, trt_op_type, reduceAxes, 0);
        if (nullptr == layer)
        {
            fprintf(stderr, "Tengine: Add Reduction(id: %d, name: %s) layer failed.\n", node->index, node->name);
            return false;
        }

        layer->setName(node->name);

        this->layer_map[node->index] = layer;

        nvinfer1::ITensor* trt_output_tensor = layer->getOutput(0);

        this->SetRange(ir_graph, node->output_tensors[0], trt_output_tensor);

        this->tensor_real_map[node->output_tensors[0]] = trt_output_tensor;
        this->tensor_swap_map[node->output_tensors[0]] = node->output_tensors[0];
    }



    return true;
}
