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
#include "pooling_param.h"
EXPORT_FINISH


bool TensorRTEngine::AddPoolingNode(struct graph *ir_graph, struct node *node)
{
    struct tensor* pool_input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* pool_output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    if (nullptr == pool_input || nullptr == pool_output)
    {
        fprintf(stderr, "Tengine: Get input & output for Pooling(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    if (!check_if_input_in_map(pool_input->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for Pooling(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    nvinfer1::ITensor* trt_tensor = tensor_real_map[tensor_swap_map[pool_input->index]];

    pool_param* param = (pool_param*)node->op.param_mem;

    nvinfer1::DimsHW kernel_size{ param->kernel_h, param->kernel_w };

    nvinfer1::PoolingType pooling_type;

    if(0 == param->pool_method)
        pooling_type=nvinfer1::PoolingType::kMAX;
    else
        pooling_type=nvinfer1::PoolingType::kAVERAGE;

    nvinfer1::IPoolingLayer* layer = this->network->addPooling(*trt_tensor, pooling_type, kernel_size);
    if (nullptr == layer)
    {
        fprintf(stderr, "Tengine: Add Pooling(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    layer->setName(node->name);

    layer->setStride(nvinfer1::DimsHW{param->stride_h, param->stride_w});
    layer->setAverageCountExcludesPadding(false);
    layer->setPrePadding(nvinfer1::DimsHW(param->pad_h0, param->pad_w0));
    layer->setPostPadding(nvinfer1::DimsHW(param->pad_h1, param->pad_w1));

    this->layer_map[node->index] = layer;

    trt_tensor = layer->getOutput(0);

    this->SetRange(pool_output, trt_tensor);

    this->tensor_real_map[pool_output->index] = trt_tensor;
    this->tensor_swap_map[pool_output->index] = pool_output->index;

    return true;
}
