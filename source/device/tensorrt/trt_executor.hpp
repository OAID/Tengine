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

#pragma once

#include "trt_define.h"

EXPORT_BEGIN
#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "device/device.h"
#include "utility/sys_port.h"
#include "utility/log.h"
EXPORT_FINISH

#include <NvInfer.h>

#include <map>
#include <vector>


class TensorRTEngine
{
public:
    TensorRTEngine();
    ~TensorRTEngine() = default;
    int PreRun(struct subgraph* subgraph, struct trt_option* opt);
    int Run(struct subgraph* subgraph);
    int PoseRun(struct subgraph* subgraph);
    int SetOption(trt_opt_t* opt);

private:
    int Build(struct subgraph* subgraph);

    void SetRange(struct graph* ir_graph, uint16_t id, nvinfer1::ITensor* trt_tensor);
    void SetRange(struct tensor* ir_tensor, nvinfer1::ITensor* trt_tensor);

    bool check_if_input_in_map(uint16_t& id, std::map<uint16_t, uint16_t>& map);
    int get_type(int mode, nvinfer1::DataType& type);

private:
    size_t   card_id;
    uint16_t tensor_swap_count;

    std::map<uint16_t, nvinfer1::ITensor*> tensor_real_map;
    std::map<uint16_t, uint16_t> tensor_swap_map;

    std::map<uint16_t, nvinfer1::ILayer*> layer_map;

    std::vector<void*> io_tensors;

    std::vector<void*> host_buffer;

    nvinfer1::DataType precision;

private:
    trt_opt_t option;

private:
    bool AddTensor(struct graph* ir_graph, struct tensor* ir_tensor);
    bool AddAbsVal(struct graph* ir_graph, struct node* node);
    bool AddAddN(struct graph* ir_graph, struct node* node);
    bool AddBatchNormNode(struct graph* ir_graph, struct node* node);
    bool AddConcatNode(struct graph* ir_graph, struct node* node);
    bool AddConvolutionNode(struct graph* ir_graph, struct node* node);
    bool AddDeConvolutionNode(struct graph* ir_graph, struct node* node);
    bool AddCropNode(struct graph* ir_graph, struct node* node);
    bool AddDropoutNode(struct graph* ir_graph, struct node* node);
    bool AddEltwiseLayer(struct graph* ir_graph, struct node* node);
    bool AddFlattenNode(struct graph* ir_graph, struct node* node);
    bool AddFullyConnectedNode(struct graph* ir_graph, struct node* node);
    bool AddHardSwishNode(struct graph* ir_graph, struct node* node);
    bool AddInterpNode(struct graph* ir_graph, struct node* node);
    bool AddMishNode(struct graph* ir_graph, struct node* node);
    bool AddPermuteNode(struct graph* ir_graph, struct node* node);
    bool AddPoolingNode(struct graph* ir_graph, struct node* node);
    bool addReLUNode(struct graph* ir_graph, struct node* node);
    bool AddReshapeNode(struct graph* ir_graph, struct node* node);
    bool AddTranspose(struct graph* ir_graph, struct node* node);
    bool AddSliceNode(struct graph* ir_graph, struct node* node);
    bool AddSoftmaxNode(struct graph* ir_graph, struct node* node);
    bool AddUpSampleNode(struct graph* ir_graph, struct node* node);

private:
    nvinfer1::IBuilder* builder;
    nvinfer1::INetworkDefinition* network;
    nvinfer1::IBuilderConfig* config;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext *context;
};
