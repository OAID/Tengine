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

#ifndef __TRT_TRT_EXECUTOR_HPP__
#define __TRT_TRT_EXECUTOR_HPP__

extern "C"
{
#include "tengine_ir.h"
}

#include <NvInfer.h>

#include <map>
#include <vector>


class TensorRTEngine
{
public:
    TensorRTEngine();
    ~TensorRTEngine() = default;
    int PreRun(struct subgraph* subgraph, int gpu_affinity, int mode);
    int Run(struct subgraph* subgraph);
    int PoseRun(struct subgraph* subgraph);

private:
    int Build(struct subgraph* subgraph);

    void SetRange(struct ir_graph* ir_graph, uint16_t id, nvinfer1::ITensor* trt_tensor);
    void SetRange(struct ir_tensor* ir_tensor, nvinfer1::ITensor* trt_tensor);

    bool check_if_input_in_map(uint16_t& id, std::map<uint16_t, uint16_t>& map);
    int get_type(int mode, nvinfer1::DataType& type);

private:
    uint16_t tensor_swap_count;

    std::map<uint16_t, nvinfer1::ITensor*> tensor_real_map;
    std::map<uint16_t, uint16_t> tensor_swap_map;

    std::map<uint16_t, nvinfer1::ILayer*> layer_map;

    std::vector<void*> io_tensors;

    std::vector<void*> host_buffer;

    nvinfer1::DataType precision;

private:
    bool AddTensor(struct ir_graph* ir_graph, struct ir_tensor* ir_tensor);
    bool AddBatchNormNode(struct ir_graph* ir_graph, struct ir_node* node);
    bool AddConcatNode(struct ir_graph* ir_graph, struct ir_node* node);
    bool AddConvolutionNode(struct ir_graph* ir_graph, struct ir_node* node);
    bool AddCropNode(struct ir_graph* ir_graph, struct ir_node* node);
    bool AddDropoutNode(struct ir_graph* ir_graph, struct ir_node* node);
    bool AddEltwiseLayer(struct ir_graph* ir_graph, struct ir_node* node);
    bool AddFlattenNode(struct ir_graph* ir_graph, struct ir_node* node);
    bool AddFullyConnectedNode(struct ir_graph* ir_graph, struct ir_node* node);
    bool AddInterpNode(struct ir_graph* ir_graph, struct ir_node* node);
    bool AddPermuteNode(struct ir_graph* ir_graph, struct ir_node* node);
    bool AddPoolingNode(struct ir_graph* ir_graph, struct ir_node* node);
    bool addReLUNode(struct ir_graph* ir_graph, struct ir_node* node);
    bool AddReshapeNode(struct ir_graph* ir_graph, struct ir_node* node);
    bool AddSliceNode(struct ir_graph* ir_graph, struct ir_node* node);
    bool AddSoftmaxNode(struct ir_graph* ir_graph, struct ir_node* node);
    bool AddUpsampleNode(struct ir_graph* ir_graph, struct ir_node* node);

private:
    nvinfer1::IBuilder* builder;
    nvinfer1::INetworkDefinition* network;
    nvinfer1::IBuilderConfig* config;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext *context;
};

#endif
