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
 * Author: hhchen@openailab.com
 */

#pragma once

#include <map>
#include <vector>
#include <functional>
#include <cstdio>

extern "C" {
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
}

#include <cublas_v2.h>
#include <cudnn.h>

#define DEFAULT_DEVICE_ID 0

typedef std::map<uint32_t, uint32_t> dict_uint2uint;
typedef std::map<uint32_t, void*> dict_uint2voidx;
typedef std::function<void()> GPU_kernel;

class CUDAEngine
{
public:
    CUDAEngine();
    ~CUDAEngine() = default;

    int CUDAEnginePreRun(struct subgraph* subgraph);
    int CUDAEngineRun(struct subgraph* subgraph);
    void CUDAEnginePostRun();

private:
    void AddClipNode(struct graph* ir_graph, struct node* ir_node);
    void AddConcatNode(struct graph* ir_graph, struct node* ir_node);
    void AddConvolutionNode(struct graph* ir_graph, struct node* ir_node);
    void AddDropoutNode(struct graph* ir_graph, struct node* ir_node);
    void AddEltwiseNode(struct graph* ir_graph, struct node* ir_node);
    void AddFullyConnectionNode(struct graph* ir_graph, struct node* ir_node);
    void AddFlattenNode(struct graph* ir_graph, struct node* ir_node);
    void AddPermuteNode(struct graph* ir_graph, struct node* ir_node);
    void AddPoolingNode(struct graph* ir_graph, struct node* ir_node);
    void AddReluNode(struct graph* ir_graph, struct node* ir_node);
    void AddReshapeNode(struct graph* ir_graph, struct node* ir_node);
    void AddSliceNode(struct graph* ir_graph, struct node* ir_node);
    void AddSoftmaxNode(struct graph* ir_graph, struct node* ir_node);

private:
    void CUDADataMalloc(struct graph* ir_graph, int ir_tensor_idx);
    int Build(struct subgraph* subgraph);
    void DataUpload(struct graph* ir_graph, int ir_tensor_idx);
    void DataDownload(struct graph* ir_graph, int ir_tensor_idx);

private:
    std::vector<GPU_kernel> ops;

private:
    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;
    cudnnConvolutionFwdAlgo_t algo1;

public:
    dict_uint2voidx gpu_addr_map;
};
