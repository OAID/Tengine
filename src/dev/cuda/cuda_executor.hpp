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

#include "cuda_helper.hpp"

#include <map>
#include <vector>
#include <functional>

#include <cuda.h>
#include <cudnn.h>
#include "cublas_v2.h"

extern "C"
{
#include "../src/dev/cpu/cpu_device.h"
#include "../src/dev/cpu/cpu_node_ops.h"
#include "nn_device.h"
#include "tengine_ir.h"
#include "tengine_op.h"
#include "tengine_log.h"
}

#define DEFAULT_DEVICE_ID 0

typedef std::map<uint32_t, uint32_t> dict_uint2uint;
typedef std::map<uint32_t, void*> dict_uint2voidx;
typedef std::function< void() >  GPU_kernel;

class CUDAEngine
{
public:
    CUDAEngine();
    ~CUDAEngine() = default;

    int CUDAEnginePreRun(struct subgraph* subgraph);
    int CUDAEngineRun(struct subgraph* subgraph);
    void CUDAEnginePostRun();

private:
    void AddClipNode(struct ir_graph* ir_graph, struct ir_node* ir_node);
    void AddConcatNode(struct ir_graph* ir_graph, struct ir_node* ir_node);
    void AddConvolutionNode(struct ir_graph* ir_graph, struct ir_node* ir_node);
    void AddDropoutNode(struct ir_graph* ir_graph, struct ir_node* ir_node);
    void AddEltwiseNode(struct ir_graph* ir_graph, struct ir_node* ir_node);
    void AddFullyConnectionNode(struct ir_graph* ir_graph, struct ir_node* ir_node);
    void AddFlattenNode(struct ir_graph* ir_graph, struct ir_node* ir_node);
    void AddPermuteNode(struct ir_graph* ir_graph, struct ir_node* ir_node);
    void AddPoolingNode(struct ir_graph* ir_graph, struct ir_node* ir_node);
    void AddReluNode(struct ir_graph* ir_graph, struct ir_node* ir_node);
    void AddReshapeNode(struct ir_graph* ir_graph, struct ir_node* ir_node);
    void AddSliceNode(struct ir_graph* ir_graph, struct ir_node* ir_node);
    void AddSoftmaxNode(struct ir_graph* ir_graph, struct ir_node* ir_node);

private:
    void CUDADataMalloc(struct ir_graph* ir_graph, int ir_tensor_idx);
    int Build(struct subgraph* subgraph);
    void DataUpload(struct ir_graph* ir_graph, int ir_tensor_idx);
    void DataDownload(struct ir_graph* ir_graph, int ir_tensor_idx);

private:
    std::vector< GPU_kernel > ops;

private:
    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;
    cudnnConvolutionFwdAlgo_t algo1;

public:
    dict_uint2voidx     gpu_addr_map;
};
