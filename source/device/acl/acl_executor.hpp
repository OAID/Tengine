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

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLFunctions.h"

#include <array>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>

#include <arm_neon.h>

extern "C" {
#include "api/c_api.h"
#include "device/device.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "executer/executer.h"
#include "optimizer/split.h"
#include "module/module.h"
#include "utility/vector.h"
#include "utility/log.h"
}

#define MAX_TENGINE_DATA_TYPE_NUM 6
static const int gs32TengineDataElemetSize[MAX_TENGINE_DATA_TYPE_NUM] = {4, 2, 1, 1, 4, 2};

using namespace arm_compute;

#define USE_CPU_CONVERT
//#define ACL_EXTENSTION
#ifdef __ANDROID__
#define dynamic_cast static_cast
#endif

template<typename T>
inline void _PermuteDataLayoutNCHWToNHWCInter(T* pvData, int n, int c, int h, int w, T* pvOutputData);
void _PermuteDataLayoutNCHWToNHWC(void* pvData, int n, int c, int h, int w, void* pvOutputData, int DataEleSize);
void copy_buffer(void* dest, const void* src, const int src_len, DataType dest_type, DataType src_type);

class CLGraph
{
public:
    CLGraph();
    ~CLGraph();

    void init(std::string name, DataType type);
    int prerun(struct subgraph* subgraph, struct acl_option* option);
    int run(struct subgraph* subgraph);
    int postrun(struct subgraph* subgraph);

private:
    bool CreateACLGraph(struct subgraph* subgraph, DataType type, bool bDataLayoutOpFlag = false);

    bool AddBNLayer(struct node* node, struct node* node_scale);
    bool AddCastLayer(struct node* node);
    bool AddConcatLayer(struct node* node);
    bool AddConvolutionLayer(struct node* node);
    bool AddCropLayer(struct node* node);
    bool AddDropoutLayer(struct node* node);
    bool AddEltwiseLayer(struct node* node);
    bool AddFCLayer(struct node* node);
    bool AddInputLayer(struct node* node);
    bool AddInterpLayer(struct node* node);
    bool AddPoolingLayer(struct node* node);
    bool AddReLuLayer(struct node* node);
    bool AddReLu6Layer(struct node* node);
    bool AddReshapeLayer(struct node* node);
    bool AddResizeLayer(struct node* node);
    bool AddSoftmaxLayer(struct node* node);

    CLTensor* GetCLTensor(std::string name);

public:
    std::string name_;
    std::vector<std::shared_ptr<IFunction> > functions_map_;
    std::unordered_map<std::string, CLTensor*> tensors_map_;
    DataType data_type_;

    int nCnt_ = 0;
    bool bForcedNHWCMode_;
    char* pcScratchMem_;
    int l32ScratchMemSize_;
    bool l32AclNHWCOptimizeFlag_;
};
