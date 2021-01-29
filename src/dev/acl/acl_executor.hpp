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

#ifndef __ACL_ACL_EXECUTOR_HPP__
#define __ACL_ACL_EXECUTOR_HPP__

extern "C"
{
#include "tengine_utils.h"
#include "tengine_ir.h"
#include "tengine_op.h"
#include "tengine_log.h"
}

#include <array>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>
#include <arm_neon.h>

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLFunctions.h"

#include "acl_helper.hpp"

#define DEFAULT_DEVICE_ID 0
#define DEFAULT_MAX_BATCH 128

#define MAX_TENGINE_DATA_TYPE_NUM 6
static const int gs32TengineDataElemetSize[MAX_TENGINE_DATA_TYPE_NUM] = {4, 2, 1, 1, 4, 2};

using namespace arm_compute;

#define USE_CPU_CONVERT
//#define ACL_EXTENSTION
#ifdef __ANDROID__
#define dynamic_cast static_cast
#endif

template <typename T>
inline void _PermuteDatalayoutNCHWToNHWCInter(T* pvData, int n, int c, int h, int w, T* pvOutputData);
inline void _PermuteDatalayoutNCHWToNHWC(void* pvData, int n, int c, int h, int w, void* pvOutputData, int DataEleSize);
inline void copy_buffer(void* dest, const void* src, const int src_len, DataType dest_type, DataType src_type);

class CLGraph
{
public:
    CLGraph();
    ~CLGraph();

    void init(std::string name, DataType type);
    int prerun(struct subgraph *subgraph, int cpu_affinity, int mode);
    int run(struct subgraph *subgraph);
    int postrun(struct subgraph *subgraph);


private:
    bool CreateACLGraph(struct subgraph* subgraph, DataType type, bool bDataLayoutOpFlag = false);

    bool AddBNLayer(struct ir_node* node, struct ir_node* node_scale);
    bool AddConvolutionLayer(struct ir_node* node);
    bool AddConcatLayer(struct ir_node* node);
    bool AddDropoutLayer(struct ir_node* node);
    bool AddEltwiseLayer(struct ir_node* node);
    bool AddFCLayer(struct ir_node* node);
    bool AddInputLayer(struct ir_node* node);
    bool AddPoolingLayer(struct ir_node* node);
    bool AddReLuLayer(struct ir_node* node);
    bool AddReLu6Layer(struct ir_node* node);
    bool AddResizeLayer(struct ir_node* node);
    bool AddSoftmaxLayer(struct ir_node* node);

    CLTensor* GetCLTensor(std::string name);

public:
    std::string name_;
    std::vector<IFunction*> functions_map_;
    std::unordered_map<std::string, CLTensor*> tensors_map_;
    DataType data_type_;

    int nCnt_ = 0;
    bool bForcedNHWCMode_;
    char* pcScratchMem_;
    int l32ScratchMemSize_;
    bool l32AclNHWCOptimizeFlag_;
};

#endif
