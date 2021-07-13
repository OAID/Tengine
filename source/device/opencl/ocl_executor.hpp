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


extern "C"
{
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

#include <map>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

#include <CL/cl.h>

#include <cmath>

typedef std::map<uint32_t, cl_mem> dict_uint2clmem;

struct OCLqueue
{
    std::string name;
    int         dims;
    cl_kernel   queue_kernel;
    cl_event    enentPoint;
    size_t*     queue_global_work_size;
    size_t*     queue_local_work_size;
};

class OCLEngine {
public:
    //    OCLEngine();
    //    ~OCLEngine() = default;

    int  OCLEnginePreRun(struct subgraph* subgraph);
    int  OCLEngineRun(struct subgraph* subgraph);
    void OCLEnginePostRun();

private:
    bool init();
    bool build_kernel(const char* filename, const char* kernel_name);
    bool OCLTensorMap(struct graph* ir_graph, int ir_tensor_idx, cl_mem_flags flag);
    int  BuildTensor(struct subgraph* subgraph);
    int  BuildKernel(struct subgraph* subgraph);


    bool AddClipNode(struct node* ir_node);
    bool AddConcatNode(struct node* ir_node);
    bool AddConvolutionNode(struct node* ir_node);
    bool AddDropoutNode(struct node* ir_node);
    bool AddEltwiseNode(struct node* ir_node);
    bool AddFlattenNode(struct node* ir_node);
    bool AddFullyConnectionNode(struct node* node);
    bool AddPoolingNode(struct node* ir_node);
    bool AddReluNode(struct node* ir_node);
    bool AddReshapeNode(struct node* ir_node);
    bool AddSliceNode(struct node* ir_node);


private:
    cl_int           status;
    cl_platform_id   platform;
    cl_device_id*    devices;
    cl_context       context;
    cl_command_queue commandQueue;

    cl_program program;
    cl_kernel  kernel;

public:
    dict_uint2clmem              ocl_tensor_map;
    std::vector<struct OCLqueue> queue_list;

public:
    int bin_num;
};
