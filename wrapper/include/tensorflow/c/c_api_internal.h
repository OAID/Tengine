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
 * Copyright (c) 2018, Open AI Lab
 * Author: jingyou@openailab.com
 */
#ifndef __TENSORFLOW_C_C_API_INTERNAL_H__
#define __TENSORFLOW_C_C_API_INTERNAL_H__

#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/status.h"
#include "tengine_c_api.h"

struct TF_Status
{
    tensorflow::Status status;
};

struct TF_Tensor
{
    int dtype;
    const char* name;    // tensor name
    std::vector<int> shape;
    void* data;
    int count;
};

struct TF_Operation
{
    const char* node_name;    // node name
};

struct TF_Graph
{
    TF_Graph() : prerun_already(false), graph_exe(nullptr){};
    ~TF_Graph();

    bool prerun_already;
    graph_t graph_exe;    // pointer to Tengine graph executor
};

struct TF_Session
{
    TF_Graph* graph;
};

struct TF_ImportGraphDefOptions
{
    const char* prefix;
};

struct TF_SessionOptions
{
    const char* target;
};

#endif    // __TENSORFLOW_C_C_API_INTERNAL_H__
