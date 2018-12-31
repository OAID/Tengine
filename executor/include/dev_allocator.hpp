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
 * Author: haitao@openailab.com
 */

#ifndef __DEV_ALLOCATOR_HPP__
#define __DEV_ALLOCATOR_HPP__

#include <string>
#include <vector>
#include <mutex>

#include "simple_object_manager.hpp"

namespace TEngine {

class GenericEngine;
class GraphExecutor;
class Graph;
class GraphTask;
struct DevExecutor;

struct DevAllocator
{
    virtual bool Allocate(GenericEngine* engine, GraphExecutor* graph_executor, Graph* graph,
                          std::vector<Subgraph*>& sub_list) = 0;
    virtual const std::string& GetName(void) = 0;
    virtual ~DevAllocator(){};
    std::string name;

    void SameGraph(TEngine::Graph*, TEngine::DevExecutor*, std::vector<TEngine::Graph*>&);
    void PartitionGraph(TEngine::GenericEngine*, TEngine::GraphExecutor*, TEngine::Graph*,
                        std::vector<TEngine::Graph*>&);
};

class DevAllocatorManager : public SimpleObjectManager<DevAllocatorManager, DevAllocator*>
{
public:
    static void OnDevExecutorRegistered(DevExecutor* dev_executor);
    static void OnDevExecutorUnregistered(DevExecutor* dev_executor);

    static void LockExecutorList(void);
    static void UnlockExecutorList(void);

    std::mutex list_lock;
    std::vector<DevExecutor*> executor_list;
};

}    // namespace TEngine

#endif
