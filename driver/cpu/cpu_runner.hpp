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
#ifndef __CPU_RUNNER_HPP__
#define __CPU_RUNNER_HPP__

#include <string>
#include <vector>
#include <functional>

#include "node_ops.hpp"
#include "graph_perf.hpp"

namespace TEngine {

class Graph;
class CPUDevice;

using Subgraph = Graph;

class CPURunner
{
public:
    bool SetGraphPerfStat(Subgraph* graph, int action);
    int GetGraphPerfStat(Subgraph* graph, struct perf_info** buf, int buf_size);

    bool Prerun(Subgraph* sub_graph);
    bool Run(Subgraph* sub_graph);
    bool Postrun(Subgraph* sub_graph);

    bool OptimizeGraph(Subgraph* sub_graph);

    void AttachCPUDevice(CPUDevice* cpu_dev);

    bool BindNodeOps(Subgraph* graph);
    bool AllocateMem(Subgraph* graph);

    bool FreeMem(Subgraph* graph);
    bool UnbindNodeOps(Subgraph* graph);

    NodeOps* BindCustomKernel(Node* node);

    CPURunner()
    {
        mem_alloc = malloc;
        mem_free = free;
    }

    ~CPURunner() {}

    mem_alloc_t mem_alloc;
    mem_free_t mem_free;
    CPUDevice* cpu_dev_;
    const CPUInfo* cpu_info_;
};

}    // namespace TEngine

#endif
