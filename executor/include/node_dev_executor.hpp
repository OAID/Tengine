
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

#ifndef __NODE_DEV_EXECUTOR_HPP__
#define __NODE_DEV_EXECUTOR_HPP__

#include "generic_dev_executor.hpp"
#include "worker_thread.hpp"

namespace TEngine {

class NodeDevice;

class NodeExecutor : public GenericDevExecutor
{
public:
    struct NodeContext
    {
        void* dev_context;
        Subgraph* sub_graph;
        Subgraph* optimized_graph;
    };

    struct NodeTask
    {
        void* dev_context;
        Node* node;
    };

    NodeExecutor(const dev_id_t& dev_id)
    {
        dev_id_ = dev_id;
        worker_ = nullptr;
        create_worker_ = true;
    }

    virtual ~NodeExecutor() {}

    void DevGetWorkload(DevWorkload& load) override;
    bool DevGetPerf(Subgraph* graph, int policy, GraphPerf& perf) override;
    float DevGetFops(Subgraph* graph, int policy) override;
    int DevGetPolicyPriority(int policy) override;
    bool DevGetProposal(Graph* graph, int policy, bool static_assign) override;

    void* DevCreateGraphHandle(Subgraph* graph) override;
    bool DevOptimizeGraph(void* graph_handle) override;
    bool DevPrerun(void* graph_handle) override;
    bool DevRun(void* graph_handle) override;
    bool DevSyncRun(void* graph_handle) override;
    bool DevPostrun(void* graph_handle) override;
    bool DevSetGraphAttr(void*, const char*, const void*, int) override;
    bool DevGetGraphAttr(void*, const char*, void*, int) override;
    bool DevReleaseGraphHandle(void* graph_handle) override;
    bool DevStart(void) override;
    bool DevStop(void) override;

    Subgraph* DevGetOptimizedGraph(void* graph_handle) override;

    const dev_id_t& DevGetID(void) override;
    const dev_type_t& DevGetType(void) override;

    dev_status_t DevGetStatus(void) override;

    virtual bool Init(void) override;
    virtual bool Release(void) override;

    void UnbindDevice(void) override;
    void BindDevice(Device*) override;

    void OnNodeDone(NodeContext* context, Node* node, bool exec_success);

    void ProcessTask(const NodeTask& task);

    void DisableCreateWorker(void)
    {
        create_worker_ = false;
    }

protected:
    NodeDevice* backend_dev_;
    bool create_worker_;
    dev_id_t dev_id_;
    WorkerThread<NodeTask>* worker_;
    std::queue<NodeTask> task_queue_;
    std::mutex worker_lock_;
    std::condition_variable worker_cv_;
};

}    // namespace TEngine

#endif
