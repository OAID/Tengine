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

#ifndef __GRAPH_TASK_HPP__
#define __GRAPH_TASK_HPP__

#include <atomic>
#include <memory>
#include <condition_variable>
#include "dev_executor.hpp"
#include "graph_executor.hpp"
#include "exec_attr.hpp"

namespace TEngine {

class SubgraphTask;
class GenericEngine;
class GraphExecutor;

using SubgraphTaskPtr = std::shared_ptr<SubgraphTask>;

class GraphTask
{
    using graph_level_cb_t = std::function<void(SubgraphTask*)>;

    struct WaitEvent
    {
        std::mutex mutex;
        std::condition_variable cond;
        std::atomic<int> wait_count;
    };

public:
    GraphTask(GraphExecutor* graph_executor);
    ~GraphTask();

    void ReclaimSubgraphTask(void);

    bool OptimizeGraph(void);
    Graph* GetOptimizedGraph(void);

    bool Prerun(void);
    bool Run(exec_event_t& e);
    bool SyncRun();
    void Postrun(void);
    int Wait(exec_event_t& e, int try_wait);
    void SignalGraphTaskDone(void);

    int GetStatus(void)
    {
        return status_;
    }
    bool SetEventHook(int event, event_handler_t cb_func, void* cb_arg);

    void OnOutputSubgraphTaskDone(SubgraphTask*);
    void OnSubgraphTaskError(SubgraphTask*);
    void AddSubgraphTask(SubgraphTask*);
    void RemoveSubgraphTask(SubgraphTask*);
    bool RunSubgraphTask(SubgraphTask*);
    bool SyncRunSubgraphTask(SubgraphTask* sub_task);

    void SetEngine(GenericEngine* engine)
    {
        dev_engine_ = engine;
    }
    GenericEngine* GetEngine(void);

    GraphExecutor* GetGraphExecutor(void)
    {
        return graph_executor_;
    }
    Graph* GetGraph(void)
    {
        return graph_;
    }

    bool SetCallback(exec_event_t& e, int event, exec_cb_t cb);

    static Graph* MergeSubgraph(Graph* origin_graph, const std::vector<Subgraph*>& sub_list);

    const ExecAttr* GetExecAttr(void)
    {
        return p_exec_attr_;
    }

    bool SetAttr(const char* name, const void* val, int size);
    bool GetAttr(const char* name, void* val, int size);
    bool SetGraphPerfAttr(const char* name, const void* val, int size);
    bool GetGraphPerfAttr(const char* name, void* val, int size);

private:
    GraphExecutor* graph_executor_;
    Graph* graph_;
    std::vector<SubgraphTask*> sub_task_list_;
    std::atomic<unsigned int> output_wait_count_;
    std::atomic<unsigned int> active_sub_task_count_;
    int output_task_number_;
    int status_;
    GenericEngine* dev_engine_;
    WaitEvent wait_event_;
    bool task_done_;
    Graph* optimized_graph_;
    ExecAttr* p_exec_attr_;
};

class SubgraphTask
{
    friend GraphTask;

public:
    static void SetSubgraphTask(Subgraph* sub_graph, SubgraphTask* task);
    static SubgraphTask* GetSubgraphTask(Subgraph* sub_graph);

    SubgraphTask(Subgraph* sub_graph);
    ~SubgraphTask(void)
    {
        delete sub_graph;
    }

    Subgraph* sub_graph;
    GraphTask* graph_task;

    void* graph_handle;    // device related handle

    void OnSyncTaskDone(void);
    void OnTaskDone(bool exec_success);

    void OnInputNodeReady(Node*, bool);
    void OnNodeInputTensorReady(Node*, int port_index, bool);
    void OnOutputNodeDone(Node*, bool);

    void Init(GraphTask* graph_task);
    void Release(void);

    int GetStatus(void) const
    {
        return status_;
    }
    void SetStatus(int status)
    {
        status_ = status;
    }

    int exec_priority;

    bool is_output_task;
    bool attached;
    bool graph_optimized;

    DevExecutor* dev_executor;

    bool operator<(const SubgraphTask& other)
    {
        return exec_priority < other.exec_priority;
    }

    void Lock(void)
    {
        task_lock_.lock();
    }
    void Unlock(void)
    {
        task_lock_.unlock();
    }

    bool SetAttr(const char* name, const void* val, int size);
    bool GetAttr(const char* name, void* val, int size);

private:
    void SetNodeInputWaitMask(Node* node, uint64_t wait_mask);
    uint64_t GetNodeInputWaitMask(Node* node);
    std::atomic<uint64_t>* GetNodeInputWaitCounter(Node* node);
    void CreateNodeInputWaitCounter(Node* node);
    void ReleaseNodeInputWaitCounter(Node* node);

    std::atomic<unsigned int> input_wait_count_;
    int saved_input_wait_count_;
    int status_;
    std::mutex task_lock_;
};

}    // namespace TEngine

#endif
