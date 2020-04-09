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

#ifndef __GENERIC_DEV_EXECUTOR_HPP_
#define __GENERIC_DEV_EXECUTOR_HPP_

#include <map>
#include <set>
#include <vector>

#include "dev_executor.hpp"

namespace TEngine {

class GenericDevExecutor : public DevExecutor
{
public:
    using task_queue_t = std::map<int, std::set<SubgraphTask*>>;

    enum QueueType
    {
        kWaitQueue,
        kReadyQueue,
        kRunQueue
    };

    bool OptimizeGraph(SubgraphTask* task) override;
    Subgraph* GetOptimizedGraph(SubgraphTask* task) override;

    bool PrerunTask(SubgraphTask* task) override;
    bool SchedTask(SubgraphTask* task) override;
    bool SchedTask(void) override;
    bool RunTask(SubgraphTask* task) override;
    bool SyncRunTask(SubgraphTask* task) override;
    bool PostrunTask(SubgraphTask* task) override;

    bool SetGraphAttr(SubgraphTask* task, const char* name, const void* val, int size) override;
    bool GetGraphAttr(SubgraphTask* task, const char* name, void* val, int size) override;

    bool GetProposal(Graph* graph, int policy, bool static_assign) override
    {
        return DevGetProposal(graph, policy, static_assign);
    }

    int GetRunTaskNum(void) override;
    int GetReadyTaskNum(void) override;
    int GetWaitTaskNum(void) override;

    void OnSubgraphDone(Subgraph* sub_graph, bool exec_success);

    const std::string& GetName(void) override
    {
        return name_;
    }
    const dev_id_t& GetDevID(void) override
    {
        return DevGetID();
    }
    const dev_type_t& GetDevType(void) override
    {
        return DevGetType();
    }
    dev_status_t GetStatus(void) override
    {
        return DevGetStatus();
    }

    void GetWorkload(DevWorkload& load) override
    {
        DevGetWorkload(load);
    }

    bool GetPerf(Subgraph* graph, int policy, GraphPerf& perf) override
    {
        return DevGetPerf(graph, policy, perf);
    }

    float GetFops(Subgraph* graph, int policy) override
    {
        return DevGetFops(graph, policy);
    }

    int GetPolicyPriority(int policy) override
    {
        return DevGetPolicyPriority(policy);
    }

    bool Start(void) override
    {
        return DevStart();
    }
    bool Stop(void) override
    {
        return DevStop();
    }

    void SetName(const std::string& name) override
    {
        name_ = name;
    }

    virtual void DevGetWorkload(DevWorkload& load) = 0;
    virtual bool DevGetPerf(Subgraph* graph, int policy, GraphPerf& perf) = 0;
    virtual float DevGetFops(Subgraph* graph, int policy) = 0;
    virtual int DevGetPolicyPriority(int policy) = 0;
    virtual bool DevGetProposal(Graph* graph, int policy, bool static_assign)
    {
        return true;
    }

    virtual void* DevCreateGraphHandle(Subgraph* graph) = 0;
    virtual bool DevOptimizeGraph(void* graph_handle) = 0;
    virtual bool DevPrerun(void* graph_handle) = 0;
    virtual bool DevRun(void* graph_handle) = 0;
    virtual bool DevSyncRun(void* graph_handle) = 0;
    virtual bool DevPostrun(void* graph_handle) = 0;
    virtual bool DevSetGraphAttr(void* graph_handle, const char* name, const void* val, int size) = 0;
    virtual bool DevGetGraphAttr(void* graph_handle, const char* name, void* val, int size) = 0;
    virtual bool DevReleaseGraphHandle(void* graph_handle) = 0;
    virtual dev_status_t DevGetStatus(void) = 0;
    virtual const dev_id_t& DevGetID(void) = 0;
    virtual const dev_type_t& DevGetType(void) = 0;
    virtual bool DevStart(void) = 0;
    virtual bool DevStop(void) = 0;
    virtual Subgraph* DevGetOptimizedGraph(void* graph_handle)
    {
        return nullptr;
    }

    virtual bool DevGetMemorySize(void* graph_handle, unsigned int& mem_size)
    {
        return false;
    }
    virtual void DevSetMemory(void* graph_handle, void* mem_addr){};

    virtual ~GenericDevExecutor() {}

    void Lock(std::mutex& mutex)
    {
        mutex.lock();
    }
    void Unlock(std::mutex& mutex)
    {
        mutex.unlock();
    }

protected:
    void InsertQueue(QueueType queue_type, SubgraphTask* task);
    bool RemoveQueue(QueueType queue_type, SubgraphTask* task);
    void InsertQueue(SubgraphTask* task);
    bool RemoveQueue(SubgraphTask* task);
    SubgraphTask* PopQueue(QueueType queue_type);
    int GetElementNumber(QueueType queue_type);
    bool GetQueueReference(QueueType queue_type, std::mutex*& p_mutex, task_queue_t*& p_queue);

    std::mutex run_queue_lock_;
    std::mutex ready_queue_lock_;
    std::mutex wait_queue_lock_;

    task_queue_t wait_queue_;
    task_queue_t run_queue_;
    task_queue_t ready_queue_;

    std::string name_;
};

}    // namespace TEngine

#endif
