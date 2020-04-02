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

#include "graph_task.hpp"
#include "generic_dev_executor.hpp"

namespace TEngine {

bool GenericDevExecutor::GetQueueReference(GenericDevExecutor::QueueType queue_type, std::mutex*& p_mutex,
                                           task_queue_t*& p_queue)
{
    bool ret = true;

    if(queue_type == kWaitQueue)
    {
        p_mutex = &wait_queue_lock_;
        p_queue = &wait_queue_;
    }
    else if(queue_type == kReadyQueue)
    {
        p_mutex = &ready_queue_lock_;
        p_queue = &ready_queue_;
    }
    else if(queue_type == kRunQueue)
    {
        p_mutex = &run_queue_lock_;
        p_queue = &run_queue_;
    }
    else
        ret = false;
    return ret;
}

void GenericDevExecutor::InsertQueue(GenericDevExecutor::QueueType queue_type, SubgraphTask* task)
{
    std::mutex* p_mutex = nullptr;
    task_queue_t* p_queue = nullptr;

    if(!GetQueueReference(queue_type, p_mutex, p_queue))
        return;

    Lock(*p_mutex);

    // first one?

    auto ir = p_queue->find(task->exec_priority);

    if(ir == p_queue->end())
    {
        std::set<SubgraphTask*> set;
        set.insert(task);
        (*p_queue)[task->exec_priority] = set;
    }
    else
    {
        ir->second.insert(task);
    }

    Unlock(*p_mutex);
}
bool GenericDevExecutor::RemoveQueue(QueueType queue_type, SubgraphTask* task)
{
    std::mutex* p_mutex = nullptr;
    task_queue_t* p_queue = nullptr;

    if(!GetQueueReference(queue_type, p_mutex, p_queue))
        return false;

    Lock(*p_mutex);

    auto ir = p_queue->find(task->exec_priority);

    if(ir == p_queue->end())
    {
        Unlock(*p_mutex);
        return false;
    }

    bool ret;

    if(ir->second.erase(task))
        ret = true;
    else
        ret = false;

    Unlock(*p_mutex);

    return ret;
}
SubgraphTask* GenericDevExecutor::PopQueue(QueueType queue_type)
{
    std::mutex* p_mutex = nullptr;
    task_queue_t* p_queue = nullptr;

    if(!GetQueueReference(queue_type, p_mutex, p_queue))
        return nullptr;

    Lock(*p_mutex);

    auto ir = p_queue->begin();

    if(ir == p_queue->end())
    {
        Unlock(*p_mutex);
        return nullptr;
    }

    auto set_ir = ir->second.begin();

    SubgraphTask* task = *set_ir;

    ir->second.erase(set_ir);

    if(ir->second.empty())
    {
        p_queue->erase(ir);
    }

    Unlock(*p_mutex);
    return task;
}
int GenericDevExecutor::GetElementNumber(QueueType queue_type)
{
    std::mutex* p_mutex;
    task_queue_t* p_queue;

    if(!GetQueueReference(queue_type, p_mutex, p_queue))
        return 0;

    Lock(*p_mutex);

    int count = 0;

    auto ir = p_queue->begin();

    while(ir != p_queue->end())
    {
        count += ir->second.size();
    }

    Unlock(*p_mutex);

    return count;
}

int GenericDevExecutor::GetRunTaskNum(void)
{
    return GetElementNumber(kRunQueue);
}
int GenericDevExecutor::GetReadyTaskNum(void)
{
    return GetElementNumber(kReadyQueue);
}
int GenericDevExecutor::GetWaitTaskNum(void)
{
    return GetElementNumber(kWaitQueue);
}

void GenericDevExecutor::InsertQueue(SubgraphTask* task)
{
    int task_status = task->GetStatus();
    QueueType queue_type = kWaitQueue;

    if(task_status == EXEC_STATUS_READY)
    {
        queue_type = kReadyQueue;
    }
    else if(task_status == EXEC_STATUS_WAIT)
    {
        queue_type = kWaitQueue;
    }
    else if(task_status == EXEC_STATUS_RUN)
    {
        queue_type = kRunQueue;
    }

    InsertQueue(queue_type, task);
}

bool GenericDevExecutor::OptimizeGraph(SubgraphTask* task)
{
    if(task->graph_optimized)
        return true;

    if(task->graph_handle == nullptr)
        task->graph_handle = DevCreateGraphHandle(task->sub_graph);

    if(task->graph_handle == nullptr)
        return false;

    task->graph_optimized = DevOptimizeGraph(task->graph_handle);

    return task->graph_optimized;
}

Subgraph* GenericDevExecutor::GetOptimizedGraph(SubgraphTask* task)
{
    if(!task->graph_optimized)
        return nullptr;

    return DevGetOptimizedGraph(task->graph_handle);
}

bool GenericDevExecutor::PrerunTask(SubgraphTask* task)
{
    if(DevGetStatus() != kDevNormal)
        return false;

    if(task->graph_handle == nullptr)
        task->graph_handle = DevCreateGraphHandle(task->sub_graph);

    if(task->graph_handle == nullptr || !OptimizeGraph(task))
        return false;

    GraphTask* graph_task = task->graph_task;
    GraphExecutor* executor = graph_task->GetGraphExecutor();

    int optimize_only = 0;

    executor->GetGraphAttr("optimize_only", &optimize_only, sizeof(int));

    if(optimize_only)
        return true;

    unsigned int mem_size;

    if(DevGetMemorySize(task->graph_handle, mem_size))
    {
        void* mem_addr = std::malloc(mem_size);

        DevSetMemory(task->graph_handle, mem_addr);
    }

    if(!DevPrerun(task->graph_handle))
        return false;

    task->SetStatus(EXEC_STATUS_WAIT);

    InsertQueue(kWaitQueue, task);

    return true;
}

bool GenericDevExecutor::SchedTask(SubgraphTask* task)
{
    RemoveQueue(task);
    return RunTask(task);
}

bool GenericDevExecutor::SchedTask(void)
{
    SubgraphTask* task = PopQueue(kReadyQueue);

    if(task)
        return RunTask(task);

    return false;
}

bool GenericDevExecutor::SyncRunTask(SubgraphTask* task)
{
    bool ret = false;

    if(DevGetStatus() == kDevNormal)
        ret = DevSyncRun(task->graph_handle);

    return ret;
}

bool GenericDevExecutor::RunTask(SubgraphTask* task)
{
    // return true: accepted running
    // return false: try again

    if(DevGetStatus() != kDevNormal)
    {
        task->SetStatus(EXEC_STATUS_READY);
        InsertQueue(task);
        return false;
    }

    if(SupportNonblockRun())
    {
        task->Lock();    // protect with OnSubgraphDone()

        bool ret = DevRun(task->graph_handle);

        if(ret)
            task->SetStatus(EXEC_STATUS_RUN);
        else
            task->SetStatus(EXEC_STATUS_READY);

        InsertQueue(task);

        task->Unlock();

        return ret;
    }
    else
    {
        task->Lock();    // protect with Postrun()

        bool ret = DevSyncRun(task->graph_handle);

        task->SetStatus(EXEC_STATUS_WAIT);

        InsertQueue(kWaitQueue, task);

        task->OnTaskDone(ret);

        task->Unlock();

        return ret;
    }
}

bool GenericDevExecutor::PostrunTask(SubgraphTask* task)
{
    if(task->graph_handle == nullptr)
        return false;

    task->Lock();

    DevPostrun(task->graph_handle);
    DevReleaseGraphHandle(task->graph_handle);

    task->graph_handle = nullptr;

    RemoveQueue(task);

    task->Unlock();

    return true;
}

bool GenericDevExecutor::RemoveQueue(SubgraphTask* task)
{
    int task_status = task->GetStatus();
    QueueType queue_type = kWaitQueue;

    if(task_status == EXEC_STATUS_READY)
    {
        queue_type = kReadyQueue;
    }
    else if(task_status == EXEC_STATUS_WAIT)
    {
        queue_type = kWaitQueue;
    }
    else if(task_status == EXEC_STATUS_RUN)
    {
        queue_type = kRunQueue;
    }

    return RemoveQueue(queue_type, task);
}

void GenericDevExecutor::OnSubgraphDone(Subgraph* sub_graph, bool exec_success)
{
    SubgraphTask* task = SubgraphTask::GetSubgraphTask(sub_graph);

    // move the task from RUN QUEUE to WAIT QUEUE
    task->Lock();    // as this is an callback function, to protect the ops in SchedTask()/RunTask()

    RemoveQueue(kRunQueue, task);

    task->SetStatus(EXEC_STATUS_WAIT);

    InsertQueue(kWaitQueue, task);

    task->Unlock();

    task->OnTaskDone(exec_success);
}

bool GenericDevExecutor::SetGraphAttr(SubgraphTask* task, const char* name, const void* val, int size)
{
    return DevSetGraphAttr(task->graph_handle, name, val, size);
}
bool GenericDevExecutor::GetGraphAttr(SubgraphTask* task, const char* name, void* val, int size)
{
    return DevGetGraphAttr(task->graph_handle, name, val, size);
}

}    // namespace TEngine
