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
#include "generic_engine.hpp"
#include "dev_allocator.hpp"
#include "dev_scheduler.hpp"
#include "graph_task.hpp"
#include "tensor_mem.hpp"
#include "logger.hpp"
#include "tengine_config.hpp"

namespace TEngine {

GenericEngine::GenericEngine(void)
{
    name = "generic";
}

GenericEngine::~GenericEngine(void)
{
    if(!graph_map_.empty())
    {
        XLOG_ERROR() << "ERROR: Not all graph task in dev engine has been released\n";
    }
}

exec_handle_t GenericEngine::AddGraphExecutor(GraphExecutor* graph_executor)
{
    GraphTask* graph_task = new GraphTask(graph_executor);

    graph_task->SetEngine(this);

    Lock();

    graph_map_[graph_executor] = graph_task;

    Unlock();

    any* ret = new any();

    (*ret) = graph_task;

    return ret;
}

void* GenericEngine::GetTensorBuffer(Tensor* tensor, exec_handle_t h)
{
    return get_tensor_mem(tensor);
}

bool GenericEngine::SetTensorBuffer(Tensor* tensor, void* addr, int size, exec_handle_t h)
{
    return set_tensor_mem(tensor, addr, size, nullptr);
}

bool GenericEngine::Prerun(exec_handle_t h)
{
    GraphTask* graph_task = any_cast<GraphTask*>(*h);
    GraphExecutor* graph_executor = graph_task->GetGraphExecutor();
    Graph* graph = graph_task->GetGraph();

    // call DevAllocator to allocate graph into

    DevAllocator* dev_allocator = nullptr;

    std::string alloc_policy;

    TEngineConfig::Get("dev_allocator", alloc_policy);

    if(alloc_policy.empty() || !DevAllocatorManager::Get(alloc_policy, dev_allocator))
    {
        XLOG_ERROR() << "cannot get proper dev allocator\n";
        return false;
    }

    std::vector<Subgraph*> sub_graph_list;

    if(!dev_allocator->Allocate(this, graph_executor, graph, sub_graph_list))
    {
        XLOG_ERROR() << "dev executor allocator failed\n";

        for(auto e : sub_graph_list)
            delete e;

        return false;
    }

    // create SubgraphTask
    for(unsigned int i = 0; i < sub_graph_list.size(); i++)
    {
        Subgraph* sub_graph = sub_graph_list[i];
        SubgraphTask* new_task = new SubgraphTask(sub_graph);

        new_task->Init(graph_task);
        graph_task->AddSubgraphTask(new_task);
    }

    if(!graph_task->Prerun())
    {
        graph_task->ReclaimSubgraphTask();
        return false;
    }

    return true;
}

bool GenericEngine::Run(exec_handle_t h, exec_event_t& event)
{
    GraphTask* graph_task = any_cast<GraphTask*>(*h);

    return graph_task->Run(event);
}

bool GenericEngine::SyncRun(exec_handle_t h)
{
    GraphTask* graph_task = any_cast<GraphTask*>(*h);

    return graph_task->SyncRun();
}

bool GenericEngine::Postrun(exec_handle_t h)
{
    GraphTask* graph_task = any_cast<GraphTask*>(*h);

    graph_task->Postrun();

    graph_task->ReclaimSubgraphTask();

    return true;
}

exec_status_t GenericEngine::GetStatus(exec_handle_t h)
{
    GraphTask* graph_task = any_cast<GraphTask*>(*h);

    return graph_task->GetStatus();
}

bool GenericEngine::SetEventHook(exec_handle_t h, int event, event_handler_t cb_func, void* cb_arg)
{
    GraphTask* graph_task = any_cast<GraphTask*>(*h);

    return graph_task->SetEventHook(event, cb_func, cb_arg);
}

const std::string& GenericEngine::GetStatusStr(const exec_status_t& status)
{
    static std::string created = "CREATED";
    static std::string inited = "INITED";
    static std::string run = "RUN";
    static std::string done = "DONE";
    static std::string bad = "BAD";
    static std::string unknown = "UNKNOWN";

    int s = any_cast<int>(status);

    switch(s)
    {
        case EXEC_STATUS_CREATED:
            return created;
        case EXEC_STATUS_INITED:
            return inited;
        case EXEC_STATUS_RUN:
            return run;
        case EXEC_STATUS_DONE:
            return done;
        case EXEC_STATUS_BAD:
            return bad;
        default:
            break;
    }

    return unknown;
}

int GenericEngine::GetStatusCode(const exec_status_t& status)
{
    int s = any_cast<int>(status);

    return s;
}

std::string GenericEngine::GetErrorStr(exec_handle_t h)
{
    return "NO ERROR:-)\n";
}

bool GenericEngine::RemoveGraphExecutor(exec_handle_t h)
{
    GraphTask* graph_task = any_cast<GraphTask*>(*h);
    GraphExecutor* graph_executor = graph_task->GetGraphExecutor();

    Lock();

    graph_map_.erase(graph_executor);

    Unlock();

    delete graph_task;
    delete h;

    return true;
}

int GenericEngine::Wait(exec_handle_t h, exec_event_t& e, int try_wait)
{
    GraphTask* graph_task = any_cast<GraphTask*>(*h);

    return graph_task->Wait(e, try_wait);
}

bool GenericEngine::SetCallback(exec_handle_t h, exec_event_t& e, int event, exec_cb_t cb)
{
    GraphTask* graph_task = any_cast<GraphTask*>(*h);

    return graph_task->SetCallback(e, event, cb);
}

bool GenericEngine::OptimizeGraph(exec_handle_t h)
{
    return true;
}

Graph* GenericEngine::GetOptimizedGraph(exec_handle_t h)
{
    GraphTask* graph_task = any_cast<GraphTask*>(*h);

    return graph_task->GetOptimizedGraph();
}

bool GenericEngine::SetScheduler(const std::string& sched_name)
{
    DevSchedulerPtr ptr;
    if(!DevSchedulerManager::Get(sched_name, ptr))
        return false;
    scheduler_ = ptr.get();
    return true;
}

bool GenericEngine::GetGraphAttr(exec_handle_t h, const char* name, void* val, int size)
{
    if(h == nullptr)
        return false;

    GraphTask* graph_task = any_cast<GraphTask*>(*h);

    return graph_task->GetAttr(name, val, size);
}

bool GenericEngine::SetGraphAttr(exec_handle_t h, const char* name, const void* val, int size)
{
    if(h == nullptr)
        return false;

    GraphTask* graph_task = any_cast<GraphTask*>(*h);

    return graph_task->SetAttr(name, val, size);
}

}    // namespace TEngine
