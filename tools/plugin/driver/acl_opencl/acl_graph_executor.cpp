
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
 * Author: haoluo@openailab.com
 */

#include "acl_graph_driver.hpp"
#include "acl_graph_executor.hpp"
#include "graph_task.hpp"

namespace TEngine {

bool ACLGraphExecutor::DevGetProposal(Subgraph* graph, int policy, bool static_assign)
{
    return backend_dev_->GetProposal(graph, policy, static_assign);
}

void ACLGraphExecutor::DevGetWorkload(DevWorkload& load)
{
    backend_dev_->GetWorkload(load);
}

bool ACLGraphExecutor::DevGetPerf(Subgraph* graph, int policy, GraphPerf& perf)
{
    return backend_dev_->GetPerf(graph, policy, perf);
}

float ACLGraphExecutor::DevGetFops(Subgraph* graph, int policy)
{
    return backend_dev_->GetFops(graph, policy);
}

int ACLGraphExecutor::DevGetPolicyPriority(int policy)
{
    return backend_dev_->GetPolicyPriority(policy);
}

bool ACLGraphExecutor::DevSetGraphAttr(void* graph_handle, const char* name, const void* val, int size)
{
    return backend_dev_->SetGraphAttr(graph_handle, name, val, size);
}

bool ACLGraphExecutor::DevGetGraphAttr(void* graph_handle, const char* name, void* val, int size)
{
    return backend_dev_->GetGraphAttr(graph_handle, name, val, size);
}

void* ACLGraphExecutor::DevCreateGraphHandle(Subgraph* graph)
{
    return backend_dev_->CreateGraphHandle(graph);
}

bool ACLGraphExecutor::DevOptimizeGraph(void* graph_handle)
{
    return backend_dev_->OptimizeGraph(graph_handle);
}

Subgraph* ACLGraphExecutor::DevGetOptimizedGraph(void* graph_handle)
{
    return backend_dev_->GetOptimizedGraph(graph_handle);
}

bool ACLGraphExecutor::DevPrerun(void* graph_handle)
{
    return backend_dev_->Prerun(graph_handle);
}

bool ACLGraphExecutor::DevRun(void* graph_handle)
{
    auto f = std::bind(&ACLGraphExecutor::OnSubgraphDone, this, std::placeholders::_1, std::placeholders::_2);

    backend_dev_->SetGraphDoneHook(graph_handle, dev_graph_cb_t(f));

    return backend_dev_->Run(graph_handle);
}

bool ACLGraphExecutor::DevSyncRun(void* graph_handle)
{
    return backend_dev_->SyncRun(graph_handle);
}

bool ACLGraphExecutor::DevPostrun(void* graph_handle)
{
    return backend_dev_->Postrun(graph_handle);
}

bool ACLGraphExecutor::DevReleaseGraphHandle(void* graph_handle)
{
    return backend_dev_->ReleaseGraphHandle(graph_handle);
}

const dev_id_t& ACLGraphExecutor::DevGetID(void)
{
    return backend_dev_->GetDeviceID();
}

const dev_type_t& ACLGraphExecutor::DevGetType(void)
{
    return backend_dev_->GetDeviceType();
}

dev_status_t ACLGraphExecutor::DevGetStatus(void)
{
    return backend_dev_->GetDeviceStatus();
}

bool ACLGraphExecutor::Init(void)
{
    return true;
}

bool ACLGraphExecutor::Release(void)
{
    return true;
}

void ACLGraphExecutor::UnbindDevice(void)
{
    backend_dev_ = nullptr;
}

void ACLGraphExecutor::BindDevice(Device* device)
{
    ACLDevice* dev = dynamic_cast<ACLDevice*>(device);
    backend_dev_ = dev;
}

bool ACLGraphExecutor::DevStart(void)
{
    return backend_dev_->Start();
}

bool ACLGraphExecutor::DevStop(void)
{
    return backend_dev_->Stop();
}

}    // namespace TEngine
