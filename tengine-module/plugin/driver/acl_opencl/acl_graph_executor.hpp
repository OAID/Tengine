
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

#ifndef __ACL_GRAPH_EXECUTOR_HPP__
#define __ACL_GRAPH_EXECUTOR_HPP__

#include "generic_dev_executor.hpp"

namespace TEngine {

class ACLGraphExecutor : public GenericDevExecutor
{
public:
    /* by disable NonblockRun(),
          driver->SyncRun() will be called instead of driver->Run()
    */

    ACLGraphExecutor(const dev_id_t& dev_id)
    {
        dev_id_ = dev_id;
    }
    void DevGetWorkload(DevWorkload& load) override;
    bool DevGetPerf(Subgraph* graph, int policy, GraphPerf& perf) override;
    float DevGetFops(Subgraph* graph, int policy) override;
    int DevGetPolicyPriority(int policy) override;
    bool DevGetProposal(Subgraph* graph, int policy, bool static_assign) override;

    bool DevSetGraphAttr(void* graph_handle, const char* name, const void* val, int size) override;
    bool DevGetGraphAttr(void* graph_handle, const char* name, void* val, int size) override;

    void* DevCreateGraphHandle(Subgraph* graph) override;
    bool DevOptimizeGraph(void* graph_handle) override;
    bool DevPrerun(void* graph_handle) override;
    bool DevRun(void* graph_handle) override;
    bool DevSyncRun(void* graph_handle) override;
    bool DevPostrun(void* graph_handle) override;
    bool DevReleaseGraphHandle(void* graph_handle) override;
    bool DevStart(void) override;
    bool DevStop(void) override;

    Subgraph* DevGetOptimizedGraph(void* graph_handle) override;

    const dev_id_t& DevGetID(void) override;
    const dev_type_t& DevGetType(void) override;

    dev_status_t DevGetStatus(void) override;

    bool Init(void) override;
    bool Release(void) override;

    void UnbindDevice(void) override;
    void BindDevice(Device*) override;

    void SetDevice(ACLDevice* dev)
    {
        backend_dev_ = dev;
    }
    ACLDevice* GetDevice(void)
    {
        return backend_dev_;
    }

protected:
    ACLDevice* backend_dev_;
    dev_id_t dev_id_;
};

}    // namespace TEngine

#endif
