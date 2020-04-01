
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

#ifndef __ACL_GRAPH_DRIVER_HPP__
#define __ACL_GRAPH_DRIVER_HPP__

#include <vector>
#include <string>
#include <atomic>
#include <cstdlib>
#include <tuple>
#include <set>

#include "graph.hpp"
#include "acl_graph_device.hpp"
#include "acl_graph_executor.hpp"

namespace TEngine {

class ACLGraph : public Driver
{
public:
    ACLGraph()
    {
        SetName("ACLGraph");
        InitOpSet();
    }
    ~ACLGraph(){};

    bool InitializeDevice(Device* device) override;
    bool ReleaseDevice(Device* device) override;

    bool StartDevice(Device* device) override;
    bool StopDevice(Device* device) override;

    dev_status_t GetDeviceStatus(Device* device) override;

    void* CreateGraphHandle(Device* dev, Subgraph* graph) override;
    void* CreateGraphHandle(Device* dev) override;
    bool ReleaseGraphHandle(Device* dev, void* graph_handle) override;

    void SetGraphDoneHook(Device* dev, void* graph_handle, dev_graph_cb_t func) override;
    void SetNodeDoneHook(Device* dev, void* node_handle, dev_node_cb_t func) override{};

    bool OptimizeGraph(Device* dev, void* graph_handle, Subgraph* graph) override;
    bool OptimizeGraph(Device* dev, void* graph_handle) override;

    Subgraph* GetOptimizedGraph(Device* dev, void* graph_handle) override;

    bool Prerun(Device* dev, void* graph_handle) override;
    bool Run(Device* dev, void* graph_handle) override;
    bool SyncRun(Device* dev, void* graph_handle) override;
    bool Postrun(Device* dev, void* graph_handle) override;

    bool Prerun(Device* dev, void* node_handle, Node* node) override
    {
        return false;
    }
    bool Run(Device* dev, void* node_handle, Node* node) override
    {
        return false;
    }
    bool SyncRun(Device* dev, void* node_handle, Node* node) override
    {
        return false;
    }
    bool Postrun(Device* dev, void* node_handle, Node* node) override
    {
        return false;
    }

    void PushGraph(ACLDevice* acl_dev, DevContext* context);

    bool GetWorkload(Device* dev, DevWorkload& load) override
    {
        return false;
    }
    bool GetPerf(Device* dev, Subgraph* graph, int policy, GraphPerf& perf) override
    {
        return false;
    }
    float GetFops(Device* dev, Subgraph* graph, int policy) override
    {
        return false;
    }
    int GetPolicyPriority(Device* dev, int policy) override
    {
        return false;
    }
    bool GetProposal(Device* dev, Subgraph* graph, int policy, bool static_assign) override;
    bool SetGraphAttr(Device*, void*, const char*, const void*, int) override;
    bool GetGraphAttr(Device*, void*, const char*, void*, int) override;

    void AddDevice(ACLDevice* new_device);    // a special interface for  ACL Device

    int ProbeDevice(void) override
    {
        return device_table_.size();
    }
    bool ProbeDevice(const dev_id_t& dev_id) override;

    int DestroyDevice(void) override
    {
        return 0;
    };
    bool DestroyDevice(Device* device) override;

    int GetDeviceNum(void) override;
    Device* GetDevice(int idx) override;
    Device* GetDevice(const std::string& name) override;

    bool OpSupported(const std::string& op_name);

protected:
    void InitOpSet(void);
    std::unordered_map<std::string, ACLDevice*> device_table_;
    std::set<std::string> op_set_;
};

}    // namespace TEngine

#endif
