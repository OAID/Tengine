
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

#ifndef __NODE_DEV_DRIVER_HPP__
#define __NODE_DEV_DRIVER_HPP__

#include <vector>
#include <string>
#include <atomic>

#include "graph.hpp"
#include "device_driver.hpp"

namespace TEngine {

class NodeDevice : public Device
{
public:
    NodeDevice(const dev_id_t& dev_id) : Device(dev_id) {}
    virtual ~NodeDevice() {}

    dev_status_t dev_status;
};

class NodeDriver : public Driver
{
public:
    struct DevContext
    {
        dev_node_cb_t node_cb;
        NodeDevice* dev;
        void* args;
    };

    NodeDriver();
    virtual ~NodeDriver(){};

    bool InitializeDevice(Device* device) override;
    bool ReleaseDevice(Device* device) override;
    bool StartDevice(Device* device) override;
    bool StopDevice(Device* device) override;

    dev_status_t GetDeviceStatus(Device* device) override;

    void* CreateGraphHandle(Device* dev, Subgraph* graph) override;

    void SetGraphDoneHook(Device* dev, void* graph_handle, dev_graph_cb_t func) override{};
    void SetNodeDoneHook(Device* dev, void* node_handle, dev_node_cb_t func) override;

    virtual bool OptimizeGraph(Device* dev, void* graph_handle, Subgraph* graph) override
    {
        return false;
    }
    bool OptimizeGraph(Device* dev, void* graph_handle) override
    {
        return false;
    }

    bool Prerun(Device* dev, void* graph_handle) override
    {
        return false;
    }
    bool Run(Device* dev, void* graph_handle) override
    {
        return false;
    }
    bool SyncRun(Device* dev, void* graph_handle) override
    {
        return false;
    }
    bool Postrun(Device* dev, void* graph_handle) override
    {
        return false;
    }

    int ProbeDevice(void) override;
    int DestroyDevice(void) override;

    int GetDeviceNum(void) override;
    Device* GetDevice(int idx) override;
    Device* GetDevice(const std::string& name) override;

    int GetDevIDTableSize(void)
    {
        return dev_id_.size();
    }
    const dev_id_t& GetDevIDByIdx(int n)
    {
        return dev_id_[n];
    }

    virtual void* CreateGraphHandle(Device* dev) override;
    virtual bool ReleaseGraphHandle(Device* dev, void* graph_handle) override;

    virtual bool ProbeDevice(const dev_id_t& dev_id) override = 0;
    virtual bool DestroyDevice(Device* device) override = 0;

    virtual bool InitDev(NodeDevice* device)
    {
        return true;
    }
    virtual bool ReleaseDev(NodeDevice* device)
    {
        return true;
    }
    virtual bool StartDev(NodeDevice* device)
    {
        return true;
    }
    virtual bool StopDev(NodeDevice* device)
    {
        return true;
    }

    virtual bool GetWorkload(Device* dev, DevWorkload& load) override;
    virtual bool GetPerf(Device* dev, Subgraph* graph, int policy, GraphPerf& perf) override;
    virtual float GetFops(Device* dev, Subgraph* graph, int policy) override;
    virtual int GetPolicyPriority(Device* dev, int policy) override;
    virtual bool GetProposal(Device* dev, Graph* graph, int policy, bool static_assign) override;

protected:
    std::vector<dev_id_t> dev_id_;
    std::vector<NodeDevice*> dev_table_;
};

}    // namespace TEngine

#endif
