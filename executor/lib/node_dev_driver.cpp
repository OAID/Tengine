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

#include "node_dev_driver.hpp"
#include "node_dev_executor.hpp"

namespace TEngine {

NodeDriver::NodeDriver(void) {}

bool NodeDriver::InitializeDevice(Device* device)
{
    NodeDevice* node_dev = dynamic_cast<NodeDevice*>(device);

    if(!InitDev(node_dev))
        return false;

    node_dev->BindDriver(this);
    node_dev->dev_status = kDevStopped;

    return true;
}

bool NodeDriver::ReleaseDevice(Device* device)
{
    NodeDevice* node_dev = dynamic_cast<NodeDevice*>(device);

    StopDevice(node_dev);

    node_dev->dev_status = kDevRemoved;

    ReleaseDev(node_dev);

    return true;
}

bool NodeDriver::StartDevice(Device* device)
{
    NodeDevice* node_dev = dynamic_cast<NodeDevice*>(device);

    if(node_dev->dev_status == kDevInvalid || node_dev->dev_status == kDevRemoved)
        return false;

    if(!StartDev(node_dev))
        return false;

    node_dev->dev_status = kDevNormal;

    return true;
}

bool NodeDriver::StopDevice(Device* device)
{
    NodeDevice* node_dev = dynamic_cast<NodeDevice*>(device);

    if(node_dev->dev_status == kDevStopped)
        return true;

    if(!StopDev(node_dev))
        return false;

    node_dev->dev_status = kDevStopped;

    return true;
}

dev_status_t NodeDriver::GetDeviceStatus(Device* device)
{
    NodeDevice* node_dev = dynamic_cast<NodeDevice*>(device);

    return node_dev->dev_status;
}

void* NodeDriver::CreateGraphHandle(Device* dev, Subgraph* graph)
{
    return nullptr;
}

void* NodeDriver::CreateGraphHandle(Device* dev)
{
    DevContext* context = new DevContext();
    NodeDevice* node_device = dynamic_cast<NodeDevice*>(dev);

    context->dev = node_device;
    context->node_cb = nullptr;

    return context;
}

bool NodeDriver::ReleaseGraphHandle(Device* dev, void* graph_handle)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);
    delete context;
    return true;
}

void NodeDriver::SetNodeDoneHook(Device* dev, void* node_handle, dev_node_cb_t func)
{
    DevContext* context = reinterpret_cast<DevContext*>(node_handle);
    context->node_cb = func;
}

int NodeDriver::ProbeDevice(void)
{
    for(auto id : dev_id_)
    {
        ProbeDevice(id);
    }

    return GetDeviceNum();
}

int NodeDriver::DestroyDevice(void)
{
    int dev_num = GetDeviceNum();
    int count = 0;

    for(int i = 0; i < dev_num; i++)
    {
        Device* dev = GetDevice(0);

        if(DestroyDevice(dev))
            count++;
        else
            break;
    }

    return count;
}

int NodeDriver::GetDeviceNum(void)
{
    return dev_table_.size();
}

Device* NodeDriver::GetDevice(int idx)
{
    if(( unsigned int )idx >= dev_table_.size())
        return nullptr;

    return dynamic_cast<Device*>(dev_table_[idx]);
}

Device* NodeDriver::GetDevice(const std::string& name)
{
    int n = dev_table_.size();
    int i;

    for(i = 0; i < n; i++)
    {
        NodeDevice* dev = dev_table_[i];
        if(dev->GetName() == name)
            break;
    }

    if(i == n)
        return nullptr;

    return dynamic_cast<Device*>(dev_table_[i]);
}

bool NodeDriver::GetWorkload(Device* dev, DevWorkload& load)
{
    return false;
}

bool NodeDriver::GetPerf(Device* dev, Subgraph* graph, int policy, GraphPerf& perf)
{
    return false;
}

float NodeDriver::GetFops(Device* dev, Subgraph* graph, int policy)
{
    return 0.0f;
}

int NodeDriver::GetPolicyPriority(Device* dev, int policy)
{
    return 10000;
}

bool NodeDriver::GetProposal(Device* dev, Graph* graph, int policy, bool static_assign)
{
    return true;
}

}    // namespace TEngine
