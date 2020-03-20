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

#include "operator/concat.hpp"
#include "acl_graph_driver.hpp"

#define ACL_OPENCL "acl_opencl"
#define ATTR_NODE_OPS "node_ops"
#ifdef __ANDROID__
#define dynamic_cast static_cast
#endif
namespace TEngine {

static int create_acl_device(const char* device)
{
    ACLDevice* acl_dev = new ACLDevice(device);
    ACLGraph* acl_graph = dynamic_cast<ACLGraph*>(DriverManager::GetDriver("ACLGraph"));
    acl_graph->AddDevice(acl_dev);

    DriverManager::LoadDevice(acl_graph, acl_dev);

    return 0;
}

bool ACLGraph::InitializeDevice(Device* device)
{
    ACLDevice* acl_dev = dynamic_cast<ACLDevice*>(device);

    acl_dev->Launch();
    acl_dev->BindDriver(this);
    acl_dev->dev_status = kDevStopped;

    return true;
}

bool ACLGraph::ReleaseDevice(Device* device)
{
    ACLDevice* acl_dev = dynamic_cast<ACLDevice*>(device);

    acl_dev->dev_status = kDevRemoved;
    return true;
}

bool ACLGraph::StartDevice(Device* device)
{
    ACLDevice* acl_dev = dynamic_cast<ACLDevice*>(device);
    if(acl_dev->dev_status == kDevInvalid || acl_dev->dev_status == kDevRemoved)
        return false;

    acl_dev->dev_status = kDevNormal;
    return true;
}

bool ACLGraph::StopDevice(Device* device)
{
    ACLDevice* acl_dev = dynamic_cast<ACLDevice*>(device);
    if(acl_dev->dev_status == kDevStopped)
        return true;

    acl_dev->dev_status = kDevStopped;
    return true;
}

dev_status_t ACLGraph::GetDeviceStatus(Device* device)
{
    ACLDevice* acl_dev = dynamic_cast<ACLDevice*>(device);
    return acl_dev->dev_status;
}

void* ACLGraph::CreateGraphHandle(Device* dev, Subgraph* graph)
{
    DevContext* context = new DevContext();
    ACLDevice* acl_dev = dynamic_cast<ACLDevice*>(dev);

    context->dev = acl_dev;
    context->sub_graph = graph;
    context->optimized_graph = nullptr;
    context->graph_cb = nullptr;
    context->graph = nullptr;

    return context;
}

void* ACLGraph::CreateGraphHandle(Device* dev)
{
    DevContext* context = new DevContext();

    context->dev = dynamic_cast<ACLDevice*>(dev);

    return context;
}

bool ACLGraph::ReleaseGraphHandle(Device* dev, void* graph_handle)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);

    // if(context->optimized_graph)
    //	   delete context->optimized_graph;

    delete context;

    return true;
}

void ACLGraph::SetGraphDoneHook(Device* dev, void* graph_handle, dev_graph_cb_t func)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);
    context->graph_cb = func;
}

bool ACLGraph::OptimizeGraph(Device* dev, void* graph_handle)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);

    return OptimizeGraph(dev, graph_handle, context->sub_graph);
}

bool ACLGraph::OptimizeGraph(Device* dev, void* graph_handle, Subgraph* graph)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);
    context->sub_graph = graph;
    ACLDevice* acl_device = reinterpret_cast<ACLDevice*>(context->dev);

    return acl_device->RealOptimizeGraph(context, graph);
}

Subgraph* ACLGraph::GetOptimizedGraph(Device* dev, void* graph_handle)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);
    return context->optimized_graph;
}

bool ACLGraph::Prerun(Device* dev, void* graph_handle)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);
    ACLDevice* acl_dev = reinterpret_cast<ACLDevice*>(context->dev);

    return acl_dev->RealPrerun(context);
}

bool ACLGraph::SyncRun(Device* dev, void* graph_handle)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);
    ACLDevice* acl_dev = reinterpret_cast<ACLDevice*>(context->dev);

    return acl_dev->RealSyncRun(context);
}

bool ACLGraph::Run(Device* dev, void* graph_handle)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);
    ACLDevice* acl_dev = reinterpret_cast<ACLDevice*>(context->dev);

    PushGraph(acl_dev, context);

    return true;
}

bool ACLGraph::Postrun(Device* dev, void* graph_handle)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);
    ACLDevice* acl_dev = reinterpret_cast<ACLDevice*>(context->dev);

    return acl_dev->RealPostrun(context);
}

void ACLGraph::PushGraph(ACLDevice* acl_dev, DevContext* context)
{
    std::vector<acl_task> task_list;

    acl_task task;

    task.context = context;

    task_list.emplace_back(task);

    acl_dev->PushTask(task_list);
}

bool ACLGraph::ProbeDevice(const dev_id_t& dev_id)
{
    Device* dev = new Device(dev_id);

    InitializeDevice(dev);

    return true;
}

bool ACLGraph::DestroyDevice(Device* device)
{
    ACLDevice* acl_dev = reinterpret_cast<ACLDevice*>(device);

    if(GetDeviceStatus(acl_dev) != kDevStopped)
        return false;

    ReleaseDevice(acl_dev);

    device_table_.erase(acl_dev->GetName());

    delete acl_dev;

    return true;
}

int ACLGraph::GetDeviceNum(void)
{
    return device_table_.size();
}

Device* ACLGraph::GetDevice(int idx)
{
    auto ir = device_table_.begin();
    auto end = device_table_.end();

    int i;

    for(i = 0; i < idx && ir != end; i++, ir++)
        ;

    if(ir == end)
        return nullptr;

    return ir->second;
}

Device* ACLGraph::GetDevice(const std::string& name)
{
    if(device_table_.count(name) == 0)
        return nullptr;

    return device_table_[name];
}

void ACLGraph::AddDevice(ACLDevice* new_device)
{
    new_device->SetName(new_device->GetDeviceID());

    device_table_[new_device->GetName()] = new_device;

    InitializeDevice(new_device);

    // register executor interface as well, so that DriverManager::LoadDevice() can work

    auto dev_executor_factory = DevExecutorFactory::GetFactory();
    dev_executor_factory->RegisterInterface<ACLGraphExecutor, const dev_id_t&>(new_device->GetName());
}

bool ACLGraph::SetGraphAttr(Device* dev, void* graph_handle, const char* name, const void* val, int size)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);

    Subgraph* graph = context->optimized_graph ? context->optimized_graph : context->sub_graph;

    LOG_ERROR() << "try to set attr " << name << " for graph: " << graph->GetName() << " on dev: " << dev->GetName()
                << "\n";
    return false;
}

bool ACLGraph::GetGraphAttr(Device* dev, void* graph_handle, const char* name, void* val, int size)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);

    Subgraph* graph = context->optimized_graph ? context->optimized_graph : context->sub_graph;

    LOG_ERROR() << "try to get attr " << name << " for graph: " << graph->GetName() << " on dev: " << dev->GetName()
                << "\n";

    return false;
}

bool ACLGraph::OpSupported(const std::string& op_name)
{
    if(op_set_.count(op_name))
        return true;
    else
        return false;
}

bool ACLGraph::GetProposal(Device* dev, Subgraph* graph, int policy, bool static_assign)
{
    int fit_level = static_assign ? DEV_PROPOSAL_STATIC : DEV_PROPOSAL_GOODAT;

    int node_number = graph->seq_nodes.size();

    for(int i = 0; i < node_number; i++)
    {
        Node* node = graph->seq_nodes[i];
        Operator* op = node->GetOp();

        if(op->GetName() == "Input" || op->GetName() == "Const" || !OpSupported(op->GetName()))
            continue;

        /* special check for concat, as 18.05 only support depth concat */

        if(op->GetName() == "Concat")
        {
            Concat* concat_op = (Concat*)(op);
            ConcatParam* param = concat_op->GetParam();

            if(param->axis != 1)
                continue;
        }

        bool bind_to_me = false;

        if(!node->ExistAttr(DEV_PROPOSAL_ATTR))
        {
            bind_to_me = true;
        }
        else
        {
            const DevProposal& orig_prop = any_cast<DevProposal>(node->GetAttr(DEV_PROPOSAL_ATTR));

            if(orig_prop.level < fit_level)
                bind_to_me = true;
        }

        if(bind_to_me)
        {
            DevProposal prop;
            prop.dev_id = dev->GetDeviceID();
            prop.level = fit_level;

            node->SetAttr(DEV_PROPOSAL_ATTR, prop);
        }
    }

    return true;
}

void ACLGraph::InitOpSet(void)
{
    op_set_.insert("BatchNormalization");
    op_set_.insert("Convolution");
    op_set_.insert("Dropout");
    op_set_.insert("Eltwise");
    op_set_.insert("FullyConnected");
    op_set_.insert("Pooling");
    op_set_.insert("ReLu");
    op_set_.insert("ReLu6");
    op_set_.insert("Resize");
    // op_set_.insert("Softmax");
    op_set_.insert("Scale");

    /*
       Concat needs special handle:
       As in SSD, if concat done by GPU, the graph will be partitioned into many parts
    */

    const char* gpu_concat = std::getenv("GPU_CONCAT");

    if(!gpu_concat || gpu_concat[0] != '0')
        op_set_.insert("Concat");
}

//////////////////////////////////////////////

extern "C" int acl_opencl_init(void)
{
    ACLGraph* acl_graph = new ACLGraph();

    DriverManager::RegisterDriver(acl_graph->GetName(), acl_graph);

    create_acl_device(ACL_OPENCL);

    LOG_INFO() << "ACL Graph Initialized\n";
    return 0;
}

}    // namespace TEngine
