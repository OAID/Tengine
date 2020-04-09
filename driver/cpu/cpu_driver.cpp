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
#include "cpu_driver.hpp"
#include "tensor_mem.hpp"
#include "cpu_executor.hpp"

#include "node_dump.hpp"
#include "graph_perf.hpp"
#include "tengine_errno.hpp"

namespace TEngine {

bool CPUDriver::InitializeDevice(Device* device)
{
    CPUDevice* cpu_dev = dynamic_cast<CPUDevice*>(device);

    cpu_dev->LaunchMaster();
    cpu_dev->LaunchAider();

    cpu_dev->BindDriver(this);
    cpu_dev->dev_status = kDevStopped;

    return true;
}

bool CPUDriver::ReleaseDevice(Device* device)
{
    CPUDevice* cpu_dev = dynamic_cast<CPUDevice*>(device);

    StopDevice(device);

    cpu_dev->KillMaster();
    cpu_dev->KillAider();

    cpu_dev->dev_status = kDevRemoved;

    return true;
}

bool CPUDriver::StartDevice(Device* device)
{
    CPUDevice* cpu_dev = dynamic_cast<CPUDevice*>(device);

    if(cpu_dev->dev_status == kDevInvalid || cpu_dev->dev_status == kDevRemoved)
        return false;

    cpu_dev->dev_status = kDevNormal;

    return true;
}

bool CPUDriver::StopDevice(Device* device)
{
    CPUDevice* cpu_dev = dynamic_cast<CPUDevice*>(device);

    if(cpu_dev->dev_status == kDevStopped)
        return true;

    cpu_dev->dev_status = kDevStopped;

    // TODO: ensure all worker threads are in idle status

    return true;
}

dev_status_t CPUDriver::GetDeviceStatus(Device* device)
{
    CPUDevice* cpu_dev = dynamic_cast<CPUDevice*>(device);
    return cpu_dev->dev_status;
}

void* CPUDriver::CreateGraphHandle(Device* dev, Subgraph* graph)
{
    DevContext* context = new DevContext();
    CPUDevice* cpu_dev = dynamic_cast<CPUDevice*>(dev);

    context->dev = cpu_dev;
    context->sub_graph = graph;
    context->graph_cb = nullptr;
    context->optimized_graph = nullptr;

    return context;
}

void* CPUDriver::CreateGraphHandle(Device* dev)
{
    DevContext* context = new DevContext();

    context->dev = dynamic_cast<CPUDevice*>(dev);
    context->sub_graph = nullptr;
    context->graph_cb = nullptr;
    context->optimized_graph = nullptr;

    return context;
}

bool CPUDriver::ReleaseGraphHandle(Device* dev, void* graph_handle)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);

    // if(context->optimized_graph)
    //     delete context->optimized_graph;

    delete context;
    return true;
}

void CPUDriver::SetGraphDoneHook(Device* dev, void* graph_handle, dev_graph_cb_t func)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);
    context->graph_cb = func;
}

Subgraph* CPUDriver::GetOptimizedGraph(Device* dev, void* graph_handle)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);

    return context->optimized_graph;
}

bool CPUDriver::OptimizeGraph(Device* dev, void* graph_handle, Subgraph* graph)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);
    CPUDevice* cpu_info = context->dev;

    context->sub_graph = graph;

    return cpu_info->RealOptimizeGraph(context, graph);
}

bool CPUDriver::OptimizeGraph(Device* dev, void* graph_handle)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);

    return OptimizeGraph(dev, graph_handle, context->sub_graph);
}

bool CPUDriver::Prerun(Device* dev, void* graph_handle)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);
    CPUDevice* cpu_dev = context->dev;

    return cpu_dev->RealPrerun(context);
}

bool CPUDriver::SyncRun(Device* dev, void* graph_handle)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);
    CPUDevice* cpu_dev = context->dev;

    return cpu_dev->RealRun(context->optimized_graph);
}

void CPUDriver::PushGraph(CPUDevice* cpu_dev, DevContext* context)
{
    std::vector<cpu_task> task_list;

    cpu_task task;

    task.context = context;

    task_list.emplace_back(task);

    cpu_dev->PushMasterTask(task_list);
}

bool CPUDriver::Run(Device* dev, void* graph_handle)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);
    CPUDevice* cpu_dev = context->dev;

    PushGraph(cpu_dev, context);

    return true;
}

bool CPUDriver::Postrun(Device* dev, void* graph_handle)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);
    CPUDevice* cpu_dev = context->dev;

    return cpu_dev->RealPostrun(context);
}

bool CPUDriver::SetDevAttr(Device* dev, const char* attr_name, const void* val, int size)
{
    LOG_ERROR() << "try to set attr " << attr_name << " on dev: " << dev->GetName() << "\n";
    set_tengine_errno(ENOTSUP);
    return false;
}

bool CPUDriver::GetDevAttr(Device* dev, const char* attr_name, void* val, int size)
{
    LOG_ERROR() << "try to get attr " << attr_name << " on dev: " << dev->GetName() << "\n";
    set_tengine_errno(ENOTSUP);
    return false;
}

bool CPUDriver::SetGraphAttr(Device* dev, void* graph_handle, const char* name, const void* val, int size)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);

    Subgraph* graph = context->optimized_graph ? context->optimized_graph : context->sub_graph;

    if(OnSetGraphAttr(context, graph, name, val, size))
        return true;
#if 0
	LOG_ERROR()<<"try to set attr "<<name <<" for graph: "<<graph->GetName()
		       <<" on dev: "<<dev->GetName()<<"\n";
#endif
    return false;
}

bool CPUDriver::GetGraphAttr(Device* dev, void* graph_handle, const char* name, void* val, int size)
{
    DevContext* context = reinterpret_cast<DevContext*>(graph_handle);

    Subgraph* graph = context->optimized_graph ? context->optimized_graph : context->sub_graph;

    if(OnGetGraphAttr(context, graph, name, val, size))
        return true;

#if 0
	LOG_ERROR()<<"try to get attr "<<name <<" for graph: "<<graph->GetName()
		       <<" on dev: "<<dev->GetName()<<"\n";
#endif

    return false;
}

int CPUDriver::DestroyDevice(void)
{
    int count = 0;
    int dev_num = GetDeviceNum();

    for(int i = 0; i < dev_num; i++)
    {
        Device* device = GetDevice(0);

        if(DestroyDevice(device))
            count++;
        else
            break;
    }

    return count;
}

bool CPUDriver::DestroyDevice(Device* device)
{
    CPUDevice* cpu_dev = reinterpret_cast<CPUDevice*>(device);

    if(GetDeviceStatus(cpu_dev) != kDevStopped)
        return false;

    ReleaseDevice(cpu_dev);

    device_table_.erase(cpu_dev->GetName());

    delete cpu_dev;

    return true;
}

int CPUDriver::GetDeviceNum(void)
{
    return device_table_.size();
}

Device* CPUDriver::GetDevice(int idx)
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

Device* CPUDriver::GetDevice(const std::string& name)
{
    if(device_table_.count(name) == 0)
        return nullptr;

    return device_table_[name];
}

void CPUDriver::AddDevice(CPUDevice* new_device)
{
    new_device->SetName(new_device->GetDeviceID());

    device_table_[new_device->GetName()] = new_device;

    InitializeDevice(new_device);

    // register executor interface as well, so that DriverManager::LoadDevice() can work

    auto dev_executor_factory = DevExecutorFactory::GetFactory();
    dev_executor_factory->RegisterInterface<CPUExecutor, const dev_id_t&>(new_device->GetName());
}

bool CPUDriver::GetProposal(Device* dev, Graph* graph, int policy, bool static_assign)
{
    int fit_level = static_assign ? DEV_PROPOSAL_STATIC : DEV_PROPOSAL_GOODAT;

    int node_number = graph->seq_nodes.size();

    for(int i = 0; i < node_number; i++)
    {
        Node* node = graph->seq_nodes[i];
        Operator* op = node->GetOp();

        if(op->GetName() == "Input" || op->GetName() == "Const")
            continue;

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

void CPUDriver::InitGraphAttrIO(void)
{
    auto f0 = std::bind(&CPUDriver::OnSetGraphPerfAttr, this, std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);

    auto f1 = std::bind(&CPUDriver::OnGetGraphPerfAttr, this, std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);

    auto f2 = std::bind(&CPUDriver::OnSetNodeDumpAttr, this, std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);

    auto f3 = std::bind(&CPUDriver::OnGetNodeDumpAttr, this, std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);

    set_attr_table_[ATTR_GRAPH_PERF_STAT] = f0;
    set_attr_table_[ATTR_GRAPH_NODE_DUMP] = f2;

    get_attr_table_[ATTR_GRAPH_PERF_STAT] = f1;
    get_attr_table_[ATTR_GRAPH_NODE_DUMP] = f3;
}

bool CPUDriver::OnSetGraphAttr(DevContext* context, Subgraph* graph, const char* attr_name, const void* val, int size)
{
    if(set_attr_table_.count(attr_name) == 0)
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    set_func_t set_func = set_attr_table_.at(attr_name);

    return set_func(context, graph, attr_name, val, size);
}

bool CPUDriver::OnGetGraphAttr(DevContext* context, Subgraph* graph, const char* attr_name, void* val, int size)
{
    if(get_attr_table_.count(attr_name) == 0)
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    get_func_t get_func = get_attr_table_.at(attr_name);

    return get_func(context, graph, attr_name, val, size);
}

bool CPUDriver::OnGetNodeDumpAttr(DevContext* context, Subgraph* graph, const char* name, void* buf, int size)
{
    if(size != sizeof(NodeDumpMsg))
    {
        set_tengine_errno(EINVAL);
        return false;
    }

    NodeDumpMsg* msg = ( NodeDumpMsg* )buf;
    const char* node_name = msg->node_name;

    Node* node = graph->FindNode(node_name);

    if(node == nullptr)
    {
        set_tengine_errno(EINVAL);
        return false;
    }

    if(!node->ExistAttr(ATTR_NODE_OPS))
    {
        set_tengine_errno(EINVAL);
        return false;
    }

    NodeOps* node_ops = any_cast<NodeOps*>(node->GetAttr(ATTR_NODE_OPS));

    msg->ret_number = node_ops->GetDump(node, msg->buf, msg->buf_size);

    if(msg->ret_number >= 0)
        return true;
    else
        return false;
}

bool CPUDriver::OnGetGraphPerfAttr(DevContext* context, Subgraph* graph, const char* name, void* buf, int size)
{
    if(size != sizeof(GraphPerfMsg))
    {
        set_tengine_errno(EINVAL);
        return false;
    }

    GraphPerfMsg* msg = ( GraphPerfMsg* )(buf);

    CPUDevice* dev = context->dev;

    msg->ret_number = dev->GetGraphPerfStat(graph, msg->buf, msg->buf_size);

    if(msg->ret_number >= 0)
        return true;
    else
        return false;
}

bool CPUDriver::OnSetNodeDumpAttr(DevContext* context, Subgraph* graph, const char* name, const void* buf, int size)
{
    if(size != sizeof(NodeDumpMsg))
    {
        set_tengine_errno(EINVAL);
        return false;
    }

    const NodeDumpMsg* msg = ( const NodeDumpMsg* )buf;
    const char* node_name = msg->node_name;

    Node* node = graph->FindNode(node_name);

    if(node == nullptr)
    {
        set_tengine_errno(EINVAL);
        return false;
    }

    if(!node->ExistAttr(ATTR_NODE_OPS))
    {
        set_tengine_errno(ENOTSUP);
        return false;
    }

    NodeOps* node_ops = any_cast<NodeOps*>(node->GetAttr(ATTR_NODE_OPS));

    int action = msg->action;

    switch(action)
    {
        case NODE_DUMP_ACTION_DISABLE:
            return node_ops->DisableDump(node);
            break;
        case NODE_DUMP_ACTION_ENABLE:
            return node_ops->EnableDump(node);
            break;
        case NODE_DUMP_ACTION_START:
            return node_ops->StartDump(node);
            break;
        case NODE_DUMP_ACTION_STOP:
            return node_ops->StopDump(node);
            break;
        default:
            set_tengine_errno(EINVAL);
            return false;
    }
}

bool CPUDriver::OnSetGraphPerfAttr(DevContext* context, Subgraph* graph, const char* name, const void* buf, int size)
{
    if(size != sizeof(GraphPerfMsg))
    {
        set_tengine_errno(EINVAL);
        return false;
    }

    const GraphPerfMsg* msg = ( const GraphPerfMsg* )(buf);

    CPUDevice* dev = context->dev;

    return dev->SetGraphPerfStat(graph, msg->action);
}

/************************************/

struct default_cpu_param
{
    int* cpu_list;
    int cpu_number;

    default_cpu_param(void)
    {
        cpu_list = nullptr;
        cpu_number = 0;
    }
    ~default_cpu_param(void)
    {
        if(cpu_list)
            free(cpu_list);
    }
};

}    // namespace TEngine

using namespace TEngine;

static default_cpu_param default_param;

extern struct cpu_info* probe_system_cpu(void);
extern void free_probe_cpu_info(struct cpu_info*);

static void probe_func(void)
{
    struct cpu_info* cpu_dev = probe_system_cpu();

    if(cpu_dev == nullptr)
    {
        XLOG_ERROR() << "cannot probe system cpu setting\n";
        return;
    }

    int* saved_list = cpu_dev->online_cpu_list;
    int saved_number = cpu_dev->online_cpu_number;

    int* online_list = NULL;

    if(default_param.cpu_number)
    {
        online_list = ( int* )malloc(sizeof(int) * default_param.cpu_number);

        for(int i = 0; i < default_param.cpu_number; i++)
            online_list[i] = default_param.cpu_list[i];

        cpu_dev->online_cpu_list = online_list;
        cpu_dev->online_cpu_number = default_param.cpu_number;
    }

    create_cpu_device(cpu_dev->cpu_name, cpu_dev);

    cpu_dev->online_cpu_list = saved_list;
    cpu_dev->online_cpu_number = saved_number;

    free_probe_cpu_info(cpu_dev);

    if(online_list)
        free(online_list);
}

namespace TEngine {

void CPUDriverInit(void)
{
    CPUDriver* cpu_driver = new CPUDriver();

    DriverManager::RegisterDriver(cpu_driver->GetName(), cpu_driver);
    DriverManager::SetProbeDefault(probe_func);
}
}    // namespace TEngine

/* implementing interface defined in cpu_info.h */

int create_cpu_device(const char* dev_name, const struct cpu_info* device)
{
    CPUDevice* new_dev = new CPUDevice(dev_name, device);
    CPUDriver* cpu_driver = dynamic_cast<CPUDriver*>(DriverManager::GetDriver("CPU"));

    cpu_driver->AddDevice(new_dev);

    DriverManager::LoadDevice(cpu_driver, new_dev);

    if(!DriverManager::HasDefaultDevice())
    {
        // set it as default device
        DriverManager::SetDefaultDevice(new_dev);
    }

    return 0;
}

const struct cpu_info* get_cpu_info(const char* dev_name)
{
    CPUDevice* cpu_dev = dynamic_cast<CPUDevice*>(DriverManager::GetDevice(dev_name));

    if(cpu_dev == nullptr)
        return NULL;

    const CPUInfo* cpu_info = cpu_dev->GetCPUInfo();

    return &cpu_info->dev;
}

static int set_default_cpu(const int* cpu_list, int cpu_number)
{
    default_param.cpu_list = ( int* )malloc(cpu_number * sizeof(int));

    default_param.cpu_number = cpu_number;

    memcpy(default_param.cpu_list, cpu_list, sizeof(int) * cpu_number);

    return 0;
}

void set_online_cpu(struct cpu_info* cpu_info, const int* cpu_list, int cpu_number)
{
    if(cpu_info == nullptr)
    {
        set_default_cpu(cpu_list, cpu_number);
        return;
    }

    if(cpu_info->online_cpu_list)
        free(cpu_info->online_cpu_list);

    cpu_info->online_cpu_list = ( int* )malloc(cpu_number * sizeof(int));

    memcpy(cpu_info->online_cpu_list, cpu_list, cpu_number * sizeof(int));
    cpu_info->online_cpu_number = cpu_number;
}
