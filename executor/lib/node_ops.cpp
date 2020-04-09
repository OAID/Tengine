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
#include "data_type.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "cpu_driver.hpp"
#include "tengine_errno.hpp"
#include "tensor_mem.hpp"

namespace TEngine {

static std::mutex node_dump_lock;

bool NodeOps::EnableDump(Node* node)
{
    dump_enabled = true;

    return true;
}

bool NodeOps::DisableDump(Node* node)
{
    dump_enabled = false;

    node_dump_lock.lock();

    dump_started = false;

    for(unsigned int i = 0; i < dump_records.size(); i++)
    {
        tensor_dump_header* p_header = &dump_records[i];

        if(p_header->data)
            free(p_header->data);
    }

    dump_records.clear();

    node_dump_lock.unlock();

    return true;
}

bool NodeOps::StartDump(Node* node)
{
    if(!dump_enabled)
    {
        set_tengine_errno(EINVAL);
        return false;
    }

    dump_started = true;

    return true;
}

bool NodeOps::StopDump(Node* node)
{
    dump_started = false;

    return true;
}

int NodeOps::GetDump(Node* node, void** buf, int buf_size)
{
    if(!dump_enabled)
    {
        set_tengine_errno(ENOTSUP);
        return -1;
    }

    node_dump_lock.lock();

    int num = dump_records.size();

    if(num > buf_size)
        num = buf_size;

    for(int i = 0; i < num; i++)
    {
        buf[i] = ( void* )&dump_records[i];
    }

    node_dump_lock.unlock();

    return num;
}

bool NodeOps::SaveDump(Node* node)
{
    if(!dump_started)
        return false;

    node_dump_lock.lock();

    if(!dump_started)
    {
        node_dump_lock.unlock();
        return false;
    }

    int input_number = node->GetInputNum();
    int output_number = node->GetOutputNum();

    if(dump_records.size() == 0)
    {
        dump_records.resize(input_number + output_number);

        for(int i = 0; i < input_number + output_number; i++)
        {
            dump_records[i].data = nullptr;
        }
    }

    /* copy data */
    for(int i = 0; i < input_number + output_number; i++)
    {
        Tensor* tensor;

        if(i < input_number)
            tensor = node->GetInputTensor(i);
        else
            tensor = node->GetOutputTensor(i - input_number);

        tensor_dump_header* head = &dump_records[i];

        if(head->data == nullptr || head->elem_number * head->elem_size != ( int )tensor->GetTotalSize())
        {
            std::vector<int>& tensor_dims = tensor->GetShape().GetDim();
            head->elem_size = DataType::GetTypeSize(tensor->GetDataType());
            head->dim_number = tensor_dims.size();
            head->elem_number = 1;

            for(int j = 0; j < head->dim_number; j++)
            {
                head->elem_number *= tensor_dims[j];
                head->dim[j] = tensor_dims[j];
            }

            if(head->data)
                free(head->data);

            head->data = malloc(head->elem_size * head->elem_number);

            if(head->data == nullptr)
            {
                node_dump_lock.unlock();
                set_tengine_errno(ENOMEM);
                return false;
            }
        }

        memcpy(head->data, get_tensor_mem(tensor), head->elem_number * head->elem_size);
    }

    node_dump_lock.unlock();

    return true;
}

/**** NodeOpsRegistryManager ********/

NodeOpsRegistryManager* NodeOpsRegistryManager::GetInstance(void)
{
    static NodeOpsRegistryManager instance;

    return &instance;
}

NodeOps* NodeOpsRegistryManager::FindNodeOps(const std::string& registry_name, const CPUInfo* cpu_info, Node* node)
{
    NodeOpsRegistry* registry = FindRegistry(registry_name);

    if(!registry)
        return nullptr;

    NodeOps* ops = registry->FindNodeOps(cpu_info, node);

    if(!ops)
        return nullptr;

    ops->SetCPUInfo(cpu_info);

    return ops;
}

NodeOps* NodeOpsRegistryManager::FindNodeOps(const CPUInfo* cpu_info, Node* node)
{
    const char* target_registry = std::getenv("OPS_REGISTRY");
    const char* target_op = std::getenv("OP_NAME");
    NodeOps* ops;
    bool target_search = false;

    if(target_registry)
    {
        target_search = true;

        if(target_op)
        {
            Operator* op = node->GetOp();

            if(op->GetName() == target_op)
                target_search = true;
            else
                target_search = false;
        }
    }

    if(target_search)
    {
        ops = FindNodeOps(target_registry, cpu_info, node);

        if(!ops)
            return nullptr;

        ops->SetCPUInfo(cpu_info);

        return ops;
    }

    ops = RealFindNodeOps(cpu_info, node);

    if(!ops)
        return nullptr;

    ops->SetCPUInfo(cpu_info);

    return ops;
}

NodeOps* NodeOpsRegistryManager::RealFindNodeOps(const CPUInfo* cpu_info, Node* node)
{
    NodeOps* ops;

    if(cpu_info != nullptr)
    {
        // search cpu_type
        int master_cpu = cpu_info->GetMasterCPU();

        const std::string& cpu_model = cpu_info->GetCPUModelString(master_cpu);

        ops = FindNodeOps(cpu_model, cpu_info, node);

        if(ops)
            return ops;

        // search arch
        std::string cpu_arch;

        int int_arch = cpu_info->GetCPUArch(master_cpu);

        if(int_arch == ARCH_ARM_V8)
        {
            cpu_arch = "arm64";
        }
        else if(int_arch == ARCH_ARM_V7)
        {
            cpu_arch = "arm32";
        }

        ops = FindNodeOps(cpu_arch, cpu_info, node);

        if(ops)
            return ops;
    }

    // search x86
#if CONFIG_ARCH_X86
    ops = FindNodeOps("x86", cpu_info, node);
    if(ops)
        return ops;
#endif

    // search common
    ops = FindNodeOps("common", cpu_info, node);
    if(ops)
        return ops;

    // the final search: reference

    ops = FindNodeOps(REF_REGISTRY_NAME, cpu_info, node);

    return ops;
}

void NodeOpsRegistryManager::AddRegistry(const std::string& name, NodeOpsRegistry* reg)
{
    auto manager = GetInstance();

    manager->registry_list[name] = reg;
}

NodeOpsRegistry* NodeOpsRegistryManager::FindRegistry(const std::string& name)
{
    auto manager = GetInstance();

    if(manager->registry_list.count(name) == 0)
        return nullptr;

    return manager->registry_list[name];
}

void NodeOpsRegistryManager::RecordNodeOpsptr(NodeOps* ops)
{
    auto manager = GetInstance();

    manager->ops_list.emplace_back(ops);
}

bool NodeOpsRegistryManager::RegisterOPImplementor(const std::string& registry_name, const std::string& op_name,
                                                   select_node_ops_t select_func, int priority)
{
    // TODO: Add Lock to protect find and register registry

    auto registry = FindRegistry(registry_name);

    if(registry == nullptr)
    {
        registry = new NodeOpsRegistry(registry_name);
        NodeOpsRegistryManager::AddRegistry(registry->reg_name, registry);
    }

    PrioSelector* prio_selector = dynamic_cast<PrioSelector*>(registry->FindSelector(op_name));

    if(prio_selector == nullptr)
    {
        prio_selector = new PrioSelector();
        prio_selector->op_name = op_name;

        registry->RegisterSelector(prio_selector);
    }

    if(!prio_selector->Register(priority, select_func))
        return false;

    return true;
}

NodeOps* NodeOpsRegistry::FindNodeOps(const CPUInfo* cpu_info, Node* node)
{
    Operator* op = node->GetOp();
    const std::string& op_name = op->GetName();

    if(registry.count(op_name) == 0)
        return nullptr;

    NodeOpsSelector* selector = registry[op_name].get();

    return selector->Select(cpu_info, node);
}

NodeOpsSelector* NodeOpsRegistry::FindSelector(const std::string& name)
{
    if(registry.count(name) == 0)
        return nullptr;

    return registry.at(name).get();
}

bool NodeOpsRegistry::RegisterSelector(NodeOpsSelector* selector)
{
    if(registry.count(selector->op_name))
    {
        LOG_ERROR() << "op: " << selector->op_name << "has been register already on: " << reg_name << "\n";
        return false;
    }

    registry[selector->op_name] = NodeOpsSelectorPtr(selector);

    return true;
}

/**** global init  function ****/

void NodeOpsRegistryManagerInit(void) {}

/**** NodeOpsRegsitry ********/

NodeOpsRegistryManager::~NodeOpsRegistryManager()
{
    for(auto e : registry_list)
        delete e.second;
}

}    // namespace TEngine
