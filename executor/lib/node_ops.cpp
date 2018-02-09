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
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "soc_runner.hpp"
#include "arm64_registry.hpp"


namespace TEngine {

void TaskSync::WaitDone(void)
{
    std::unique_lock<std::mutex> lock(wait_mutex);

    if(done!=request)
         cv.wait(lock,[this]{return done==request;});
 
    lock.unlock(); 
}


void TaskSync::IncRequest(unsigned int number)
{
    request+=number;
}


void TaskSync::IncDone(unsigned int number)
{
    uint64_t prev_val=done.fetch_add(number);

    if(prev_val+number==request)
    {
        std::unique_lock<std::mutex> lock(wait_mutex);

        cv.notify_all();

        lock.unlock();
    }

}


/**** NodeOpsRegistryManager ********/

NodeOpsRegistryManager * NodeOpsRegistryManager::GetInstance(void)
{
    static NodeOpsRegistryManager instance;

    return &instance;
}

NodeOps * NodeOpsRegistryManager::FindNodeOps(SocInfo * soc_info, Node * node)
{
    NodeOps * ops;

    //search soc specific
    NodeOpsRegistry * soc_registry=FindRegistry(soc_info->soc_name);

    if(soc_registry)
    {
        ops=soc_registry->FindNodeOps(soc_info,node);

        if(ops) 
           return ops;
    }

    //search cpu_type
   
    const std::string& cpu_type=soc_info->cpu_info[soc_info->master_cpu].cpu_type;

    NodeOpsRegistry * type_registry=FindRegistry(cpu_type);

    if(type_registry)
    {
        ops=type_registry->FindNodeOps(soc_info,node);

        if(ops) 
           return ops;
    }

    //search arch

    const std::string& cpu_arch=soc_info->cpu_info[soc_info->master_cpu].cpu_arch;

    NodeOpsRegistry * arch_registry=FindRegistry(cpu_arch);

    if(arch_registry)
    {
        ops=arch_registry->FindNodeOps(soc_info,node);

        if(ops) 
           return ops;
    }

    //the final search: reference
    NodeOpsRegistry * ref_registry=FindRegistry("reference");

    if(ref_registry)
    {
        ops=ref_registry->FindNodeOps(soc_info,node);

        if(ops) 
           return ops;
    }

    return nullptr;
}

void NodeOpsRegistryManager::AddRegistry(const std::string& name, NodeOpsRegistry * reg)
{
    auto manager=GetInstance();

    manager->registry_list[name]=reg;

}

NodeOpsRegistry * NodeOpsRegistryManager::FindRegistry(const std::string& name)
{
   auto manager=GetInstance();

   if(manager->registry_list.count(name)==0)
          return nullptr;

   return manager->registry_list[name];
}

bool NodeOpsRegistryManager::RegisterOPSelector(const std::string& registry_name, 
                                         NodeOpsSelector * selector)
{
    auto registry=FindRegistry(registry_name);

    if(registry==nullptr)
          return false;

    return registry->RegisterSelector(selector);
}

bool NodeOpsRegistryManager::RegisterOPImplementor(const std::string& registry_name, 
                                       const std::string& op_name, NodeOps * ops)
{
    auto registry=FindRegistry(registry_name);

    if(registry==nullptr)
          return false;

    return registry->RegiserSimpleNodeOps(op_name,ops);
}


bool NodeOpsRegistry::RegiserSimpleNodeOps(const std::string& op_name, NodeOps * ops)
{
   SimpleSelector * p_selector=new SimpleSelector(ops);

   p_selector->op_name=op_name;
   p_selector->node_ops=ops;

   if(RegisterSelector(p_selector))
      return true;

   delete p_selector;

   return false;
}


NodeOps * NodeOpsRegistry::FindNodeOps(SocInfo *soc_info,Node * node)
{
   Operator * op=node->GetOp();
   const std::string& op_name=op->GetName();

   if(registry.count(op_name)==0)
          return nullptr;

   NodeOpsSelector* selector=registry[op_name].get();

   return selector->Select(soc_info,node);
}

NodeOpsSelector *  NodeOpsRegistry::FindSelector(const std::string& name)
{
   if(registry.count(name)==0)
          return nullptr;

   return registry.at(name).get(); 
}

bool  NodeOpsRegistry::RegisterSelector(NodeOpsSelector * selector)
{
   if(registry.count(selector->op_name))
   {
       LOG_ERROR()<<"op: "<<selector->op_name<<"has been register already on: "<<reg_name<<"\n";
       return false;
   }

   registry[selector->op_name]=NodeOpsSelectorPtr(selector);

   return true;
}


/**** global init  function ****/

namespace ref_ops {

   static NodeOpsRegistry  ref_ops_registry(REF_REGISTRY_NAME);

   NodeOpsRegistry * GetRegistry(void)
   {
         return &ref_ops_registry;
   }

}

void NodeOpsRegistryManagerInit(void)
{
    NodeOpsRegistry * registry;

    registry=arm64_ops::GetRegistry();
    NodeOpsRegistryManager::AddRegistry(registry->reg_name,registry);

    registry=a72_ops::GetRegistry();
    NodeOpsRegistryManager::AddRegistry(registry->reg_name,registry);

    registry=a53_ops::GetRegistry();
    NodeOpsRegistryManager::AddRegistry(registry->reg_name,registry);

    registry=ref_ops::GetRegistry();
    NodeOpsRegistryManager::AddRegistry(registry->reg_name,registry);
}

/**** NodeOpsRegsitry ********/





} //namespace TEngine
