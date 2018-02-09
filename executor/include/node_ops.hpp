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
#ifndef __NODE_OPS_HPP__
#define __NODE_OPS_HPP__

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <map>
#include <atomic>
#include <condition_variable>
#include <mutex>


namespace TEngine {

#define ATTR_NODE_OPS "node_ops"
#define ATTR_INPLACE "inplace"


class SocInfo;
class Node;
struct NodeOps;
struct sub_op_task;

using inplace_t=std::unordered_map<int,int>;

using task_exec_t=std::function<bool(int cpu, int seq, void * data)>;
using task_dispatch_t=std::function<bool(std::vector<sub_op_task>& tasks, int cpu)>;

using mem_alloc_t=std::function<void *(int size)>;
using mem_free_t= std::function<void (void *)>;
using select_node_ops_t=std::function<NodeOps *(SocInfo *, Node *)>;
using node_ops_create_t=std::function<NodeOps*(SocInfo *, Node *)>;

struct sub_op_task {
   task_exec_t exec_func;
   int seq;
   void * data;
};

class TaskSync{

public:
   void WaitDone(void);
   void IncRequest(unsigned int);
   void IncDone(unsigned int);

private:
   std::atomic<uint64_t> request;
   std::atomic<uint64_t> done;
   std::mutex   wait_mutex;
   std::condition_variable cv;  

};

struct NodeOps {
    virtual bool OnBind(Node *){return true;}
    virtual bool Prerun(Node *){return true;}
    virtual bool Postrun(Node *){return true;}
    virtual bool Run(Node *)=0;


    NodeOps(void)
    {
       need_free=false;
    }

    /* for delete this usage: https://isocpp.org/wiki/faq/freestore-mgmt#delete-this */
    void Release(void) { if (need_free) delete this;}
	
    void SetHelper(mem_alloc_t alloc, mem_free_t free, task_dispatch_t disp) 
    {
         mem_alloc=alloc;
         mem_free=free;
         task_dispatch=disp;
    }
	

    virtual ~NodeOps(){}

    bool        need_free;
    mem_alloc_t mem_alloc;
    mem_free_t mem_free;
    task_dispatch_t task_dispatch;
};


struct MTNodeOps: public NodeOps {

    MTNodeOps(void) { need_free=true; sync=new TaskSync();}
 
    void WaitDone(void) { sync->WaitDone(); }
    void IncRequest(unsigned int num) {sync->IncRequest(num);}
    void IncDone(unsigned int num) { sync->IncDone(num);}

     virtual ~MTNodeOps(){ if(sync) delete sync;}

    TaskSync * sync;

};

struct NodeOpsSelector {

   virtual NodeOps * Select(SocInfo * soc_info,Node * node)=0;
   virtual ~NodeOpsSelector(){};

   std::string op_name;
};


struct SimpleSelector : public NodeOpsSelector
{

    SimpleSelector(NodeOps * ops){ node_ops=ops; creator=nullptr;}
    SimpleSelector(node_ops_create_t func){ creator=func; node_ops=nullptr;}

    NodeOps * Select(SocInfo * soc_info,Node * node)
    {

       if(node_ops)
       	{
       	       node_ops->need_free=false;
	   	return node_ops;
       	}
       else
       	{
              NodeOps * new_ops=creator(soc_info,node);
	      new_ops->need_free=true;
              return  new_ops;
       }
    }

    NodeOps* node_ops;
    node_ops_create_t creator;
	

    ~SimpleSelector(void) { if(node_ops) delete node_ops;}
};



struct PrioSelector: public NodeOpsSelector
{
   NodeOps * Select(SocInfo * soc_info,Node * node)
   {
        auto begin=prio_list.begin();
        auto end=prio_list.end();

        for(auto ir=begin;ir!=end;ir++)
        {
            auto match_func=ir->second;

            auto ops=match_func(soc_info,node);

            if(ops)
              return ops;
        }

        return nullptr;
   }

   void Register(int priority, select_node_ops_t func)
   {
       prio_list[priority]=func;
   }

   std::map<int,select_node_ops_t> prio_list;
};

using NodeOpsSelectorPtr=std::shared_ptr<NodeOpsSelector>;

struct NodeOpsRegistry {

    NodeOpsRegistry(const std::string& name) { reg_name=name;}

    NodeOps * FindNodeOps(SocInfo *,Node *);
    bool RegiserSimpleNodeOps(const std::string& op_name, NodeOps * ops);

    NodeOpsSelector * FindSelector(const std::string& name);
    bool RegisterSelector(NodeOpsSelector * selector);

    std::unordered_map<std::string,NodeOpsSelectorPtr> registry;
    std::string reg_name;
};


#define REF_REGISTRY_NAME "reference"

class NodeOpsRegistryManager {

public:

   static NodeOps * FindNodeOps(SocInfo *,Node *);

   static NodeOpsRegistryManager * GetInstance(void);

   static void AddRegistry(const std::string& name, NodeOpsRegistry * reg);
   static NodeOpsRegistry * FindRegistry(const std::string& name); 

   static bool RegisterOPSelector(const std::string& registry_name, NodeOpsSelector * selector);
   static bool RegisterOPImplementor(const std::string& registry_name, 
                                     const std::string& op_name, NodeOps * ops);

   std::unordered_map<std::string, NodeOpsRegistry *>  registry_list;

};






} //namespace TEngine


#endif
