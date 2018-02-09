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

#include <queue>

#include "logger.hpp"
#include "graph_executor.hpp"
#include "graph_task.hpp"
#include "generic_engine.hpp"

#include "dev_allocator.hpp"
#include "dev_executor.hpp"

#include "tengine_lock.hpp"
#include "tengine_config.hpp"

namespace TEngine {

struct LatencyAllocator: public DevAllocator {

   LatencyAllocator() { name="Latency";}

   bool Allocate(GenericEngine * engine,GraphExecutor * graph_executor, Graph * graph, std::vector<Subgraph *>& sub_list) override;
   const std::string& GetName(void) override { return  name;}

};

bool LatencyAllocator::Allocate(GenericEngine * engine,GraphExecutor * graph_executor, Graph * graph, std::vector<Subgraph *>& sub_list)
{
   return false;
}

struct ManualAllocator: public DevAllocator {

   ManualAllocator(void) { name="Manual";}
   bool Allocate(GenericEngine * engine, GraphExecutor * graph_executor,Graph * graph, std::vector<Subgraph *>& sub_list) override;
   const std::string& GetName(void) override { return  name;}

};


bool ManualAllocator::Allocate(GenericEngine * engine,GraphExecutor * graph_executor, Graph * graph, std::vector<Subgraph *>& sub_list)
{
    //step 1: assign dev_executor for all  nodes

    int node_number=graph->seq_nodes.size();

    for(int i=0;i<node_number;i++)
    {
        Node * node=graph->seq_nodes[i];
        Operator * op=node->GetOp();
        DevExecutor * dev_executor;

        if(op->GetName()=="Input" || op->GetName()=="Const")
            continue;

        if(node->ExistAttr("dev_id"))
        {
              const std::string& dev_id=any_cast<std::string>(node->GetAttr("dev_id"));

              if(DevExecutorManager::GetDevExecutorByID(dev_id,dev_executor))
              {
                   node->SetAttr("dev_executor",dev_executor);
                   continue;
              }
	      else
	      {
	            LOG_ERROR()<<"cannot find dev exeuctor with name: "<<dev_id<<"\n";
                    return false;				
	      }
        }

        //not assigned using the default one
        if(!DevExecutorManager::GetDefaultDevExecutor(dev_executor))
        {
            LOG_ERROR()<<"failed to assign dev executor for node: "<<node->GetName()<<"\n";
            return false;
        }

        node->SetAttr("dev_executor",dev_executor);
    }

    
    //step 2: partition graph into subgraph according to dev_executor allocation 

    std::vector<int> visited(node_number,0);
    std::queue<Node *> search_root_nodes;

    //push all output nodes into search_root_nodes
    for(unsigned int i=0;i<graph->output_nodes.size();i++)
         search_root_nodes.push(graph->output_nodes[i]);

    int sub_graph_count=0;

    while(1)
    {
        if(search_root_nodes.empty())
             break;

        Node * node=search_root_nodes.front();
        search_root_nodes.pop();

        if(visited[node->GetNodeIndex()])
            continue;

        visited[node->GetNodeIndex()]=1;

        DevExecutor * dev_executor=any_cast<DevExecutor *>(node->GetAttr("dev_executor"));

	sub_graph_count++;

        std::string sub_graph_name=graph->GetName();

        sub_graph_name=sub_graph_name+":"+std::to_string(sub_graph_count);

        Subgraph * sub_graph=new Subgraph(sub_graph_name);

        sub_graph->seq_nodes.push_back(node);

        //for all nodes ...
        std::queue<Node *> check_list;

        check_list.push(node);
 
        do {

          node=check_list.front();
          check_list.pop();

         bool node_in_input=false;

         //check all parenet nodes, if any
         for(int i=0;i<node->GetParentNum();i++)
         {
             Node * parent_node=node->GetParentNode(i);

             //input or const node
             if(!parent_node->ExistAttr("dev_executor"))
             {
                 sub_graph->seq_nodes.push_back(parent_node);

                 Operator * op=parent_node->GetOp();

                 if(op->GetName()=="Input")
                      sub_graph->input_nodes.push_back(parent_node);

                 visited[parent_node->GetNodeIndex()]=1;

                 continue;
             }

             DevExecutor * dev=any_cast<DevExecutor *>(parent_node->GetAttr("dev_executor"));

             if(dev==dev_executor)
             {
                if(!visited[parent_node->GetNodeIndex()])
                {
                    visited[parent_node->GetNodeIndex()]=1;
                    sub_graph->seq_nodes.push_back(parent_node);
                    check_list.push(parent_node);
                }
             }
             else
             {
                if(!visited[parent_node->GetNodeIndex()])
                      search_root_nodes.push(parent_node);

                 //I'm an input node, receive outputs from other sub graph
                 if(!node_in_input)
                 {
                     sub_graph->input_nodes.push_back(node);
                     node_in_input=true;
                 }

             }
         }

         bool node_in_output=false;
		 
         for(unsigned int i=0;i<node->GetOutputNum();i++)
         {
               Tensor * out_tensor=node->GetOutputTensor(i);
			  	
              for(unsigned int j=0;j<out_tensor->consumer.size();j++)
              {
			  	
	             Node * child_node=out_tensor->consumer[j]->owner;
	             DevExecutor * dev=any_cast<DevExecutor *>(child_node->GetAttr("dev_executor"));

	             if(dev==dev_executor)
	             {
                        if(!visited[child_node->GetNodeIndex()])
                        {
	                   sub_graph->seq_nodes.push_back(child_node);
	                   visited[child_node->GetNodeIndex()]=1;
	                   check_list.push(child_node);
                        }
	             }
	             else
	             {
	                 //I'm an output node, output node for other sub graph
	                 if(!visited[child_node->GetNodeIndex()])
	                      search_root_nodes.push(child_node);

			  if(!node_in_output)
			  {
		                 sub_graph->output_nodes.push_back(node);
				  node_in_output=true;
			  }
	             }
              	}

		if(out_tensor->consumer.empty())
		{
			if(!node_in_output)
			{
				sub_graph->output_nodes.push_back(node);
				node_in_output=true;
			}
		}
           }
        } while(!check_list.empty());

         //Got one subgraph now!
        sub_graph->SetAttr("dev_executor",dev_executor);
        sub_list.insert(sub_list.begin(),sub_graph);
    }
    
    for(auto sub_graph: sub_list)
    {
        sub_graph->SanitizeGraph();
        //sub_graph->DumpGraph();
    }

    return true;
}

void DevAllocatorManager::OnDevExecutorRegistered(DevExecutor * dev_executor)
{
	DevAllocatorManager * manager=GetInstance();

	LockExecutorList();

	manager->executor_list.push_back(dev_executor);

	UnlockExecutorList();
	
}

void DevAllocatorManager::OnDevExecutorUnregistered(DevExecutor* dev_executor)
{
	DevAllocatorManager * manager=GetInstance();

	LockExecutorList();

	auto ir=manager->executor_list.begin();
       auto end=manager->executor_list.end();

	while(ir!=end)
	{
	   if(*ir==dev_executor)
	   {
	         manager->executor_list.erase(ir);
		  break; 
	   }

	   ir++;
	}
	
	UnlockExecutorList();

}

void DevAllocatorManager::LockExecutorList(void)
{
     DevAllocatorManager * manager=GetInstance();
    TEngineLock(manager->list_lock);
}

void DevAllocatorManager::UnlockExecutorList(void)
{
     DevAllocatorManager * manager=GetInstance();
     TEngineUnlock(manager->list_lock);
}
	

void DevAllocatorManagerInit(void)
{
    ManualAllocator  * manual=new ManualAllocator();
    DevAllocatorManager::Add(manual->GetName(),manual);

    TEngineConfig::Set("dev_allocator",manual->GetName());
}

} //namespace TEngine
