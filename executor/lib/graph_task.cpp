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

#include <atomic>
#include <set>

#include "generic_engine.hpp"
#include "logger.hpp"
#include "graph_task.hpp"
#include "graph_executor.hpp"
#include "dev_scheduler.hpp"

namespace TEngine {

GraphTask::GraphTask(GraphExecutor * graph_executor)
{
     graph_executor_=graph_executor;
     graph_=graph_executor->GetGraph();
     exec_policy_=graph_executor->GetExecPolicy();
     exec_priority_=graph_executor->GetExecPriority();
     dev_engine_=nullptr;
     optimized_graph_=nullptr;
     status_=EXEC_STATUS_CREATED;
}

void GraphTask::AddSubgraphTask(SubgraphTask * sub_task)
{
      sub_task_list_.push_back(sub_task);
}

void GraphTask::RemoveSubgraphTask(SubgraphTask * sub_task)
{
	auto ir=sub_task_list_.begin();
	auto end=sub_task_list_.end();

	while(ir!=end)
	{
	    if(*ir==sub_task)
			sub_task_list_.erase(ir);
	}
}

bool GraphTask::SyncRunSubgraphTask(SubgraphTask * sub_task)
{
     DevScheduler * scheduler=dev_engine_->GetScheduler();

     active_sub_task_count_++;

     bool ret=scheduler->SyncRunTask(dev_engine_,sub_task->dev_executor,sub_task);

     if(!ret)
         return false;

     sub_task->OnSyncTaskDone();

     return true;
}

bool GraphTask::RunSubgraphTask(SubgraphTask * sub_task)
{

        if(status_!= EXEC_STATUS_READY &&
	     status_!=EXEC_STATUS_RUN)
	     return false;
		
	DevScheduler * scheduler=dev_engine_->GetScheduler();

	 active_sub_task_count_++;

	return scheduler->SchedTask(dev_engine_,sub_task->dev_executor,sub_task);
}

bool GraphTask::OptimizeGraph(void)
{
      bool ret=false;

      for(auto e: sub_task_list_)
      {
         DevExecutor * dev_executor=e->dev_executor;

         ret=dev_executor->OptimizeGraph(e);

         if(!ret)
            break;
      }

      return ret;
}

Graph * GraphTask::MergeSubgraph(Graph * origin_graph, const std::vector<Subgraph *>& sub_list)
{
      std::string graph_name=origin_graph->GetName()+".optimized";

      Graph * graph=new Graph(graph_name);
      int subgraph_number=sub_list.size();

      /* first: search the graph input nodes and graph output nodes
          and collect all nodes and tensors */

      graph->input_nodes=origin_graph->input_nodes;

      for(int i=0;i<subgraph_number;i++)
      {
          Subgraph * sub=sub_list[i];

          graph->seq_nodes.insert(graph->seq_nodes.end(),
                sub->seq_nodes.begin(),sub->seq_nodes.end());

          for(unsigned int k=0;k<sub->output_nodes.size();k++)
          {
              Node * node=sub->output_nodes[k];
              
              unsigned int l;

              for(l=0;l<node->GetOutputNum();l++)
              {
                  Tensor * tensor=node->GetOutputTensor(l);

                  if(tensor->consumer.size()==0)
                      break;
              }

              if(l<node->GetOutputNum())
                  graph->output_nodes.push_back(node);
          }

      }
      

     /*second: setup the tensor map */

     for(unsigned int i=0;i<graph->seq_nodes.size();i++)
     {
         Node * node=graph->seq_nodes[i];
         node->SetNodeIndex(i);

         for(unsigned int l=0;l<node->GetOutputNum();l++)
         {
              Tensor * tensor=node->GetOutputTensor(l);
              graph->AddTensorMap(tensor->GetName(),tensor);
         }

     }

     /*third: get the output nodes order*/

     if(graph->output_nodes.size()>1)
     {
         graph->output_nodes=origin_graph->output_nodes;
     }

     /* last reorder the nodes */

     graph->SanitizeGraph();

     return graph;

}

Graph * GraphTask::GetOptimizedGraph(void)
{
    
     std::vector<Subgraph *> sub_list;

     for(auto e: sub_task_list_)
     {
         if(!e->graph_optimized)
         {
             sub_list.push_back(e->sub_graph);
             continue;
         }

         DevExecutor * dev_executor=e->dev_executor;

         Subgraph * sub_graph=dev_executor->GetOptimizedGraph(e);

         sub_list.push_back(sub_graph);
     }

     if(sub_list.empty())
     {
        return nullptr;
     }

     if(optimized_graph_)
        delete optimized_graph_;

     optimized_graph_=MergeSubgraph(graph_,sub_list);

     return optimized_graph_;
}


bool GraphTask::Prerun(void)
{
      output_task_number_=0;
	  
      DevScheduler * scheduler=dev_engine_->GetScheduler();
	  
      for(auto e: sub_task_list_)
      {
          if( !scheduler->PrerunTask(dev_engine_, e->dev_executor,e))
          {
               XLOG_ERROR()<<"failed to Prerun task on  dev executor: "
			   	        <<e->dev_executor->GetName()<<"\n";
               return false;
          }

          e->attached=true;
	 
	   if(e->is_output_task)
	   	 output_task_number_++;
      }

      status_=EXEC_STATUS_INITED;

     return true;
}

bool GraphTask::SyncRun(void)
{
        if(status_!=EXEC_STATUS_INITED &&
	     status_!=EXEC_STATUS_READY)

        {
            XLOG_ERROR()<<"bad status: "<<dev_engine_->GetStatusStr(status_)<<"\n";
	     return false;
        }	

        status_=EXEC_STATUS_RUN;
        output_wait_count_=output_task_number_*2; //never signal graph task done
        active_sub_task_count_=0;


        std::set<SubgraphTask *> working_set;

        for(unsigned int i=0;i<sub_task_list_.size();i++)
             working_set.insert(sub_task_list_[i]);

        //first try: sequentially execution 
        for(unsigned int j=0;j<sub_task_list_.size();j++)
        {
           SubgraphTask * sub_task=sub_task_list_[j];

           if(!sub_task->input_wait_count_)
            {
                bool ret=SyncRunSubgraphTask(sub_task);
                if(ret)
                   working_set.erase(sub_task);
                else
                {
                   status_=EXEC_STATUS_BAD;
                   return false;
                }
           }
        }
            

        //second round: repeatly try the tasks in working_set

       while(!working_set.empty())
       {
         for(auto ir=working_set.begin();ir!=working_set.end();)
         {
              SubgraphTask * sub_task=*ir;

              if(sub_task->input_wait_count_)
                  ir++;
              else
              {
                  if(SyncRunSubgraphTask(sub_task))
                  {
                      ir=working_set.erase(ir);
                  }
                  else
                  {
                       status_=EXEC_STATUS_BAD;
                       return false;
                  }
              }
         }   
       }
       

       status_=EXEC_STATUS_READY;
       return true;
}

bool GraphTask::Run(exec_event_t& event)
{
        if(status_!=EXEC_STATUS_INITED &&
	     status_!=EXEC_STATUS_READY)
        {
            XLOG_ERROR()<<"bad status: "<<dev_engine_->GetStatusStr(status_)<<"\n";
	     return false;
        }	

	/* inital status */
        status_=EXEC_STATUS_RUN;
	output_wait_count_=output_task_number_;  
	active_sub_task_count_=0;
	task_done_=false;
	
        WaitEvent * p_event=new WaitEvent();
        p_event->wait_count=0;
	wait_event_=p_event;

	//let all input tasks run
	for(auto e: sub_task_list_)
	{
	    if(!e->saved_input_wait_count_ 
			&& !RunSubgraphTask(e))
	    {
	         XLOG_ERROR()<<"failed to run task on dev executor: "
			 	            <<e->dev_executor->GetName()<<"\n";
		  status_=EXEC_STATUS_BAD;
                  delete wait_event_;
	         return false;
	    }
	}

        if(active_sub_task_count_==0 && status_== EXEC_STATUS_RUN)
        {
             XLOG_ERROR()<<"No sub task launched!!\n";
             
             delete wait_event_;
 
             return false;
        }

	event=p_event;
	 
	return true;
}

bool GraphTask::SetCallback(exec_event_t& e, int event, exec_cb_t cb)
{
    return false;
}



int GraphTask::Wait(exec_event_t& event, int try_wait)
{
        WaitEvent * p_event=any_cast<WaitEvent *>(event);

		{
		
           std::unique_lock<std::mutex> lock(p_event->mutex);

           if(try_wait&& !task_done_)
           {
             lock.unlock();
             return 0;
           }

           p_event->wait_count++;

           if(!task_done_)
	           p_event->cond.wait(lock,[this]{return task_done_;});

           lock.unlock();
	   }

       if(p_event->wait_count.fetch_sub(1)==1)
       	      delete p_event;
	
	return 1;
}

void GraphTask::SignalGraphTaskDone(void)
{
       std::unique_lock<std::mutex> lock(wait_event_->mutex, std::defer_lock);

       lock.lock();
	   
	task_done_=true;

       lock.unlock();
	   
	wait_event_->cond.notify_all();
      
}

void GraphTask::Postrun(void)
{
	if(status_==EXEC_STATUS_RUN)
		return ;

       DevScheduler * scheduler=dev_engine_->GetScheduler();

       for(auto e: sub_task_list_)
       	{
        	scheduler->PostrunTask(dev_engine_,e->dev_executor,e);   	
		e->attached=false;
       	}

       status_=EXEC_STATUS_INVALID;
}


void GraphTask::OnOutputSubgraphTaskDone(SubgraphTask * sub_task)
{
       int active_task=active_sub_task_count_.fetch_sub(1);

	 if(active_task==1 && status_==EXEC_STATUS_BAD)
	 {
	       SignalGraphTaskDone();
		return ;
	 }

	if(output_wait_count_.fetch_sub(1)==1)
	{
		status_=EXEC_STATUS_READY;
		SignalGraphTaskDone();
	}

}

void GraphTask::OnSubgraphTaskError(SubgraphTask * sub_task)
{
	status_=EXEC_STATUS_BAD;  //this will stop to generate new tasks

       OnOutputSubgraphTaskDone(sub_task);
}


void GraphTask::ReclaimSubgraphTask(void)
{

    DevScheduler * scheduler=dev_engine_->GetScheduler();
	
     for(SubgraphTask * sub : sub_task_list_)
     {

	  if(sub->GetStatus()==EXEC_STATUS_RUN)
	  {
	  	XLOG_ERROR()<<"cannot reclaim subgraph while it is running\n";
		return ;
	  }
	  
         if(sub->attached)
         {
               scheduler->PostrunTask(dev_engine_,sub->dev_executor,sub); 
         }
	  
         sub->Release();

	  delete sub;
     }

     sub_task_list_.clear();

}

GraphTask::~GraphTask(void)
{
     
}


/**************Subgraph Task ***************/

void SubgraphTask::SetSubgraphTask(Subgraph * sub_graph, SubgraphTask * task)
{
      sub_graph->SetAttr("sub_task",task);
}

SubgraphTask * SubgraphTask::GetSubgraphTask(Subgraph * sub_graph)
{
      if(!sub_graph->ExistAttr("sub_task"))
		return nullptr;

       SubgraphTask * sub_task=any_cast<SubgraphTask *>(sub_graph->GetAttr("sub_task"));

       return sub_task;
}

SubgraphTask::SubgraphTask(Subgraph * sub)
{
       sub_graph=sub;
       graph_optimized=false;
       graph_handle=nullptr;

       SetSubgraphTask(sub,this);
}

void SubgraphTask::Init(GraphTask * graph_tsk)
{
    graph_task=graph_tsk;
    GraphExecutor * graph_executor=graph_task->GetGraphExecutor();
    Graph * graph=graph_task->GetGraph();

    //

    exec_policy= DevScheduler::MapPolicy(graph_executor->GetExecPolicy());
    exec_priority=graph_executor->GetExecPriority();


    //check if it is a global output task
    for(auto e: sub_graph->output_nodes)
    {
        for(auto m: graph->output_nodes)
        {
           if(e==m)
           {
              is_output_task=true;
              break;
           }
        }
    }

    saved_input_wait_count_=0;

    //for input nodes, calculate the input_wait_mask
    for(Node * node: sub_graph->input_nodes)
    {
          std::uint64_t mask=0;

	   for(unsigned int i=0;i<node->GetInputNum();i++)
	   {
	   	NodePort * port=node->GetInputPort(i);
		Tensor * tensor=port->tensor;

		if(tensor->GetType()==kVarTensor)
		    mask|=1<<port->port_index;
	   }

          if(mask)
          {
        	SetNodeInputWaitMask(node,mask);
		CreateNodeInputWaitCounter(node);
	  	auto p_counter=GetNodeInputWaitCounter(node);
	        *p_counter=mask;

                 node->SetAttr("consumer_task",this);
		   
		saved_input_wait_count_++;
          }
    }

     input_wait_count_=saved_input_wait_count_;
     SetStatus(EXEC_STATUS_INITED);
     dev_executor=any_cast<DevExecutor *>(sub_graph->GetAttr("dev_executor"));
}

void SubgraphTask::OnSyncTaskDone(void)
{
     input_wait_count_=saved_input_wait_count_;

     for(unsigned int i=0;i<sub_graph->output_nodes.size();i++)
     {
       	 Node * node=sub_graph->output_nodes[i];
       	 OnOutputNodeDone(node,true);
     }

     
}

void SubgraphTask::OnTaskDone(bool exec_success)
{
      input_wait_count_=saved_input_wait_count_;

      if(!exec_success)
      {
 	    graph_task->OnSubgraphTaskError(this);
	     return ;
      	}

       //for all output nodes, 
       for(unsigned int i=0;i<sub_graph->output_nodes.size();i++)
       	{
       	      Node * node=sub_graph->output_nodes[i];
       	      OnOutputNodeDone(node,false);
       	}

       if(is_output_task)
        	 graph_task->OnOutputSubgraphTaskDone(this);
  }


 void SubgraphTask::OnOutputNodeDone(Node * node,bool sync_mode)
 {
       int out_tensor_num=node->GetOutputNum();

	for(int i=0;i<out_tensor_num;i++)
	{
	     Tensor * out_tensor=node->GetOutputTensor(i);

	     for(unsigned int j=0;j<out_tensor->consumer.size();j++)
	     {
	           NodePort * port=out_tensor->consumer[j];
                   Node * consumer=port->owner;

                   SubgraphTask * consumer_task=any_cast<SubgraphTask *>
                               (consumer->GetAttr("consumer_task"));

		   consumer_task->OnNodeInputTensorReady(consumer, port->port_index, sync_mode);
	     }
	}
 }

 void SubgraphTask::OnNodeInputTensorReady(Node * node, int input_idx,bool sync_mode)
 {
	  std::atomic<std::uint64_t> * p_counter;
	  std::uint64_t mask=1<<input_idx;

	  p_counter=GetNodeInputWaitCounter(node);

	  std::uint64_t prev_val=p_counter->fetch_sub(mask);

         //if it is the last waited tensor?
	  if((prev_val-mask)==0)
	  {
	  	OnInputNodeReady(node,sync_mode);

               //prepare for next run
		(*p_counter)=GetNodeInputWaitMask(node);
	  }
 	
 }


void SubgraphTask::OnInputNodeReady(Node * node, bool sync_mode)
{
	if(input_wait_count_.fetch_sub(1)==1 && !sync_mode)
	{
	       //all input nodes are ready
	       graph_task->RunSubgraphTask(this);
	}

 }

void SubgraphTask::SetNodeInputWaitMask(Node * node, std::uint64_t wait_mask)
{
           node->SetAttr("input_wait",wait_mask);
}

std::uint64_t SubgraphTask::GetNodeInputWaitMask(Node * node)
{
          if(!node->ExistAttr("input_wait"))
		  	return 0;
	   return any_cast<std::uint64_t>(node->GetAttr("input_wait"));
}

std::atomic<std::uint64_t> * SubgraphTask::GetNodeInputWaitCounter(Node * node)
{
	return any_cast<std::atomic<std::uint64_t>*>(node->GetAttr("input_wait_counter"));
}

void SubgraphTask::CreateNodeInputWaitCounter(Node * node)
{
	std::atomic<std::uint64_t>* p_counter=new std::atomic<std::uint64_t>();
	node->SetAttr("input_wait_counter",p_counter);
}
   	
void SubgraphTask::ReleaseNodeInputWaitCounter(Node * node)
{
      if(!node->ExistAttr("input_wait_counter"))
	  	return;
	  
       std::atomic<std::uint64_t>* p_counter=any_cast<std::atomic<std::uint64_t>*>(node->GetAttr("input_wait_counter"));
	   
       delete p_counter; 		
}

void SubgraphTask::Release(void)
{
       for(unsigned int i=0;i<sub_graph->input_nodes.size();i++)
       	{
       		Node * node=sub_graph->input_nodes[i];

	       ReleaseNodeInputWaitCounter(node); 	
       	}

}

} //namespace TEngine
