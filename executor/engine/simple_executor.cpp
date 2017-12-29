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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#include "simple_executor.hpp"
#include "executor.hpp"
#include "tensor_mem.hpp"
#include "prof_record.hpp"
#include "graph_optimizer.hpp"

#define ATTR_NODE_RUNNER "NodeRunner"

namespace TEngine {

exec_handle_t SimpleExec::AddGraphExecutor(GraphExecutor * graph_executor)
{
     exec_env * env=new exec_env();

     env->graph_executor=graph_executor;
     env->status=EXEC_STATUS_CREATED;
 
     any * ret=new any();

     (*ret)=env;

     return ret;
}

void * SimpleExec::GetTensorBuffer(Tensor * tensor, exec_handle_t h)
{
    return get_tensor_mem(tensor);
}

bool SimpleExec::SetTensorBuffer(Tensor * tensor, void *addr, int size, exec_handle_t h)
{
    return set_tensor_mem(tensor,addr,size,nullptr);
}

bool SimpleExec::Prerun(exec_handle_t h)
{
    exec_env * env=any_cast<exec_env *>(*h);

    GraphExecutor * graph_executor=env->graph_executor;

    Graph * graph=graph_executor->GetGraph();

    GraphOptimizerManager::RunOpt("BNScaleReLu",graph);
    GraphOptimizerManager::RunOpt("ConvReLu",graph);

   if(!BindNodeRunner(graph))
          return false;

    std::vector<Node *>& seq_nodes=graph->seq_nodes;

    for(unsigned int i=0; i<seq_nodes.size();i++)
    {
        Node * node=seq_nodes[i];

        for(unsigned int i=0;i<node->GetOutputNum();i++)
        {

             Tensor * tensor=node->GetOutputTensorSeq(i);

             if(get_tensor_mem(tensor))
                   continue;

              int input_idx=-1;

              /* process inplace */
              if(node->ExistAttr(ATTR_INPLACE))
              {
                  const inplace_t & inplace=any_cast<inplace_t>(node->GetAttr("inplace"));

                  if(inplace.count(i))
                      input_idx=inplace.at(i);
              }
              
             if(input_idx>=0)
              {
                   Tensor * input_tensor=node->GetInputTensorSeq(input_idx);
                   void * tensor_addr=get_tensor_mem(input_tensor);
                   int total_size=tensor->GetTotalSize();

                   set_tensor_mem(tensor,tensor_addr, total_size,nullptr);
              }
              else
              {
                  int total_size=tensor->GetTotalSize();
                  void * tensor_addr=std::malloc(total_size);

                  set_tensor_mem(tensor,tensor_addr,total_size,std::free);
              }
        }
    }

    for(unsigned int i=0;i<seq_nodes.size();i++)
    {
        Node * node=seq_nodes[i];

        if(!node->ExistAttr(ATTR_NODE_RUNNER))
            continue;

        const NodeExec& node_exec=any_cast<NodeExec>(node->GetAttr(ATTR_NODE_RUNNER));

        if(node_exec.pre_run==nullptr)
             continue;

        if(!node_exec.pre_run(node,this))
             return false;
    }

    env->status=EXEC_STATUS_INITED;

    return true;
}

static void parse_node(void * data)
{
   Node * node=(Node *)data;

   std::printf("Node: %d %s ",node->GetNodeIndex(),node->GetName().c_str());

   std::printf(" op: %s",node->GetOp()->GetName().c_str());

   std::cout<<" input: ";
   Tensor * input_tensor=node->GetInputTensor(0);
   TShape& ishape=input_tensor->GetShape();
   ishape.DumpShape(std::cout);

   std::cout<<" output: ";

   Tensor * output_tensor=node->GetOutputTensor(0);
   TShape& oshape=output_tensor->GetShape();
   oshape.DumpShape(std::cout);

   std::printf(" Mfops: %.2f",1.0f*node->GetFops()/1000000);
}


bool SimpleExec::Run(exec_handle_t h,exec_event_t& event)
{
    exec_env * env=any_cast<exec_env *>(*h);

    GraphExecutor * graph_executor=env->graph_executor;

    Graph * graph=graph_executor->GetGraph();

    std::vector<Node *>& seq_nodes=graph->seq_nodes;


    int s=env->status;

    if(s!= EXEC_STATUS_INITED && s!=EXEC_STATUS_DONE)
    {
        return false;
    }

    env->status=EXEC_STATUS_RUN;

    static ProfRecord * prof=ProfRecordManager::Create("simple",seq_nodes.size(),parse_node);

    bool do_prof=false;

    const char * prof_env=std::getenv("PROF_TIME");
    
    if(prof_env && prof_env[0]=='1')
          do_prof=true;

    for(unsigned int i=0; i<seq_nodes.size();i++)
    {
        Node * node=seq_nodes[i];
        Operator * op=node->GetOp();

        bool skip=false;

        for(unsigned int i=0;i<node->GetInputNum();i++)
        {
             TShape& shape=node->GetInputTensor(i)->GetShape();

             if(shape.GetSize()<=0)
             {
                 skip=true;
                 break;
             }
        }

        if(skip==true)
        {
            LOG_INFO()<<"skip node: "<<node->GetName()<<" due to input not ready\n";
            continue;
        }

        if(!node->ExistAttr(ATTR_NODE_RUNNER))
            continue;

        const NodeExec& node_exec=any_cast<NodeExec>(node->GetAttr(ATTR_NODE_RUNNER));

        if(do_prof)
            prof->Start(i,node);
  
        if(node_exec.run==nullptr || !node_exec.run(node,this))
        {
            LOG_ERROR()<<"Failed to execute on: "<<node->GetName() <<" Op: "<<op->GetName()<<std::endl;

            Lock();
            status_map_[graph_executor]=EXEC_STATUS_BAD;
            Unlock();

            return false;

        }

       if(do_prof)
          prof->Stop(i);
    }

    env->status=EXEC_STATUS_DONE;

    return true;
}


bool SimpleExec::Postrun(exec_handle_t h)
{
    exec_env * env=any_cast<exec_env *>(*h);
    GraphExecutor * graph_executor=env->graph_executor;

    Graph * graph=graph_executor->GetGraph();

    std::vector<Node *>& seq_nodes=graph->seq_nodes;

    for(unsigned int i=0; i<seq_nodes.size();i++)
    {
        Node * node=seq_nodes[i];
        Operator * op=node->GetOp();

       if(op->GetName()=="Const" || op->GetName()=="Input")
            continue;

        for(unsigned int i=0;i<node->GetOutputNum();i++)
        {
             Tensor * tensor=node->GetOutputTensorSeq(i);
             free_tensor_mem(tensor);
        }

        const NodeExec& node_exec=any_cast<NodeExec>(node->GetAttr(ATTR_NODE_RUNNER));

        if(node_exec.post_run==nullptr)
             continue;

        if(!node_exec.post_run(node,this))
        {
            LOG_ERROR()<<"Postrun failed for node: "<<node->GetName()<<"\n";
        }
    }


    //if need to dump?
    bool do_prof=false;

    const char * prof_env=std::getenv("PROF_TIME");
    
    if(prof_env && prof_env[0]=='1')
          do_prof=true;

   if(do_prof)
   {
      std::unordered_map<std::string,uint64_t > time_stats;
      ProfRecord * prof=ProfRecordManager::Get("simple");
      ProfTime * time_prof=dynamic_cast<ProfTime *>(prof);
      float total_fops=0;
      int repeat_count;

      int number=time_prof->GetRecordNum();
      uint64_t total_time=0;

      for(int i=0;i<number;i++)
      {
          const ProfTime::TimeRecord * p_record=time_prof->GetRecord(i);

          if(p_record->count==0)
                 continue;

          total_time+=p_record->total_used_time;
          repeat_count=p_record->count;

          Node * node=reinterpret_cast<Node *>(p_record->ident);
          Operator * op=node->GetOp();

           uint64_t op_time;

           if(time_stats.count(op->GetName()))
               op_time=time_stats[op->GetName()];
           else
               op_time=0;

           op_time+=p_record->total_used_time;

           time_stats[op->GetName()]=op_time;

           total_fops+=node->GetFops();
      }

      std::printf("\n======== time stats by operator: repeat %d  =====\n",repeat_count);
      std::printf("total time: %lu us with %.2f Mfops\n",total_time,total_fops/1000000);
      int n=0;

      for(auto ir=time_stats.begin(); ir!=time_stats.end();ir++)
      {
           std::printf("%d: %s used %lu us (%.2f%%)\n",n++,
                 ir->first.c_str(),ir->second, 100.0f*ir->second/total_time);
      }
      std::printf("\n\n");

   }

    return true;
}


exec_status_t SimpleExec::GetStatus(exec_handle_t h) 
{
    exec_env * env=any_cast<exec_env *>(*h);

    return env->status;
}

const std::string& SimpleExec::GetStatusStr(const exec_status_t& status)
{
   static std::string created="CREATED";
   static std::string inited="INITED";
   static std::string run="RUN";
   static std::string done="DONE";
   static std::string bad="BAD";
   static std::string unknown="UNKNOWN";

   int s=any_cast<int>(status);

   switch(s)
   {
      case EXEC_STATUS_CREATED:
           return created;
      case EXEC_STATUS_INITED:
           return inited;
      case EXEC_STATUS_RUN:
           return run;
      case EXEC_STATUS_DONE:
           return done;
      case EXEC_STATUS_BAD:
           return bad;
      default:
           break;
   }

   return unknown; 
}

int SimpleExec::GetStatusCode(const exec_status_t& status)
{
   int s=any_cast<int>(status);

   return s;
}

std::string  SimpleExec::GetErrorStr(exec_handle_t h)
{
    return "NO ERROR:-)\n";
}

bool SimpleExec::RemoveGraphExecutor(exec_handle_t h)
{
    exec_env * env=any_cast<exec_env *>(*h);

    delete env;
    delete h;
    
    return true;
}


bool SimpleExec::BindNodeRunner(Graph * graph)
{
    std::vector<Node *>& seq_nodes=graph->seq_nodes;
    int node_size=seq_nodes.size();

    for(int i=0; i<node_size;i++)
    {
        Node * node=seq_nodes[i];
        Operator * op=node->GetOp();

       if(op->GetName()=="Const" || op->GetName()=="Input")
            continue;
   
        NodeExec node_exec;

        if(!GetNodeExec(op->GetName(),node_exec))
             return false;

        node->SetAttr(ATTR_NODE_RUNNER,node_exec);

        if(node_exec.on_bind!=nullptr)
            node_exec.on_bind(node,this);

    }

    return true;
}




} //namespace TEngine
