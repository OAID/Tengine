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

#include <functional>

#include "simple_engine.hpp"
#include "tensor_mem.hpp"
#include "logger.hpp"
#include "tengine_config.hpp"


namespace TEngine {

SimpleEngine::SimpleEngine(void)
{
     name="simple";
     backend_dev_=nullptr;
}
 
SimpleEngine:: ~SimpleEngine(void)
{
}

   
exec_handle_t SimpleEngine::AddGraphExecutor(GraphExecutor * graph_executor)
{
    ExecHandle * new_handle=new ExecHandle();
    new_handle->graph_executor=graph_executor;
    new_handle->status=EXEC_STATUS_CREATED;

    any * ret=new any();

    (*ret)=new_handle;

    return ret;
}

void * SimpleEngine::GetTensorBuffer(Tensor * tensor, exec_handle_t h)
{
    return get_tensor_mem(tensor);
}

bool SimpleEngine::SetTensorBuffer(Tensor * tensor, void *addr, int size, exec_handle_t h)
{
    return set_tensor_mem(tensor,addr,size,nullptr);
}

bool SimpleEngine::Prerun(exec_handle_t h)
{
     if(backend_dev_==nullptr)
     {
         std::string default_dev_id;

         if(!TEngineConfig::Get("device.default",default_dev_id))
             return false;

         Device * device=DriverManager::GetDevice(default_dev_id);

         if(device==nullptr)
             return false;

        backend_dev_=device;

     }

     ExecHandle * handle=any_cast<ExecHandle*>(*h);     

     Graph * graph=handle->graph_executor->GetGraph();

     void * graph_handle=backend_dev_->CreateGraphHandle(graph);

     if(graph_handle==nullptr)
           return false;


     if(!backend_dev_->OptimizeGraph(graph_handle))
         return false;

     if(!backend_dev_->Prerun(graph_handle))
         return false;

     auto f=std::bind(&SimpleEngine::OnGraphDone,this,std::placeholders::_1,std::placeholders::_2);
     backend_dev_->SetGraphDoneHook(graph_handle,dev_graph_cb_t(f));

     handle->graph_handle=graph_handle;
     handle->status=EXEC_STATUS_INITED;

     return true;
    
}

void SimpleEngine::OnGraphDone(Graph * graph, bool exec_success)
{
    WaitEvent * p_event=any_cast<WaitEvent *>(graph->GetAttr("wait_event"));
    ExecHandle * handle=any_cast<ExecHandle*>(graph->GetAttr("exec_handle"));     
 
    std::unique_lock<std::mutex> lock(p_event->mutex);
    p_event->task_done=true;

    if(exec_success)
         handle->status=EXEC_STATUS_READY;
    else
         handle->status=EXEC_STATUS_BAD;

    lock.unlock();
     p_event->cond.notify_all();


}

bool SimpleEngine::SyncRun(exec_handle_t h)
{
    ExecHandle * handle=any_cast<ExecHandle*>(*h);

    if(handle->status!=EXEC_STATUS_INITED &&
       handle->status!=EXEC_STATUS_READY)
    {
          XLOG_ERROR()<<"bad status: "<<GetStatusStr(handle->status)<<"\n";
	  return false;
    }	


    handle->status=EXEC_STATUS_RUN;

    void * graph_handle=handle->graph_handle;
  
    if(backend_dev_->SyncRun(graph_handle))
    {
         handle->status=EXEC_STATUS_READY;
         return true; 
    }

    handle->status=EXEC_STATUS_BAD;
    return false;
}

bool SimpleEngine::Run(exec_handle_t h,exec_event_t& event)
{
    ExecHandle * handle=any_cast<ExecHandle*>(*h);

    if(handle->status!=EXEC_STATUS_INITED &&
       handle->status!=EXEC_STATUS_READY)
    {
          XLOG_ERROR()<<"bad status: "<<GetStatusStr(handle->status)<<"\n";
	  return false;
    }	

    WaitEvent * p_event=new WaitEvent();
    p_event->wait_count=0;
    p_event->task_done=false;
    event=p_event;

    Graph * graph=handle->graph_executor->GetGraph();

    graph->SetAttr("wait_event",p_event);
    graph->SetAttr("exec_handle",handle);
    
    void * graph_handle=handle->graph_handle;

    handle->status=EXEC_STATUS_RUN;

    if(backend_dev_->Run(graph_handle))
         return true;

    handle->status=EXEC_STATUS_BAD;

    delete p_event;

    return false;
}


bool SimpleEngine::Postrun(exec_handle_t h)
{
    ExecHandle * handle=any_cast<ExecHandle*>(*h);
    void * graph_handle=handle->graph_handle;

    return backend_dev_->Postrun(graph_handle);
}


exec_status_t SimpleEngine::GetStatus(exec_handle_t h) 
{
     ExecHandle * handle=any_cast<ExecHandle*>(*h);

     return handle->status;     

}

const std::string& SimpleEngine::GetStatusStr(const exec_status_t& status)
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

int SimpleEngine::GetStatusCode(const exec_status_t& status)
{
   int s=any_cast<int>(status);

   return s;
}

std::string  SimpleEngine::GetErrorStr(exec_handle_t h)
{
    return "NO ERROR:-)\n";
}

bool SimpleEngine::RemoveGraphExecutor(exec_handle_t h)
{
    ExecHandle * handle=any_cast<ExecHandle*>(*h);
    void * graph_handle=handle->graph_handle;
     
    backend_dev_->ReleaseGraphHandle(graph_handle);

    delete handle;
    delete h;

    return true;
}

int SimpleEngine::Wait(exec_handle_t h, exec_event_t& e, int try_wait)
{
     WaitEvent * p_event=any_cast<WaitEvent *>(e);
		
     std::unique_lock<std::mutex> lock(p_event->mutex);

     if(try_wait && !p_event->task_done)
     {
         lock.unlock();
         return 0;
     }

     p_event->wait_count++;

     if(!p_event->task_done)
         p_event->cond.wait(lock,[p_event]{return p_event->task_done;});

    if(p_event->wait_count.fetch_sub(1)==1)
    {
       	 delete p_event;
    }
    else
    {
         lock.unlock();
    }

    return 1;
}

Graph * SimpleEngine::GetOptimizedGraph(exec_handle_t h)
{
    if(backend_dev_==nullptr)
       return nullptr;

    ExecHandle * handle=any_cast<ExecHandle*>(*h);

    return backend_dev_->GetOptimizedGraph(handle->graph_handle);
}

bool SimpleEngine::OptimizeGraph(exec_handle_t h)
{
    return false;
}


bool SimpleEngine::SetCallback(exec_handle_t h , exec_event_t& e, int event, exec_cb_t cb) 
{
    return false;
}


} //namespace TEngine
