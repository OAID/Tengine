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
#ifndef __RESOURCE_CONTAINER_HPP__
#define __RESOURCE_CONTAINER_HPP__

#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>

#include "tengine_config.hpp"
#include "tengine_lock.hpp"
#include "attribute.hpp"
#include "graph_executor.hpp"
#include "safe_object_manager.hpp"

namespace TEngine {


class UserContext;

class RuntimeWorkspace{

public:
    RuntimeWorkspace(const std::string& name) { name_=name;}
    const std::string& GetName(void) { return name_;}

    UserContext * GetUserContext(void) { return context_;}
    void SetUserContext(UserContext * context) { context_=context;}

    GraphExecutor * CreateGraphExecutor(const std::string& graph_name, const std::string& model_name)
    {
         GraphExecutor * executor=new GraphExecutor();

         if(!executor->CreateGraph(graph_name,model_name))
          {
               delete executor;
               return nullptr;
          }

         executor->SetWorkspace(this);

         Lock();

         executor_list_.push_back(executor);

         Unlock();

         return executor;
    }


    bool DestroyGraphExecutor(GraphExecutor * executor)
    {
          Lock();

          std::vector<GraphExecutor *>::iterator ir=executor_list_.begin();

          while(ir!=executor_list_.end())
          {
              if(*ir==executor)
              {
                  executor_list_.erase(ir);
                  ReleaseGraphExecutor(executor);
                  Unlock();
                  return true;
              }
              ir++;
          }

          Unlock();

          return false;
    }

    ~RuntimeWorkspace(void) 
    {
        for(unsigned int i=0;i<executor_list_.size();i++)
        {
              GraphExecutor * graph_executor=executor_list_[i];
              ReleaseGraphExecutor(graph_executor);
        }

    }

private:

    void Lock(void)
    {
         TEngineLock(my_mutex);
    }

    void Unlock(void)
    {
         TEngineUnlock(my_mutex);
    }

    void ReleaseGraphExecutor(GraphExecutor * executor)
    {
         std::cout<<"Release Graph Executor for graph "<< executor->GetGraphName()<<"\n";
         delete executor;
    }

    Attribute config_;
    std::string name_;
    UserContext * context_;
    std::vector<GraphExecutor*> executor_list_;
    std::mutex  my_mutex;
    

};


class UserContext {

public:
     UserContext(const std::string& name) {name_=name;max_ws_=10;}
     const std::string& GetName(void) { return name_;}

     RuntimeWorkspace * CreateWorkspace(const std::string& ws_name)
     {
 
          RuntimeWorkspace * ws=new RuntimeWorkspace(ws_name);
          ws->SetUserContext(this);

          Lock();
          ws_list.push_back(ws);
          Unlock();
          return ws;
     }

     RuntimeWorkspace * FindWorkspace(const std::string& ws_name)
     {
          Lock();

          for(size_t i=0;i<ws_list.size();i++)
          {
            RuntimeWorkspace * ws=ws_list[i];

            if(ws->GetName()==ws_name)
             {
                Unlock();
                return ws;
              }
          }

          Unlock();
          return nullptr;
     }

     bool DestroyWorkspace(RuntimeWorkspace * ws)
     {
          Lock();

          std::vector<RuntimeWorkspace *>::iterator ir=ws_list.begin();

          while(ir!=ws_list.end())
          {
              if(*ir==ws)
              {
                  ws_list.erase(ir);
                  ReleaseWorkspace(ws);
                  Unlock();

                  return true;
              }
          }

          Unlock();
          return false;

     }

     ~UserContext(void)
     {
        for(unsigned int i=0;i<ws_list.size();i++)
        {
              RuntimeWorkspace * ws=ws_list[i];
              ReleaseWorkspace(ws);
           
        }
     }


private:

     void Lock(void)
     { 
          TEngineLock(ws_mutex);
     }

     void Unlock(void)
     {
         TEngineUnlock(ws_mutex);
     }

     void ReleaseWorkspace(RuntimeWorkspace * ws)
     {
         std::cout<<"Release workspace "<<ws->GetName()<<" resource\n";
         delete ws;
     }

     Attribute config;
     int max_ws_;
     std::string name_;
     std::vector<RuntimeWorkspace *> ws_list;
     std::mutex  ws_mutex;

};



class UserContextManager: public SimpleObjectManagerWithLock<UserContextManager, UserContext *> {};

} //namespace TEngine

#endif
