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
#ifndef __EXEC_ENGINE_HPP__
#define __EXEC_ENGINE_HPP__

#include <memory>

#include "any.hpp"
#include "safe_object_manager.hpp"
#include "generic_factory.hpp"

#define EXEC_STATUS_CREATED 0
#define EXEC_STATUS_INITED   1
#define EXEC_STATUS_WAIT      2
#define EXEC_STATUS_READY    3
#define EXEC_STATUS_RUN        4
#define EXEC_STATUS_DONE      5
#define EXEC_STATUS_BAD        6
#define EXEC_STATUS_INVALID 7

namespace TEngine {

class Graph;
class GraphExecutor;
class Tensor;
struct ExecEngine;

using exec_status_t=any;
using exec_event_t=any;
using exec_cb_t=std::function<void(GraphExecutor *, ExecEngine *, int event, int status)>;
using exec_handle_t=any *;

struct ExecEngine 
{
    virtual void Test(void) {}
    virtual exec_handle_t AddGraphExecutor(GraphExecutor *)=0;

    virtual void * GetTensorBuffer(Tensor *,exec_handle_t=nullptr)=0;
    virtual bool SetTensorBuffer(Tensor *, void *, int, exec_handle_t=nullptr)=0;
    virtual bool   Prerun(exec_handle_t)=0;

    virtual bool Run(exec_handle_t,exec_event_t&)=0;
    virtual bool SyncRun(exec_handle_t)=0;
    virtual int  Wait(exec_handle_t,exec_event_t&,int try_wait=0)=0;
    virtual bool SetCallback(exec_handle_t, exec_event_t&, int event, exec_cb_t)=0;

    virtual bool Postrun(exec_handle_t)=0;

    virtual exec_status_t GetStatus(exec_handle_t)=0;

    virtual int GetStatusCode(const exec_status_t&)=0;
    virtual const std::string& GetStatusStr(const exec_status_t&)=0; 

    virtual std::string    GetErrorStr(exec_handle_t)=0;

    virtual bool RemoveGraphExecutor(exec_handle_t)=0;

    virtual Graph * GetOptimizedGraph(exec_handle_t) { return nullptr;}
    virtual bool OptimizeGraph(exec_handle_t) { return false;}
   
    virtual ~ExecEngine(){}
	
    const std::string& GetName() { return name;}
    std::string name;
	
};

using ExecEnginePtr=std::shared_ptr<ExecEngine>;
using ExecEngFactory=SpecificFactory<ExecEngine>;

class ExecEngineManager: public SimpleObjectManagerWithLock<ExecEngineManager,ExecEnginePtr> {};

} //namespace TEngine


#endif
