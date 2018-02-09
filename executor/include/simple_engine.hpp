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

#ifndef __SIMPLE_ENGINE_HPP__
#define __SIMPLE_ENGINE_HPP__

#include <unordered_map>
#include <string>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "exec_engine.hpp"
#include "tengine_lock.hpp"
#include "graph_executor.hpp"
#include "device_driver.hpp"

namespace TEngine {


class SimpleEngine: public ExecEngine {

public:

    struct WaitEvent {
	std::mutex mutex;
	std::condition_variable cond; 
	std::atomic<int> wait_count;
        bool task_done;
    };

    struct ExecHandle {
        GraphExecutor * graph_executor;
        void * graph_handle;
        int status;
    };

    SimpleEngine(void);
   ~SimpleEngine(void);
   
    exec_handle_t AddGraphExecutor(GraphExecutor *graph_executor);
    void * GetTensorBuffer(Tensor *, exec_handle_t h=nullptr) override;
    bool SetTensorBuffer(Tensor *, void *, int, exec_handle_t h=nullptr) override;
    bool   Prerun(exec_handle_t) override;

    bool Run(exec_handle_t,exec_event_t&) override;
    bool SyncRun(exec_handle_t) override;

    int Wait(exec_handle_t, exec_event_t&, int try_wait) override; 

    bool SetCallback(exec_handle_t, exec_event_t&, int event, exec_cb_t) override; 

    bool Postrun(exec_handle_t) override;

    exec_status_t GetStatus(exec_handle_t) override;

    const std::string& GetStatusStr(const exec_status_t&) override; 

    int GetStatusCode(const exec_status_t&) override;

    std::string    GetErrorStr(exec_handle_t) override;
    bool RemoveGraphExecutor(exec_handle_t) override;


    void OnGraphDone(Graph * graph, bool exec_success);
    
private:

    void Lock(void)
    {
       TEngineLock(my_mutex_);
    }

    void Unlock(void)
    {
       TEngineUnlock(my_mutex_);
    }

    std::mutex   my_mutex_;
    Device *     backend_dev_;

};


} //namespace TEngine


#endif
