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
#ifndef __CPU_ENGINE_HPP__
#define __CPU_ENGINE_HPP__
#include <unordered_map>
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>


#include "soc_runner.hpp"
#include "exec_engine.hpp"
#include "tengine_lock.hpp"
#include "graph_executor.hpp"

namespace TEngine {

class CPUEngine: public ExecEngine 
{
    struct exec_env 
    {
        GraphExecutor * graph_executor;
        void * graph_handle;
        int status;
    };

public:
    CPUEngine(void); 
    ~CPUEngine(void);

    exec_handle_t AddGraphExecutor(GraphExecutor *graph_executor);

    void * GetTensorBuffer(Tensor *, exec_handle_t h=nullptr) override;
    bool SetTensorBuffer(Tensor *, void *, int, exec_handle_t h=nullptr) override;
    bool Prerun(exec_handle_t) override;

    bool SyncRun(exec_handle_t) override;
    bool Run(exec_handle_t h,exec_event_t&) override { return SyncRun(h);} 
    int  Wait(exec_handle_t, exec_event_t&, int ) override { return 1;}; 
    bool SetCallback(exec_handle_t, exec_event_t&, int event, exec_cb_t) override { return false; }

    bool Postrun(exec_handle_t) override;

    exec_status_t GetStatus(exec_handle_t) override;

    const std::string& GetStatusStr(const exec_status_t&) override; 

    int GetStatusCode(const exec_status_t&) override;

    std::string    GetErrorStr(exec_handle_t) override;
    bool RemoveGraphExecutor(exec_handle_t) override;

    bool TaskDispatch(std::vector<sub_op_task>& tasks, int cpu);
    void  Aider(int cpu);
    void  LaunchAider(SocInfo * soc_info);

protected:
    CPURunner  backend_runner;
    bool initialized_;

    std::mutex queue_lock;
    std::condition_variable queue_cv;
    std::queue<sub_op_task> task_queue;

    std::vector<std::thread *> thread_list;



};


} //namespace TEngine

#endif
