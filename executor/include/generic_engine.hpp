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

#ifndef __GENERIC_ENGINE_HPP__
#define __GENERIC_ENGINE_HPP__

#include <unordered_map>
#include <string>

#include "tengine_c_api.h"
#include "exec_engine.hpp"
#include "graph_executor.hpp"

namespace TEngine {

class DevExecutorManager;
class GraphTask;
struct DevScheduler;

class GenericEngine : public ExecEngine
{
public:
    GenericEngine(void);
    ~GenericEngine(void);

    exec_handle_t AddGraphExecutor(GraphExecutor* graph_executor) override;
    void* GetTensorBuffer(Tensor*, exec_handle_t h = nullptr) override;
    bool SetTensorBuffer(Tensor*, void*, int, exec_handle_t h = nullptr) override;
    bool Prerun(exec_handle_t) override;

    bool Run(exec_handle_t, exec_event_t&) override;
    bool SyncRun(exec_handle_t) override;

    int Wait(exec_handle_t, exec_event_t&, int) override;

    bool SetCallback(exec_handle_t, exec_event_t&, int event, exec_cb_t) override;

    bool Postrun(exec_handle_t) override;

    exec_status_t GetStatus(exec_handle_t) override;
    bool SetEventHook(exec_handle_t, int event, event_handler_t cb_func, void* cb_arg) override;

    const std::string& GetStatusStr(const exec_status_t&) override;

    int GetStatusCode(const exec_status_t&) override;

    std::string GetErrorStr(exec_handle_t) override;
    bool RemoveGraphExecutor(exec_handle_t) override;

    Graph* GetOptimizedGraph(exec_handle_t) override;
    bool OptimizeGraph(exec_handle_t) override;

    DevScheduler* GetScheduler(void)
    {
        return scheduler_;
    }
    bool SetScheduler(const std::string& sched_name);

    bool GetGraphAttr(exec_handle_t, const char*, void*, int) override;
    bool SetGraphAttr(exec_handle_t, const char*, const void*, int) override;

private:
    void Lock(void)
    {
        my_mutex_.lock();
    }

    void Unlock(void)
    {
        my_mutex_.unlock();
    }

    std::mutex my_mutex_;

    std::unordered_map<GraphExecutor*, GraphTask*> graph_map_;

    DevScheduler* scheduler_;
};

}    // namespace TEngine

#endif
