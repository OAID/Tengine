
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
 * Author: haoluo@openailab.com
 */

#ifndef __ACL_GRAPH_DEVICE_HPP__
#define __ACL_GRAPH_DEVICE_HPP__

#include "device_driver.hpp"
#include "node_dev_driver.hpp"

#include "graph_optimizer.hpp"
#include "worker_thread.hpp"

#include "acl_graph.hpp"

using namespace arm_compute;

namespace TEngine {

struct DevContext
{
    Device* dev;
    Subgraph* sub_graph;
    Subgraph* optimized_graph;
    CLGraph* graph;
    dev_graph_cb_t graph_cb;
};

struct acl_task
{
    DevContext* context;
};

class ACLDevice : public Device
{
public:
    ACLDevice(const char* dev_name) : Device(dev_name)
    {
        thread_ = nullptr;
        pcScratchMem_ = NULL;
        l32ScratchMemSize_ = 0;
    }
    ~ACLDevice()
    {
        delete pcScratchMem_;
        Kill();
    }

    bool RealPrerun(DevContext* context);
    bool RealRun(DevContext* context);
    bool RealSyncRun(DevContext* context);
    bool RealPostrun(DevContext* context);
    bool RealOptimizeGraph(DevContext* context, Subgraph* graph);

    void RunGraph(DevContext* context, dev_graph_cb_t graph_cb);

    void Process(const acl_task& task, int acl_id);
    void Launch(void);
    void WaitDone(void);
    void Kill(void);

    void IncRequest(int req_number);
    void IncDone(int done_number);

    void PushTask(std::vector<acl_task>& task_list);

    dev_status_t dev_status;

private:
    WorkerThread<acl_task>* thread_;
    std::mutex queue_lock_;
    std::condition_variable queue_cv_;
    std::queue<acl_task> task_queue_;

    std::atomic<uint64_t> request_;
    std::atomic<uint64_t> done_;
    std::mutex wait_mutex_;
    std::condition_variable wait_cv_;

    DataType data_type_;

    char* pcScratchMem_;
    int l32ScratchMemSize_;
    int l32AclNHWCOptimizeFlag_;    // OPT FLAG�� 0: DO NOTHING;
                                    // 1: FORECE RUNNING IN NHWC MODE
};
}    // namespace TEngine

#endif
