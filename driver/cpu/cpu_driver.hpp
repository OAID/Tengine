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
#ifndef __CPU_DRIVER_HPP__
#define __CPU_DRIVER_HPP__

#include <vector>
#include <string>
#include <atomic>
#include <queue>
#include <thread>
#include <condition_variable>

#include "cpu_device.h"
#include "cpu_info.hpp"
#include "node_ops.hpp"
#include "cpu_runner.hpp"

#include "graph.hpp"
#include "device_driver.hpp"
#include "worker_thread.hpp"

#include "graph_perf.hpp"

namespace TEngine {

class CPUDevice;

namespace cpu_driver {

struct DevContext
{
    CPUDevice* dev;
    Subgraph* sub_graph;
    Subgraph* optimized_graph;
    dev_graph_cb_t graph_cb;
};

struct cpu_task
{
    struct DevContext* context;
};
}    // namespace cpu_driver

using cpu_task = cpu_driver::cpu_task;
using DevContext = cpu_driver::DevContext;

class CPUDevice : public Device
{
public:
    CPUDevice(const char* dev_name, const struct cpu_info* dev_def) : Device(dev_name), cpu_info_(dev_def)
    {
        master_thread_ = nullptr;
        done_ = 0;
        request_ = 0;

        /* backend runner */
        backend_runner_.AttachCPUDevice(this);
    }

    virtual ~CPUDevice()
    {
        if(master_thread_)
            delete master_thread_;

        for(auto tr : aider_threads_)
            delete tr;
    }

    const CPUInfo* GetCPUInfo(void)
    {
        return &cpu_info_;
    }

    void RunGraph(Subgraph* sub_graph, dev_graph_cb_t graph_cb)
    {
        bool ret = RealRun(sub_graph);

        if(graph_cb)
            graph_cb(sub_graph, ret);
    }

    void MasterProcess(const cpu_task& task, int cpu_id)
    {
        RunGraph(task.context->optimized_graph, task.context->graph_cb);
    }

    void AiderProcess(const sub_op_task& task, int cpu_id)
    {
        task.exec_func(cpu_id, task.seq, task.data);
    }

    bool SetGraphPerfStat(Subgraph* graph, int action)
    {
        return backend_runner_.SetGraphPerfStat(graph, action);
    }

    int GetGraphPerfStat(Subgraph* graph, struct perf_info** buf, int buf_size)
    {
        return backend_runner_.GetGraphPerfStat(graph, buf, buf_size);
    }

    void LaunchMaster(void)
    {
        auto f = std::bind(&CPUDevice::MasterProcess, this, std::placeholders::_1, std::placeholders::_2);

        master_thread_ = new WorkerThread<cpu_task>(f, cpu_info_.master_cpu);

        master_thread_->SetQueue(&master_task_queue_, &master_queue_lock_, &master_queue_cv_);
        master_thread_->LaunchWorker(true);

        master_thread_->Activate(cpu_info_.master_cpu);
    }

    void LaunchAider(void)
    {
        if(cpu_info_.GetCPUNumber() == 1)
            return;

        auto f = std::bind(&CPUDevice::AiderProcess, this, std::placeholders::_1, std::placeholders::_2);

        for(int i = 0; i < cpu_info_.GetCPUNumber(); i++)
        {
            int cpu = cpu_info_.GetOnlineCPU(i);

            WorkerThread<sub_op_task>* tr = new WorkerThread<sub_op_task>(f, cpu);

            tr->SetQueue(&aider_task_queue_, &aider_queue_lock_, &aider_queue_cv_);

            auto inc_req = std::bind(&CPUDevice::IncRequest, this, std::placeholders::_1);
            auto inc_done = std::bind(&CPUDevice::IncDone, this, std::placeholders::_1);

            tr->SetCount(inc_req, inc_done);

            aider_threads_.push_back(tr);

            tr->LaunchWorker();
        }
    }

    void WaitDone(void)
    {
        std::unique_lock<std::mutex> lock(wait_mutex_);

        if(done_ != request_)
            wait_cv_.wait(lock, [this] { return done_ == request_; });

        lock.unlock();
    }

    void IncRequest(int req_number)
    {
        request_ += req_number;
    }

    void IncDone(int done_number)
    {
        uint64_t prev_val = done_.fetch_add(done_number);

        if(prev_val + done_number == request_)
        {
            std::unique_lock<std::mutex> lock(wait_mutex_);

            wait_cv_.notify_all();

            lock.unlock();
        }
    }

    bool PushAiderTask(std::vector<sub_op_task>& task_list, int cpu)
    {
        auto tr = aider_threads_[0];

        tr->PushTask(task_list);

        return true;
    }

    void PushMasterTask(std::vector<cpu_task>& task_list)
    {
        master_thread_->PushTask(task_list);
    }

    void KillMaster(void)
    {
        if(master_thread_)
        {
            master_thread_->Deactivate();
            delete master_thread_;
            master_thread_ = nullptr;
        }
    }

    void KillAider(void)
    {
        if(aider_threads_.size() > 0)
        {
            for(auto& tr : aider_threads_)
                delete tr;

            aider_threads_.clear();
        }
    }

    bool RealPrerun(DevContext* context)
    {
        prerun_lock_.lock();
        bool ret = backend_runner_.Prerun(context->optimized_graph);
        prerun_lock_.unlock();

        return ret;
    }

    bool RealPostrun(DevContext* context)
    {
        postrun_lock_.lock();
        bool ret = backend_runner_.Postrun(context->optimized_graph);
        postrun_lock_.unlock();

        return ret;
    }

    bool RealRun(Subgraph* graph)
    {
        ActivateWorker();
        run_lock_.lock();
        bool ret = backend_runner_.Run(graph);
        run_lock_.unlock();
        DeActivateWorker();

        return ret;
    }

    void ActivateWorker(void)
    {
       int master_cpu=cpu_info_.master_cpu;

       for(auto t: aider_threads_)
            t->Activate(master_cpu);
    }

    void DeActivateWorker(void)
    {
       for(auto t: aider_threads_)
            t->Deactivate();
    }

    bool RealOptimizeGraph(DevContext* context, Subgraph* graph)
    {
        context->optimized_graph = graph;

        return backend_runner_.OptimizeGraph(context->optimized_graph);
    }

    dev_status_t dev_status;

private:
    WorkerThread<cpu_task>* master_thread_;
    std::vector<WorkerThread<sub_op_task>*> aider_threads_;

    std::mutex master_queue_lock_;
    std::condition_variable master_queue_cv_;
    std::queue<cpu_task> master_task_queue_;

    std::mutex aider_queue_lock_;
    std::condition_variable aider_queue_cv_;
    std::queue<sub_op_task> aider_task_queue_;

    CPUInfo cpu_info_;
    CPURunner backend_runner_;

    std::atomic<uint64_t> request_;
    std::atomic<uint64_t> done_;
    std::mutex wait_mutex_;
    std::condition_variable wait_cv_;

    std::mutex prerun_lock_;
    std::mutex postrun_lock_;
    std::mutex run_lock_;
};

class CPUDriver : public Driver
{
public:
    CPUDriver()
    {
        SetName("CPU");
        auto_probe_ = false;
        InitGraphAttrIO();
    }

    bool InitializeDevice(Device* device) override;
    bool ReleaseDevice(Device* device) override;

    bool StartDevice(Device* device) override;
    bool StopDevice(Device* device) override;

    dev_status_t GetDeviceStatus(Device* device) override;

    void* CreateGraphHandle(Device* dev, Subgraph* graph) override;
    void* CreateGraphHandle(Device* dev) override;
    bool ReleaseGraphHandle(Device* dev, void* graph_handle) override;

    void SetGraphDoneHook(Device* dev, void* graph_handle, dev_graph_cb_t func) override;
    void SetNodeDoneHook(Device* dev, void* node_handle, dev_node_cb_t func) override{};

    bool OptimizeGraph(Device* dev, void* graph_handle, Subgraph* graph) override;
    bool OptimizeGraph(Device* dev, void* graph_handle) override;

    Subgraph* GetOptimizedGraph(Device* dev, void* graph_handle) override;

    bool Prerun(Device* dev, void* graph_handle) override;
    bool Run(Device* dev, void* graph_handle) override;
    bool SyncRun(Device* dev, void* graph_handle) override;
    bool Postrun(Device* dev, void* graph_handle) override;

    bool SetGraphAttr(Device*, void*, const char*, const void*, int) override;
    bool GetGraphAttr(Device*, void*, const char*, void*, int) override;

    bool SetDevAttr(Device* dev, const char* attr_name, const void* val, int size) override;
    bool GetDevAttr(Device* dev, const char* attr_name, void* val, int size) override;

    bool Prerun(Device* dev, void* node_handle, Node* node) override
    {
        return false;
    }
    bool Run(Device* dev, void* node_handle, Node* node) override
    {
        return false;
    }
    bool SyncRun(Device* dev, void* node_handle, Node* node) override
    {
        return false;
    }
    bool Postrun(Device* dev, void* node_handle, Node* node) override
    {
        return false;
    }

    void PushGraph(CPUDevice* cpu_info, DevContext* context);

    bool GetWorkload(Device* dev, DevWorkload& load) override
    {
        return false;
    }
    bool GetPerf(Device* dev, Subgraph* graph, int policy, GraphPerf& perf) override
    {
        return false;
    }
    float GetFops(Device* dev, Subgraph* graph, int policy) override
    {
        return false;
    }
    int GetPolicyPriority(Device* dev, int policy) override
    {
        return false;
    }
    bool GetProposal(Device* dev, Graph* graph, int policy, bool static_assign) override;

    /*
      Since there are some many different CPU, there is no predefined CPU inside now.
      so, the probe function does not work.
      The new interface: AddDevice is to insert a CPUDevice into drvier management system
    */

    void AddDevice(CPUDevice* new_device);    // a special interface for  CPU Device

    int ProbeDevice(void) override
    {
        return 0;
    }
    bool ProbeDevice(const dev_id_t& dev_id) override
    {
        return false;
    }

    int DestroyDevice(void) override;
    bool DestroyDevice(Device* device) override;

    int GetDeviceNum(void) override;
    Device* GetDevice(int idx) override;
    Device* GetDevice(const std::string& name) override;

protected:
    using get_func_t = std::function<bool(DevContext* context, Subgraph* graph, const char* name, void*, int)>;

    using set_func_t = std::function<bool(DevContext* context, Subgraph* graph, const char* name, const void*, int)>;

    void InitGraphAttrIO(void);

    bool OnSetGraphAttr(DevContext* context, Subgraph* graph, const char* attr_name, const void* val, int size);

    bool OnGetGraphAttr(DevContext* context, Subgraph* graph, const char* attr_name, void* val, int size);

    bool OnGetNodeDumpAttr(DevContext* context, Subgraph* graph, const char* name, void*, int);

    bool OnGetGraphPerfAttr(DevContext* context, Subgraph* graph, const char* name, void*, int);

    bool OnSetNodeDumpAttr(DevContext* context, Subgraph* graph, const char* name, const void*, int);

    bool OnSetGraphPerfAttr(DevContext* context, Subgraph* graph, const char* name, const void*, int);

    std::unordered_map<std::string, CPUDevice*> device_table_;
    std::unordered_map<std::string, get_func_t> get_attr_table_;
    std::unordered_map<std::string, set_func_t> set_attr_table_;
};

}    // namespace TEngine

#endif
