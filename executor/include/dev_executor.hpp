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

#ifndef __DEV_EXECUTOR_HPP__
#define __DEV_EXECUTOR_HPP__

#include "device_driver.hpp"

namespace TEngine {

class SubgraphTask;

struct DevExecutor
{
    /*!
     * @brief Insert new task into device executor
     *
     * @param task the subgraph task to be inserted
     *
     * @return true on success
     */

    virtual bool PrerunTask(SubgraphTask* task) = 0;
    virtual bool SchedTask(SubgraphTask* task) = 0;
    virtual bool SchedTask(void) = 0;
    virtual bool PostrunTask(SubgraphTask* task) = 0;
    virtual bool RunTask(SubgraphTask* task) = 0;
    virtual bool SyncRunTask(SubgraphTask* task) = 0;

    virtual bool GetGraphAttr(SubgraphTask* task, const char* name, void* val, int size) = 0;
    virtual bool SetGraphAttr(SubgraphTask* task, const char* name, const void* val, int size) = 0;
    virtual bool OptimizeGraph(SubgraphTask* task) = 0;
    virtual Subgraph* GetOptimizedGraph(SubgraphTask* task) = 0;

    /*!
     * @brief Get current workload of the device executor
     *
     * @param load the returned result
     *
     * @return none
     */
    virtual void GetWorkload(DevWorkload& load) = 0;
    virtual int GetRunTaskNum(void) = 0;
    virtual int GetReadyTaskNum(void) = 0;
    virtual int GetWaitTaskNum(void) = 0;

    /*!
     * @brief check how fast the dev run the graph, under the policy
     *
     * @param graph the graph/subgraph to be checked
     * @param policy the policy to run the graph: latency/power/throughput
     * @param perf  to receive the evaluation result
     *
     * @return true, if the graph is completely supported by the dev.
     *         otherwise, false
     */

    virtual bool GetPerf(Subgraph* graph, int policy, GraphPerf& perf) = 0;

    /*!
     * @brief Get the nominal performance to run a graph, in terms of Mfops per second
     *
     * @param graph the graph to examine. if graph is nullptr, return the maximum fops rate
     *
     * @return the estimated fops rate for specific graph
     */

    virtual float GetFops(Subgraph* graph, int policy) = 0;

    virtual int GetPolicyPriority(int policy) = 0;

    virtual bool GetProposal(Graph* graph, int policy, bool static_assign) = 0;

    /*
     *
     *
     */
    virtual bool Start(void) = 0;
    virtual bool Stop(void) = 0;
    virtual bool Init(void) = 0;
    virtual bool Release(void) = 0;

    /*
     *
     *
     *
     */

    virtual void UnbindDevice(void) = 0;
    virtual void BindDevice(Device*) = 0;

    virtual dev_status_t GetStatus(void) = 0;

    virtual void SetName(const std::string& name) = 0;
    virtual const std::string& GetName() = 0;
    virtual const dev_id_t& GetDevID() = 0;
    virtual const dev_type_t& GetDevType() = 0;

    virtual ~DevExecutor(){};

    DevExecutor()
    {
        nonblock_run_ = true;
    }

    bool SupportNonblockRun(void)
    {
        return nonblock_run_;
    }
    void DisableNonblockRun(void)
    {
        nonblock_run_ = false;
    }

private:
    bool nonblock_run_;
};
extern template class SpecificFactory<DevExecutor>;
extern template SpecificFactory<DevExecutor> SpecificFactory<DevExecutor>::instance;

using DevExecutorFactory = SpecificFactory<DevExecutor>;

class DevExecutorManager : public SimpleObjectManagerWithLock<DevExecutorManager, DevExecutor*>
{
public:
    static bool RegisterDevExecutor(DevExecutor* dev_executor);
    static bool UnregisterDevExecutor(DevExecutor* dev_executor);
    static bool GetDevExecutorByID(const dev_id_t& dev_id, DevExecutor*& dev_executor);
    static bool GetDefaultDevExecutor(DevExecutor*& dev_executor);
    static bool GetDevExecutorByName(const std::string& dev_name, DevExecutor*& dev_executor);
    static int GetDevExecutorNum(void);

    DevExecutor* default_executor;
};

}    // namespace TEngine

#endif
