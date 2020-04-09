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

#ifndef __DEV_SCHEDULER_HPP__
#define __DEV_SCHEDULER_HPP__

#include <string>
#include <vector>
#include <mutex>

#include "generic_factory.hpp"
#include "simple_object_manager.hpp"

namespace TEngine {

class SubgraphTask;
struct DevExecutor;
class GenericEngine;

struct DevScheduler
{
    virtual bool SchedTask(GenericEngine* engine, DevExecutor* dev, SubgraphTask* task) = 0;
    virtual bool SyncRunTask(GenericEngine* engine, DevExecutor* dev, SubgraphTask* task) = 0;
    virtual bool PrerunTask(GenericEngine*, DevExecutor* dev, SubgraphTask* task) = 0;
    virtual bool PostrunTask(GenericEngine*, DevExecutor* dev, SubgraphTask* task) = 0;
    virtual bool AbortTask(GenericEngine* engine, DevExecutor* dev, SubgraphTask* task) = 0;
    virtual bool SuspendTask(GenericEngine* engine, DevExecutor* dev, SubgraphTask* task) = 0;
    virtual bool ResumeTask(GenericEngine* engine, DevExecutor* dev, SubgraphTask* task) = 0;
    virtual ~DevScheduler() {}

    static int MapPolicy(const std::string& policy);

    const std::string& GetName(void)
    {
        return sched_name;
    }

    std::string sched_name;
};

using DevSchedulerFactory = SpecificFactory<DevScheduler>;
using DevSchedulerPtr = std::shared_ptr<DevScheduler>;

class DevSchedulerManager : public SimpleObjectManager<DevSchedulerManager, DevSchedulerPtr>
{
public:
    static void OnDevExecutorRegistered(DevExecutor* dev_executor);
    static void OnDevExecutorUnregistered(DevExecutor* dev_executor);

    static void LockExecutorList(void);
    static void UnlockExecutorList(void);

    std::mutex list_lock;
    std::vector<DevExecutor*> executor_list;
};

}    // namespace TEngine

#endif
