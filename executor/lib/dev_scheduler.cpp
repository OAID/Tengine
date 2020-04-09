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

#include "logger.hpp"
#include "dev_scheduler.hpp"
#include "dev_executor.hpp"
#include "graph_task.hpp"

namespace TEngine {

int DevScheduler::MapPolicy(const std::string& policy)
{
    return 1;
}

class SimpleScheduler : public DevScheduler
{
public:
    SimpleScheduler()
    {
        sched_name = "Simple";
    }

    bool SyncRunTask(GenericEngine* engine, DevExecutor* dev, SubgraphTask* task) override;
    bool SchedTask(GenericEngine* engine, DevExecutor* dev, SubgraphTask* task) override;
    bool PrerunTask(GenericEngine*, DevExecutor* dev, SubgraphTask* task) override;
    bool PostrunTask(GenericEngine*, DevExecutor* dev, SubgraphTask* task) override;
    bool AbortTask(GenericEngine* engine, DevExecutor* dev, SubgraphTask* task) override;
    bool SuspendTask(GenericEngine* engine, DevExecutor* dev, SubgraphTask* task) override;
    bool ResumeTask(GenericEngine* engine, DevExecutor* dev, SubgraphTask* task) override;
};

bool SimpleScheduler::SyncRunTask(GenericEngine* engine, DevExecutor* dev, SubgraphTask* task)
{
    return dev->SyncRunTask(task);
}

bool SimpleScheduler::SchedTask(GenericEngine* engine, DevExecutor* dev, SubgraphTask* task)
{
    return dev->SchedTask(task);
}

bool SimpleScheduler::PrerunTask(GenericEngine*, DevExecutor* dev, SubgraphTask* task)
{
    return dev->PrerunTask(task);
}

bool SimpleScheduler::PostrunTask(GenericEngine*, DevExecutor* dev, SubgraphTask* task)
{
    dev->PostrunTask(task);

    return true;
}

bool SimpleScheduler::AbortTask(GenericEngine* engine, DevExecutor* dev, SubgraphTask* task)
{
    XLOG_ERROR() << "NOT SUPPORT\n";
    return false;
}

bool SimpleScheduler::SuspendTask(GenericEngine* engine, DevExecutor* dev, SubgraphTask* task)
{
    XLOG_ERROR() << "NOT SUPPORT\n";
    return false;
}

bool SimpleScheduler::ResumeTask(GenericEngine* engine, DevExecutor* dev, SubgraphTask* task)
{
    XLOG_ERROR() << "NOT SUPPORT\n";
    return false;
}

void DevSchedulerManager::OnDevExecutorRegistered(DevExecutor* dev_executor)
{
    DevSchedulerManager* manager = GetInstance();

    LockExecutorList();

    manager->executor_list.push_back(dev_executor);

    UnlockExecutorList();
}

void DevSchedulerManager::OnDevExecutorUnregistered(DevExecutor* dev_executor)
{
    DevSchedulerManager* manager = GetInstance();

    LockExecutorList();

    auto ir = manager->executor_list.begin();
    auto end = manager->executor_list.end();

    while(ir != end)
    {
        if(*ir == dev_executor)
        {
            manager->executor_list.erase(ir);
            break;
        }

        ir++;
    }

    UnlockExecutorList();
}

void DevSchedulerManager::LockExecutorList(void)
{
    DevSchedulerManager* manager = GetInstance();
    manager->list_lock.lock();
}

void DevSchedulerManager::UnlockExecutorList(void)
{
    DevSchedulerManager* manager = GetInstance();
    manager->list_lock.unlock();
}

/* for init */

void DevSchedulerManagerInit(void)
{
    SimpleScheduler* sched = new SimpleScheduler();

    DevSchedulerManager::Add(sched->GetName(), DevSchedulerPtr(sched));
}

}    // namespace TEngine
