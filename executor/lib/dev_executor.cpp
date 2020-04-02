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

#include "dev_executor.hpp"
#include "dev_allocator.hpp"
#include "dev_scheduler.hpp"
#include "tengine_config.hpp"
#include "device_driver.hpp"

namespace TEngine {

// template SpecificFactory<DevExecutor> SpecificFactory<DevExecutor>::instance;
template class SpecificFactory<DevExecutor>;

bool DevExecutorManager::RegisterDevExecutor(DevExecutor* dev_executor)
{
    // add the new created dev_executor into executor manager
    // one dev_id only could has one dev_executor at any time
    if(!SafeAdd(dev_executor->GetDevID(), dev_executor))
        return false;

    // add the new created dev_executor to dev_allocator_manager
    DevAllocatorManager::OnDevExecutorRegistered(dev_executor);

    // add the new created dev_executor to  dev_scheduler_manager
    DevSchedulerManager::OnDevExecutorRegistered(dev_executor);

    return true;
}

bool DevExecutorManager::UnregisterDevExecutor(DevExecutor* dev_executor)
{
    // remove the dev_executor from dev_scheduler_manager
    DevSchedulerManager::OnDevExecutorUnregistered(dev_executor);

    // remove the dev_executor from dev_allocator_manager

    DevAllocatorManager::OnDevExecutorUnregistered(dev_executor);

    // remove the dev_executor from executor_manager
    if(SafeRemoveOnly(dev_executor->GetDevID()))
        return true;

    return false;
}

bool DevExecutorManager::GetDevExecutorByID(const dev_id_t& dev_id, DevExecutor*& dev_executor)
{
    return SafeGet(dev_id, dev_executor);
}

bool DevExecutorManager::GetDefaultDevExecutor(DevExecutor*& dev_executor)
{
    std::string default_dev_id;

    if(!DriverManager::GetDefaultDeviceName(default_dev_id))
        return false;

    return GetDevExecutorByID(default_dev_id, dev_executor);
}

bool DevExecutorManager::GetDevExecutorByName(const std::string& dev_name, DevExecutor*& dev_executor)
{
    auto manager = GetInstance();
    bool found = false;

    manager->Lock();
    auto ir = (*manager).begin();
    auto end = (*manager).end();

    while(ir != end)
    {
        auto dev = ir->second;

        if(dev->GetName() == dev_name)
        {
            found = true;
            dev_executor = dev;
            break;
        }

        ir++;
    }

    manager->Unlock();

    return found;
}

int DevExecutorManager::GetDevExecutorNum(void)
{
    auto manager = GetInstance();
    manager->Lock();

    int number = manager->size();

    manager->Unlock();

    return number;
}

}    // namespace TEngine
