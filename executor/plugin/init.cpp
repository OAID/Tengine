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
#include <iostream>
#include <functional>
#include "logger.hpp"
#include "generic_engine.hpp"
#include "graph_optimizer.hpp"
#include "tengine_plugin.hpp"
#include "device_driver.hpp"

using namespace TEngine;

namespace TEngine {

extern void NodeOpsRegistryManagerInit(void);

void DevAllocatorManagerInit(void);
void DevSchedulerManagerInit(void);
};    // namespace TEngine

int executor_plugin_init(void)
{
    NodeOpsRegistryManagerInit();

    DevAllocatorManagerInit();
    DevSchedulerManagerInit();

    GenericEngine* p_generic = new GenericEngine();
    p_generic->SetScheduler("Simple");

    ExecEngineManager::SafeAdd(p_generic->GetName(), ExecEnginePtr(p_generic));

    GraphOptimizerManager::Init();

    TEnginePlugin::RegisterModuleInit(1, DriverManager::ProbeDevice);
    TEnginePlugin::RegisterModuleRelease(1, DriverManager::ReleaseDevice);

    return 0;
}
