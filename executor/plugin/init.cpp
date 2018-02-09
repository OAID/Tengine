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
#include "cpu_engine.hpp"
#include "generic_engine.hpp"
#include "simple_engine.hpp"
#include "graph_optimizer.hpp"
#include "tengine_plugin.hpp"
#include "device_driver.hpp"



using namespace TEngine;
 
namespace TEngine {

    extern void RegisterDefaultSoc(void);
    extern void NodeOpsRegistryManagerInit(void);

#ifdef CONFIG_CAFFE_REF
    extern void RegisterCaffeExecutors(void);
    extern void RegisterFusedDemoNodeExec(void);
#endif
    extern void RegisterDemoOps(void);

#ifdef CONFIG_ARCH_ARM64
    namespace conv_selector {
    extern bool RegisterConvSelector(const std::string& registry_name);
    }


    extern void RegisterConv2dDepth(void);
    extern void RegisterConv2dFast(void);
    extern void RegisterFullyConnectedNodeExec(void);
    extern void RegisterReLuNodeExec(void);
    extern void RegisterDropoutNodeExec(void);
    extern void RegisterSoftmaxNodeExec(void);
    extern void RegisterConcatNodeExec(void);
    extern void RegisterLRNNodeExec(void);
    extern void RegisterScaleNodeExec(void);
    extern void RegisterBatchNormNodeExec(void);
    extern void RegisterPoolingNodeExec(void);
    extern void RegisterFusedBNScaleReluNodeExec(void);
    extern void RegisterEltwiseNodeExec(void);
    extern void RegisterSliceNodeExec(void);
    extern void RegisterPReLUNodeExec(void);
    extern void RegisterNormalizeNodeExec(void);
#endif

    

    void DevAllocatorManagerInit(void);
    void DevSchedulerManagerInit(void);


};

extern "C" {
int tengine_plugin_init(void);
}

int tengine_plugin_init(void)
{
    RegisterDefaultSoc();
    NodeOpsRegistryManagerInit();

    RegisterDemoOps();

#ifdef CONFIG_CAFFE_REF
    std::cout<<"Register reference's operators\n";
    RegisterCaffeExecutors();
    RegisterFusedDemoNodeExec();
#endif

#ifdef CONFIG_ARCH_ARM64
    std::cout<<"Register  arm64's operators ...\n";

    conv_selector::RegisterConvSelector("arm64");
    RegisterFullyConnectedNodeExec();
    RegisterReLuNodeExec();
    RegisterDropoutNodeExec();
    RegisterSoftmaxNodeExec();
    RegisterConcatNodeExec();
    RegisterLRNNodeExec();
    RegisterScaleNodeExec();
    RegisterBatchNormNodeExec();
    RegisterPoolingNodeExec();
    RegisterConv2dFast();
    RegisterConv2dDepth();

    RegisterFusedBNScaleReluNodeExec();
    RegisterPReLUNodeExec();
    RegisterEltwiseNodeExec();
    RegisterSliceNodeExec();
    RegisterNormalizeNodeExec();

#endif


    DevAllocatorManagerInit();
    DevSchedulerManagerInit();
    
    ExecEngine * p_engine=new CPUEngine();
    ExecEngineManager::SafeAdd(p_engine->GetName(),ExecEnginePtr(p_engine));

    GenericEngine * p_generic=new GenericEngine();
    p_generic->SetScheduler("Simple");

    ExecEngineManager::SafeAdd(p_generic->GetName(),ExecEnginePtr(p_generic));

    SimpleEngine * p_simple=new SimpleEngine();
    ExecEngineManager::SafeAdd(p_simple->GetName(),ExecEnginePtr(p_simple));


    GraphOptimizerManager::Init();

    TEnginePlugin::RegisterModuleInit(1,DriverManager::ProbeDevice);
    TEnginePlugin::RegisterModuleRelease(1,DriverManager::ReleaseDevice);

    std::cout<<"EXECUTOR PLUGIN INITED\n";

    return 0;
}

