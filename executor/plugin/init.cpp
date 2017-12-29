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
#include "simple_executor.hpp"
#include "graph_optimizer.hpp"


using namespace TEngine;
 
namespace TEngine {

#ifdef CONFIG_EVENT_EXECUTOR
    extern void tengine_init_event_executor(void);
#endif

#ifdef CONFIG_CAFFE_REF
    extern void RegisterCaffeExecutors(void);
    extern void RegisterFusedDemoNodeExec(void);
#endif

#ifdef CONFIG_ARCH_ARM64
    extern void RegisterConvolutionNodeExec(void);
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
#endif


};

extern "C" {
int tengine_plugin_init(void);
}

int tengine_plugin_init(void)
{
#ifdef CONFIG_CAFFE_REF
    std::cout<<"EXECUTOR PLUGIN INITED\n";
    RegisterCaffeExecutors();
    RegisterFusedDemoNodeExec();
#endif

#ifdef CONFIG_ARCH_ARM64
#ifdef CONFIG_CAFFE_REF
    std::cout<<"Replace Caffe's operators ...\n";
#endif
    RegisterConvolutionNodeExec();
    RegisterFullyConnectedNodeExec();
    RegisterReLuNodeExec();
    RegisterDropoutNodeExec();
    RegisterSoftmaxNodeExec();
    RegisterConcatNodeExec();
    RegisterLRNNodeExec();
    RegisterScaleNodeExec();
    RegisterBatchNormNodeExec();
    RegisterPoolingNodeExec();
    RegisterFusedBNScaleReluNodeExec();

#endif
    ExecEngine * p_engine=new SimpleExec();
    ExecEngineManager::SafeAdd("default",ExecEnginePtr(p_engine));

#ifdef CONFIG_EVENT_EXECUTOR
    tengine_init_event_executor();
#endif

    GraphOptimizerManager::Init();

    return 0;
}

