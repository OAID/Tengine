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
#include <functional>
#include <iostream>
#include "device_driver.hpp"
#include "generic_engine.hpp"
#include "graph_optimizer.hpp"
#include "logger.hpp"
#include "simple_engine.hpp"
#include "tengine_plugin.hpp"

using namespace TEngine;

namespace TEngine {

extern void NodeOpsRegistryManagerInit(void);

extern void RegisterConcatNodeExec(void);
extern void RegisterDropoutNodeExec(void);
extern void RegisterSoftmaxNodeExec(void);
extern void RegisterLRN_NodeExec(void);
extern void RegisterEltwiseNodeExec(void);
extern void RegisterSliceNodeExec(void);
extern void RegisterPReLUNodeExec(void);
extern void RegisterNormalizeNodeExec(void);
extern void RegisterPermuteNodeExec(void);
extern void RegisterFlattenNodeExec(void);
extern void RegisterPriorBoxNodeExec(void);
extern void RegisterReshapeNodeExec(void);
extern void RegisterDetectionOutputNodeExec(void);
extern void RegisterRegionNodeExec(void);
extern void RegisterReorgNodeExec(void);
extern void RegisterRPNNodeExec(void);
extern void RegisterROIPoolingNodeExec(void);
extern void RegisterReLu6NodeExec(void);
extern void RegisterReLuNodeExec(void);
extern void RegisterResizeNodeExec(void);
#ifdef CONFIG_ARCH_BLAS
extern void RegisterConvBlasNodeExec(void);
extern void RegisterDeconvBlasNodeExec(void);
extern void RegisterFcBlasNodeExec(void);
#endif
extern void RegisterPooling_NodeExec(void);
extern void RegisterBatchNorm_NodeExec(void);
extern void RegisterScale_NodeExec(void);

extern void RegisterCommonFusedBNScaleReluNodeExec(void);
#ifdef CONFIG_CAFFE_REF
extern void RegisterCaffeExecutors(void);
#endif
extern void RegisterDemoOps(void);

#if CONFIG_ARCH_ARM64 == 1 || CONFIG_ARCH_ARM32 == 1
extern void RegisterConv2dFast(void);
extern void RegisterConv2dDepth(void);
extern void RegisterFullyConnectedFast(void);
extern void RegisterPoolingNodeExec(void);
extern void RegisterBatchNormNodeExec(void);
extern void RegisterScaleNodeExec(void);
extern void RegisterDeconvNodeExec(void);
extern void RegisterLRNNodeExec(void);

#endif

#ifdef CONFIG_ARCH_ARM64
extern void RegisterFusedBNScaleReluNodeExec(void);

#endif

#ifdef CONFIG_ACL_GPU
extern void RegisterConv2dOpencl(void);
#endif

void DevAllocatorManagerInit(void);
void DevSchedulerManagerInit(void);

};  // namespace TEngine

extern "C" {
int executor_plugin_init(void);
}

int executor_plugin_init(void) {
  NodeOpsRegistryManagerInit();

#ifndef ANDROID
  RegisterDemoOps();
#endif

  RegisterConcatNodeExec();
  RegisterDropoutNodeExec();
  RegisterSoftmaxNodeExec();
  RegisterLRN_NodeExec();
  // RegisterLRNNodeExec();
  RegisterPReLUNodeExec();
  RegisterEltwiseNodeExec();
  RegisterSliceNodeExec();
  RegisterNormalizeNodeExec();
  RegisterPermuteNodeExec();
  RegisterFlattenNodeExec();
  RegisterPriorBoxNodeExec();
  RegisterReshapeNodeExec();
  RegisterDetectionOutputNodeExec();
  RegisterRegionNodeExec();
  RegisterReorgNodeExec();
  RegisterRPNNodeExec();
  RegisterROIPoolingNodeExec();
  RegisterReLu6NodeExec();
  RegisterReLuNodeExec();
  RegisterResizeNodeExec();
#ifdef CONFIG_ARCH_BLAS
  RegisterConvBlasNodeExec();
  RegisterDeconvBlasNodeExec();
  RegisterFcBlasNodeExec();
#endif
  RegisterPooling_NodeExec();
  RegisterBatchNorm_NodeExec();
  RegisterScale_NodeExec();

  RegisterCommonFusedBNScaleReluNodeExec();

#ifdef CONFIG_CAFFE_REF
  std::cout << "Register reference's operators\n";
  RegisterCaffeExecutors();
#endif

#if CONFIG_ARCH_ARM64 || CONFIG_ARCH_ARM32
  RegisterConv2dFast();
  RegisterConv2dDepth();
  RegisterFullyConnectedFast();
  RegisterPoolingNodeExec();
  RegisterBatchNormNodeExec();
  RegisterScaleNodeExec();
  RegisterLRNNodeExec();
#endif

#ifdef CONFIG_ARCH_ARM64
  RegisterFusedBNScaleReluNodeExec();

#ifdef CONFIG_ACL_GPU
  // RegisterConv2dOpencl();
#endif

#endif

  DevAllocatorManagerInit();
  DevSchedulerManagerInit();

  GenericEngine* p_generic = new GenericEngine();
  p_generic->SetScheduler("Simple");

  ExecEngineManager::SafeAdd(p_generic->GetName(), ExecEnginePtr(p_generic));

  SimpleEngine* p_simple = new SimpleEngine();
  ExecEngineManager::SafeAdd(p_simple->GetName(), ExecEnginePtr(p_simple));

  GraphOptimizerManager::Init();

  TEnginePlugin::RegisterModuleInit(1, DriverManager::ProbeDevice);
  TEnginePlugin::RegisterModuleRelease(1, DriverManager::ReleaseDevice);

  return 0;
}
