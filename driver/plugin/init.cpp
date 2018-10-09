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
#include <functional>
#include <iostream>

#include "logger.hpp"

extern "C" {
int driver_plugin_init(void);
}

namespace TEngine {

#ifdef CONFIG_ACL_GPU
extern void ACLDriverInit(void);
extern void ACLGraphInit(void);
#endif

#ifdef CONFIG_CAFFE_REF
extern void CaffeDriverInit(void);
#endif

extern void CPUDriverInit(void);
}

using namespace TEngine;

int driver_plugin_init(void) {
#ifdef CONFIG_ACL_GPU
  ACLDriverInit();
  ACLGraphInit();
#endif

#ifdef CONFIG_CAFFE_REF
  CaffeDriverInit();
#endif

  CPUDriverInit();

  return 0;
}
