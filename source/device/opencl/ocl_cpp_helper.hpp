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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: hbshi@openailab.com
 */

#ifndef TENGINE_LITE_SOURCE_DEVICE_OPENCL_OCL_CPP_HELPER_HPP_
#define TENGINE_LITE_SOURCE_DEVICE_OPENCL_OCL_CPP_HELPER_HPP_

extern "C" {
#include "api/c_api.h"
#include "device/device.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "executer/executer.h"
#include "optimizer/split.h"
#include "module/module.h"
#include "utility/vector.h"
#include "utility/log.h"
}

#include <map>
#include <set>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <fstream>
#include <memory>
#include "cache/cache.hpp"

#define UP_DIV(x, y)   (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x)   ROUND_UP((x), 4)
#define ALIGN_UP8(x)   ROUND_UP((x), 8)

#define CL_TARGET_OPENCL_VERSION      200
#define CL_HPP_TARGET_OPENCL_VERSION  110
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "device/opencl/include/CL/cl2.hpp"
#pragma GCC diagnostic pop

#endif //TENGINE_LITE_SOURCE_DEVICE_OPENCL_OCL_CPP_HELPER_HPP_
