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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: hhchen@openailab.com
 */

#include <cstdlib>
#include <cstdio>
#include <sys/stat.h>

#include <fstream>
#include <string>
#include <cmath>
#ifdef _MSC_VER
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif

#include "quant_utils.hpp"
#include "save_graph.hpp"

#include "api/c_api.h"

extern "C" {
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "graph/node.h"
#include "graph/tensor.h"
#include "utility/sys_port.h"
#include "utility/utils.h"
}

int save_graph_u8_perlayer(const char* model_file, const char* scale_file, const std::string& output_file, int inplace, bool internal);

int save_graph_i8_perchannel(const char* model_file, const char* scale_file, const std::string& output_file, int inplace, bool internal);

int save_graph_u8_perchannel(const char* model_file, const char* scale_file, const std::string& output_file, int inplace, bool internal);
