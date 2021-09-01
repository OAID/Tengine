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
#pragma once

#include <dirent.h>

#include <string>
#include <vector>
#include <unordered_map>

extern "C" {
#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "graph/node.h"
#include "graph/tensor.h"
#include "utility/sys_port.h"
#include "utility/utils.h"

#include "convolution_param.h"
}

#include "quant_save_graph.hpp"
#include "quant_utils.hpp"

#define DEFAULT_IMG_H 227
#define DEFAULT_IMG_W 227
#define DEFAULT_SCALE1 1.f
#define DEFAULT_SCALE2 1.f
#define DEFAULT_SCALE3 1.f
#define DEFAULT_MEAN1 104.007
#define DEFAULT_MEAN2 116.669
#define DEFAULT_MEAN3 122.679
#define DEFAULT_LOOP_COUNT 1
#define DEFAULT_THREAD_COUNT 1

#define DO_DFQ 1

struct node_graph
{
    int pass;
    std::vector<uint16_t> input_node_list;
    std::vector<uint16_t> output_node_list;
};

int data_free_quant(const char* model_file, const char* image_dir,
                    int img_c, int img_h, int img_w, const float* mean, const float* scale,
                    int num_thread, int sw_RGB, int center_crop);
