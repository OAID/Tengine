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
 * Copyright (c) 2021, Institute of Computing Technology
 * Author: wanglei21c@mails.ucas.ac.cn
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tensor;
struct subgraph;

#define TENGINE_DUMP_DIR   "TG_DEBUG_DUMP_DIR"
#define TENGINE_DUMP_LAYER "TG_DEBUG_DATA"

void extract_feature_from_tensor_odla(const char* comment, const char* layer_name, const struct tensor* tensor);

void dump_sub_graph_odla(struct subgraph* sub_graph);

void odla_data_dump(const char* filename, int8_t* data, int w, int h, int c);