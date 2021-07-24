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
 * Author: haitao@openailab.com
 * Revised: lswang@openailab.com
 */

#pragma once

#include "cpu_device.h"

#include <stddef.h>

struct exec_graph
{
    struct vector* exec_node_list;
    struct mem_pool* mem_pool;
    struct cpu_device* dev;

    void* shared_mem;
    int shared_mem_size;
    void* shared_pack4_mem;
    int shared_pack4_mem_size;
    int num_thread;
    int mode;
    size_t cpu_affinity;
    void* timer;
};

struct exec_graph* create_exec_graph(struct subgraph* subgraph, int num_thread, int mode, size_t cpu_affinity);

int prerun_exec_graph(struct exec_graph* exec_graph);

void release_exec_graph(void* exec_graph);
