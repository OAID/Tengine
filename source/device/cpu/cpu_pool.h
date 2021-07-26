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

#include "cpu_define.h"

#include <stdint.h>

struct exec_graph;

struct mem_block_entry
{
    void* addr;
    int block_size;
    int max_req_size;
    int alloc_count;
    int free_count;
};

struct mem_pool
{
    uint8_t align_size; /* must be 2^n */
    struct vector* block_list;

    int (*get_backend_mem)(struct mem_pool*);
    void* (*get_mem_block)(struct mem_pool*, int block_id);
    int (*allocate)(struct mem_pool*, int size);
    void (*free)(struct mem_pool*, int block_id);
    void (*dump)(struct mem_pool*);
};

void release_mem_pool(struct mem_pool* mem_pool);
int alloc_exec_graph_mem(struct exec_graph* exec_graph);
void free_exec_graph_mem(struct exec_graph* graph);
