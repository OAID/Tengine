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
 * Author: haitao@openailab.com
 */

#ifndef __CPU_DEVICE_H__
#define __CPU_DEVICE_H__

#include "nn_device.h"

#define MEM_POOL_ALLOCATED 8

struct node_ops;
struct ir_node;

struct cpu_device
{
    struct nn_device base;
    uint8_t master_cpu;
    uint8_t cpu_model;
};

struct exec_node
{
    struct ir_node* ir_node;
    struct node_ops* node_ops;
    void* ops_priv; /* priv data for ops */

    int8_t inplace_map_num;
    int8_t output_num;

    union
    {
        uint8_t* inplace_map_ptr;
        uint8_t inplace_map[4]; /* opt for single inplace map, such as relu */
    };

    union
    {
        int8_t block_id[4];
        int8_t* block_id_ptr;
    };

    int shared_mem_size;
    int shared_pack4_mem_size;
};

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
    int cpu_affinity;
};

#define GET_MEM_PTR_HEADER(ptr) ( struct mem_ptr_header* )(( char* )ptr - 4);

int register_cpu_device(void);

#endif
