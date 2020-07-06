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

#ifndef __TENGINE_EXEC_H__
#define __TENGINE_EXEC_H__

#include <stdint.h>

#include "vector.h"
#include "nn_device.h"
#include "dev_allocator.h"
#include "exec_scheduler.h"

#define EXEC_KERNEL_FP32 0
#define EXEC_KERNEL_FP16 1
#define EXEC_KERNEL_INT8 2
#define EXEC_KERNEL_UINT8 3

#define MODEL_FORMAT_UNKNOWN 0
#define MODEL_FORMAT_TENGINE 1
#define MODEL_FORMAT_CAFFE 2
#define MODEL_FORMAT_ONNX 3
#define MODEL_FORMAT_MXNET 4
#define MODEL_FORMAT_TENSORFLOW 5
#define MODEL_FORMAT_TFLITE 6
#define MODEL_FORMAT_DLA 7

struct ir_graph;

/* define the memory block used in device */
struct dev_mem
{
    uint32_t dev_mem_size;
    uint8_t dev_type;
    uint8_t cpu_read_ready;
    uint8_t cpu_write_done;

    void* mapped_mem;
    void* dev_priv; /* opaque pointer for device to interpret the dev_mem_addr */
    uint64_t dev_mem_addr; /* why not pointer? as in 32bit CPU, the dev address may be 64bit */
};

struct exec_context
{
    struct exec_scheduler* scheduler;
    struct dev_allocator* dev_allocator;
    struct nn_device* def_dev;
    char* name;
    struct vector* dev_list;
};

struct exec_attr
{
    uint8_t exec_status;
    uint8_t priority;
    uint8_t policy;
    uint8_t fc_mt;
    uint8_t pool_mt;
    uint8_t priv_context;
    struct exec_context* exec_context;
    void* sched_priv;
    void* allocator_priv;
};

void init_exec_attr(struct exec_attr* attr, struct exec_context* context);
void destroy_exec_attr(struct ir_graph* g, struct exec_attr* attr);

int release_dev_mem(struct nn_device* dev, struct dev_mem* dev_mem);

struct exec_scheduler* get_default_scheduler(void);
struct dev_allocator* get_default_dev_allocator(void);
struct nn_device* get_default_nn_device(void);

#endif
