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

#include <stddef.h>
#include <stdint.h>

struct device;
struct graph;
struct scheduler;

/*!
 * @struct ir_context_t
 * @brief  Abstract neural network runnable execution context
 */
typedef struct context
{
    char* name;
    struct scheduler* scheduler; //!< binding scheduler of this context
    struct device* device;       //!< binding device of this context
    void* default_options;       //<! default device options of this context
    void* device_options;        //<! device options of this context
} ir_context_t;

/*!
 * @struct ir_memory_t
 * @brief  The memory block used in device
 */
typedef struct memory
{
    uint32_t dev_mem_size;
    uint8_t dev_type;
    uint8_t cpu_read_ready;
    uint8_t cpu_write_done;

    void* mapped_mem;
    void* privacy;     /* opaque pointer for device to interpret the dev_mem_addr */
    uintptr_t address; /* why not pointer? as in 32bit CPU, the dev address may be 64bit */
} ir_memory_t;

typedef struct attribute
{
    uint8_t status;
    uint8_t priority;
    uint8_t policy;
    uint8_t private_context;
    struct context* context;
    void* device_privacy;
    void* scheduler_privacy;
} ir_attribute_t;

/*!
 * @brief  Initialize a context.
 *
 * @param [in]  context: The specific context.
 * @param [in]  name: The name of the context.
 */
void init_ir_context(ir_context_t* context, const char* name);

/*!
 * @brief  Init graph attribute.
 *
 * @param [in]  attribute: The point to graph attribute.
 * @param [in]  context: The point to graph context.
 */
void init_attribute(ir_attribute_t* attribute, struct context* context);

/*!
 * @brief  Release graph attribute.
 *
 * @param [in]  graph: The point to a graph.
 * @param [in]  attribute: The point to graph attribute.
 */
void destroy_attribute(struct graph* graph, ir_attribute_t* attribute);

/*!
 * @brief  Release device memory.
 *
 * @param [in]  dev: The point to a device.
 * @param [in]  dev_mem: The point to a device memory struct.
 *
 * @return statue value, 0 success, other value failure.
 */
int release_device_mem(struct device* dev, ir_memory_t* dev_mem);
