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
 * Author: lswang@openailab.com
 */

#pragma once

struct tensor;
struct graph;
struct subgraph;
struct options;
struct device;
struct vector;

#include <stddef.h>

/*!
 * @struct ir_interface_t
 * @brief  Abstract neural network runnable device interface struct
 */
typedef struct interface
{
    //!< interface of init this neural network device
    int (*init)(struct device* device);

    //!< interface of prepare runnable subgraph on device
    int (*pre_run)(struct device* device, struct subgraph* subgraph, void* options);

    //!< interface of run runnable subgraph on device
    int (*run)(struct device* device, struct subgraph* subgraph);

    //!< interface of post run runnable subgraph on device
    int (*post_run)(struct device* device, struct subgraph* subgraph);

    //!< interface of async run runnable subgraph on device
    int (*async_run)(struct device* device, struct subgraph* subgraph);

    //!< interface of async wait runnable subgraph on device
    int (*async_wait)(struct device* device, struct subgraph* subgraph, int try_wait);

    //!< interface of release runnable subgraph on device
    int (*release_graph)(struct device* device, void* device_graph);

    //!< interface of release this neural network device
    int (*release_device)(struct device* device);
} ir_interface_t;

/*!
 * @struct ir_allocator_t
 * @brief  Abstract neural network runnable device allocator struct
 */
typedef struct allocator
{
    //!< interface of describe this neural network device supported operators and precision
    int (*describe)(struct device*, struct vector* allowed_ops, struct vector* blocked_ops, struct vector* precision);

    //!< interface of evaluation subgraph, and report tensors which need to be changed
    int (*evaluation)(struct device*, struct subgraph*, struct vector* tensors, struct vector* nodes);

    //!< interface of allocate resources which subgraph needed
    int (*allocate)(struct device*, struct subgraph*);

    //!< interface of release allocated resources
    int (*release)(struct device*, struct subgraph*);
} ir_allocator_t;

/*!
 * @struct ir_optimizer_t
 * @brief  Abstract neural network runnable device expend optimizer
 */
typedef struct optimizer
{
    int (*split_graph)(struct graph* ir_graph);                   //!< interface of split graph delegation
    int (*optimize_graph)(struct graph* ir_graph, int precision); //!< interface of optimizing graph delegation
} ir_optimizer_t;

/*!
 * @struct nn_device_t
 * @brief  Abstract neural network runnable device description struct
 */
typedef struct device
{
    const char* name;
    struct interface* interface; //!< device scheduler operation interface
    struct allocator* allocator; //!< device allocation operation interface
    struct optimizer* optimizer; //!< device optimizer operation interface
    struct scheduler* scheduler; //!< device scheduler
    void* privacy;               //!< device privacy data
} ir_device_t;

/*!
 * @brief  Initialize a device.
 *
 *         This function will fill other pointer of ir_device_t to NULL.
 *
 * @param [in]  device: The specific device.
 * @param [in]  name: The name of the device.
 */
void init_ir_device(ir_device_t* device, const char* name);

/*!
 * @brief  Size of a device option struct.
 *
 *         This function will return device specific option length.
 *
 * @param [in]  device: The specific device.
 *
 *  @return size of device option struct.
 */
int get_device_option_size(ir_device_t* device);
