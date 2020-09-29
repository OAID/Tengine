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

#ifndef __NN_DEVICE_H__
#define __NN_DEVICE_H__

struct subgraph;

struct nn_device
{
    char* name;

    int (*init)(struct nn_device* dev);
    int (*prerun)(struct nn_device* dev, struct subgraph* subgraph, int num_thread, int cpu_affinity, int mode);
    int (*run)(struct nn_device* dev, struct subgraph* subgraph);
    int (*postrun)(struct nn_device* dev, struct subgraph* subgraph);
    int (*async_run)(struct nn_device* dev, struct subgraph* subgraph);
    int (*async_wait)(struct nn_device* dev, struct subgraph* subgraph, int try_wait);
    int (*release)(struct nn_device* dev);
    int (*release_exec_graph)(struct nn_device* dev, void* exec_graph);
};

extern struct nn_device* get_nn_device_by_name(const char* name);
extern struct nn_device* get_nn_device(int idx);
extern int get_nn_device_number(void);

extern void release_nn_dev_exec_graph(struct nn_device* dev, void* exec_graph);
extern int init_nn_dev_registry(void);
extern void release_nn_dev_registry(void);

extern int register_nn_device(struct nn_device* dev);

#define REGISTER_NN_DEVICE(dev) \
    REGISTER_MODULE_INIT_ARG(MOD_DEVICE_LEVEL, "register_nn_device", ( module_init_func_t )register_nn_device, dev)

#endif
