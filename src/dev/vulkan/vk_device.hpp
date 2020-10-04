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
 * Copyright (c) 2020, Open AI Lab
 * Author: ddzhao@openailab.com
 */

#ifndef __VK_DEVICE_H__
#define __VK_DEVICE_H__

#include <stdint.h>

extern "C"
{
    #include "nn_device.h"
}

#ifndef MAX_SHAPE_DIM_NUM
#define MAX_SHAPE_DIM_NUM 4
#endif

struct ir_graph;

struct vx_dev_priv
{
    uint64_t graph;
};

struct tee_tensor_info
{
    char* name;
    int dims[MAX_SHAPE_DIM_NUM];
    int dim_num;
    int data_type;
    int tensor_type;
};

struct tee_graph_io
{
    int input_num;
    int output_num;
    struct tee_tensor_info* inputs;
    struct tee_tensor_info* outputs;
};

struct vk_device
{
    struct nn_device base;

    int (*load_graph)(struct vk_device* dev);

    int (*load_ir_graph)(struct vk_device* dev);

    int (*unload_graph)(struct vk_device* dev);
};

int register_vk_device(void);

#endif
