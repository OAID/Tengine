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
 * Copyright (c) 2021, Open AI Lab
 * Author: lswang@openailab.com
 */

#include "trt_device.hpp"

EXPORT_BEGIN
#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "executer/executer.h"
#include "optimizer/split.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/vector.h"
#include "utility/log.h"

#include <string.h>
EXPORT_FINISH

#include "trt_limit.hpp"
#include "trt_graph.hpp"

#include <cuda_runtime_api.h>


int trt_describe(struct device* device, struct vector* allowed_ops, struct vector* blocked_ops, struct vector* precision)
{
    (void)device;

    for (int op_type : trt_supported_ops)
    {
        push_vector_data(allowed_ops, &op_type);
    }

    for (int i = 0; i < OP_BUILTIN_LAST; i++)
    {
        bool in_list = false;

        for (const auto& type : trt_supported_ops)
        {
            if (type == i)
            {
                in_list = true;
                break;
            }
        }

        if (!in_list)
        {
            push_vector_data(blocked_ops, &i);
        }
    }

    int precision_var = TENGINE_DT_INT8;
    push_vector_data(precision, &precision_var);
    precision_var = TENGINE_DT_UINT8;
    push_vector_data(precision, &precision_var);
    precision_var = TENGINE_DT_FP16;
    push_vector_data(precision, &precision_var);
    precision_var = TENGINE_DT_FP32;
    push_vector_data(precision, &precision_var);

    return 0;
}


int trt_evaluation(struct device* device, struct subgraph* sub_graph, struct vector* evolution_tensors, struct vector* evolution_nodes)
{
    // nothing to do with tensorrt
    (void)device;
    (void)sub_graph;
    (void)evolution_tensors;
    (void)evolution_nodes;

    return 0;
}


int trt_allocate(struct device* device, struct subgraph* sub_graph)
{
    if (nullptr == device)
    {
        return -1;
    }

    /* set the correct input wait count: INPUT tensor is always ready */
    sub_graph->input_wait_count = 0;

    for (int i = 0; i < sub_graph->input_num; i++)
    {
        struct tensor* tensor = get_ir_graph_tensor(sub_graph->graph, sub_graph->input_tensor_list[i]);

        if (tensor->tensor_type == TENSOR_TYPE_VAR)
            sub_graph->input_wait_count++;
    }

    return 0;
}


int trt_release(struct device* device, struct subgraph* sub_graph)
{
    (void)sub_graph;

    if (nullptr == device)
    {
        return -1;
    }

    return 0;
}


int trt_split_graph(struct graph* ir_graph)
{
    struct device* cur_dev = ir_graph->attribute->context->device;

    if (0 != strcmp(TRT_DEVICE_NAME, cur_dev->name))
    {
        return -1;
    }

    struct vector* allowed_ops = create_vector(sizeof(int), NULL);
    struct vector* blocked_ops = create_vector(sizeof(int), NULL);
    struct vector* precision = create_vector(sizeof(int), NULL);

    cur_dev->allocator->describe(cur_dev, allowed_ops, blocked_ops, precision);

    split_graph_node_to_sub_graph(ir_graph, allowed_ops, blocked_ops, precision);

    release_vector(allowed_ops);
    release_vector(blocked_ops);
    release_vector(precision);

    //
    generate_sub_graph_io(ir_graph);
    add_sub_graph_to_ir_graph(ir_graph);

    // add node sub graph id
    for (int i = 0; i < (uint16_t)get_vector_num(ir_graph->subgraph_list); i++)
    {
        struct subgraph* sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);
        sub_graph->index = i;

        for (uint16_t j = 0; j < sub_graph->node_num; j++)
        {
            uint16_t node_id = sub_graph->node_list[j];
            struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
            ir_node->subgraph_idx = sub_graph->index;
        }
    }

    return 0;
}


EXPORT_BEGIN
static struct interface trt_interface = {
        .init           = trt_dev_init,
        .pre_run        = trt_dev_prerun,
        .run            = trt_dev_run,
        .post_run       = trt_dev_postrun,
        .async_run      = nullptr,
        .async_wait     = nullptr,
        .release_graph  = nullptr,
        .release_device = trt_dev_release,
};


static struct allocator trt_allocator = {
        .describe       = trt_describe,
        .evaluation     = trt_evaluation,
        .allocate       = trt_allocate,
        .release        = trt_release,
};


static struct optimizer trt_optimizer = {
        .split_graph    = trt_split_graph,
        .optimize_graph = nullptr,
};


static struct trt_device nv_trt_dev = {
        .base = {
                .name       = TRT_DEVICE_NAME,
                .interface  = &trt_interface,
                .allocator  = &trt_allocator,
                .optimizer  = &trt_optimizer,
                .scheduler  = nullptr,
                .privacy    = nullptr,
        },
};


int register_trt_device(void)
{
    int deviceCount;
    cudaError_t cudaError;
    cudaError = cudaGetDeviceCount(&deviceCount);

    if (cudaSuccess == cudaError && 0 < deviceCount)
    {
        int ret = register_device(&nv_trt_dev.base);
        if (0 != ret)
        {
            TLOG_INFO("Tengine plugin %s register failed.\n", nv_trt_dev.base.name);
            return ret;
        }

        TLOG_INFO("Tengine plugin device %s is registered.\n", nv_trt_dev.base.name);

        cudaDeviceProp prop;
        for (int i = 0; i < deviceCount; i++)
        {
            cudaError = cudaGetDeviceProperties(&prop, i);

            TLOG_DEBUG("  Device ID: %d; Name: %s\n", i, prop.name);
            TLOG_DEBUG("    SM Count:     %d\n", prop.multiProcessorCount);
            TLOG_DEBUG("    SM Clock:     %.2f GHz\n", float(prop.clockRate / 1e6));
            TLOG_DEBUG("    Memory Clock: %.2f GHz\n", float(prop.memoryClockRate / 1e6));
            TLOG_DEBUG("    Block Thread: %d\n", prop.maxThreadsPerBlock);
            TLOG_DEBUG("    Total Memory: %d GiByte\n", int(prop.totalGlobalMem / 1024 / 1024 / 1024));
            TLOG_DEBUG("    Block Memory: %d KiByte\n", int(prop.sharedMemPerBlock / 1024));
            TLOG_DEBUG("    Compute Capability: %d.%d\n", prop.major, prop.minor);
        }

        return 0;
    }

    TLOG_INFO("Tengine plugin %s: No GPU device found.\n", nv_trt_dev.base.name);
    return -1;
}


int unregister_trt_device(void)
{
    int deviceCount;
    cudaError_t cudaError;
    cudaError = cudaGetDeviceCount(&deviceCount);

    if (cudaSuccess == cudaError && 0 < deviceCount)
    {
        int ret = unregister_device(&nv_trt_dev.base);
        if (0 != ret)
        {
            TLOG_INFO("Tengine plugin %s unregister failed.\n", nv_trt_dev.base.name);
            return ret;
        }

        TLOG_INFO("Tengine plugin device %s is unregistered.\n", nv_trt_dev.base.name);

        return 0;
    }

    TLOG_INFO("Tengine plugin %s: No GPU device found.\n", nv_trt_dev.base.name);
    return -1;
}

EXPORT_FINISH
