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
 * Author: ddzhao@openailab.com
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include "sys_port.h"
extern "C" {
    #include "tengine_errno.h"
    #include "tengine_utils.h"
    #include "tengine_ir.h"
    #include "tengine_log.h"
    #include "tengine_op.h"
    #include "tengine_log.h"
}

#include "vk_device.hpp"
#include "vk_allocator.hpp"
#include "vulkan_gpu.hpp"
#include "vulkan_graph.hpp"

using namespace TEngine;

static int init_device(struct nn_device *dev)
{
    return 0;
}

static int release_device(struct nn_device *dev)
{
    return 0;
}

static int prerun(struct nn_device *dev, struct subgraph *subgraph, int num_threads, int cpu_affinity, int mode)
{
    TLOG_INFO("==== vulkan prerun start ====\n");
    create_gpu_instance();
    struct vk_device *vk_dev = (struct vk_device *)dev; 
    struct ir_graph *orig_graph = subgraph->graph;
    struct vk_dev_priv *priv = (struct vk_dev_priv *)orig_graph->dev_priv;

    /* todo: set the tensor shape ? */

    /* create exec_graph */
    VulkanGraph* vk_exec_graph = new VulkanGraph(subgraph);

    if (vk_exec_graph == nullptr)
    {
        set_tengine_errno(EIO);
        TLOG_ERR("vulkan exec graph is NULL\n");
        return -1;        
    }

    vk_exec_graph->upload_model();
    vk_exec_graph->create_pipeline();

    subgraph->exec_graph = vk_exec_graph;

    int node_num = subgraph->node_num;
    TLOG_INFO("num_threads:%d, node num:%d\n", num_threads, node_num);

    TLOG_INFO("==== vulkan prerun done ====\n");
    return 0;
}

static int run(struct nn_device *dev, struct subgraph *subgraph)
{
    struct vk_device *vk_dev = (struct vk_device *)dev; 
    struct ir_graph *orig_graph = subgraph->graph;
    struct vk_dev_priv *priv = (struct vk_dev_priv *)orig_graph->dev_priv;

    VulkanGraph *vk_exec_graph = (VulkanGraph *)subgraph->exec_graph;
    if (vk_exec_graph == nullptr)
    {
        set_tengine_errno(EIO);
        TLOG_ERR("vulkan exec graph is NULL\n");
        return -1;
    }

    vk_exec_graph->record_graph_pipeline();
    return 0;
}

static int postrun(struct nn_device *dev, struct subgraph *subgraph)
{
    destroy_gpu_instance();
    TLOG_INFO("done vk postrun!!!\n");
    return 0;
}

static struct vk_device vulkan_dev = {
    .base = {.name = "VK",
             .init = init_device,
             .prerun = prerun,
             .run = run,
             .postrun = postrun,
             .async_run = NULL,
             .async_wait = NULL,
             .release = release_device,
             .release_exec_graph = NULL,
             },
    .load_graph = NULL,
    .load_ir_graph = NULL,
    .unload_graph = NULL,
};

REGISTER_NN_DEVICE(&vulkan_dev.base);
