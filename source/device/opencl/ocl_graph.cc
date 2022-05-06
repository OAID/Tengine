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
 * Author: hhchen@openailab.com
 */

#include "ocl_graph.hpp"
#include "ocl_executor.hpp"
#include "ocl_define.h"

extern "C" {
#include "graph/graph.h"
#include "graph/subgraph.h"
}

int ocl_dev_init(struct device* dev)
{
    (void)dev;
    auto engine = new OCLEngine;
    dev->privacy = engine;

    return 0;
}

static bool ocl_graph_index_first(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;
    int subgraph_num = get_vector_num(ir_graph->subgraph_list);

    for (int i = 0; i < subgraph_num; i++)
    {
        struct subgraph* _subgraph = get_ir_graph_subgraph(ir_graph, i);
        ir_device_t* device = _subgraph->device;
        char* ocl_name = "OCL";
        if (0 == strcmp(device->name, ocl_name))
        {
            return i == subgraph->index;
        }
    }

    return false;
}

static bool ocl_graph_index_last(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;
    int subgraph_num = get_vector_num(ir_graph->subgraph_list);

    int last_ocl_index = -1;
    for (int i = 0; i < subgraph_num; i++)
    {
        struct subgraph* _subgraph = get_ir_graph_subgraph(ir_graph, i);
        ir_device_t* device = _subgraph->device;
        char* ocl_name = "OCL";
        if (0 == strcmp(device->name, ocl_name))
        {
            last_ocl_index = i;
        }
    }

    return last_ocl_index == subgraph->index;
}

int ocl_dev_prerun(struct device* dev, struct subgraph* subgraph, void* options)
{
    auto engine = (OCLEngine*)dev->privacy;
    auto opt = (ocl_option*)options;
    std::string cache_path = opt->cache_path;
    if (ocl_graph_index_first(subgraph) && opt->load_cache)
    {
        engine->load_cache(cache_path);
    }
    auto ret = engine->OCLEnginePreRun(subgraph);
    if (ocl_graph_index_last(subgraph) && opt->store_cache)
    {
        engine->store_cache(cache_path);
    }
    return ret;
}

int ocl_dev_run(struct device* dev, struct subgraph* subgraph)
{
    auto engine = (OCLEngine*)dev->privacy;
    return engine->OCLEngineRun(subgraph);
}

int ocl_dev_postrun(struct device* dev, struct subgraph* subgraph)
{
    auto engine = (OCLEngine*)subgraph->device_graph;
    engine->OCLEnginePostRun();
    delete engine;
    return 0;
}

int ocl_dev_release(struct device* dev)
{
    auto engine = (OCLEngine*)dev->privacy;
    engine->OCLEnginePostRun();
    delete engine;

    return 0;
}
