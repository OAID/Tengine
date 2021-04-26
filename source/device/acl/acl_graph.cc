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

#include "acl_graph.hpp"

#include "acl_define.h"
#include "acl_executor.hpp"

extern "C"
{
#include "device/device.h"
}

int acl_dev_init(struct device* dev)
{
    (void)dev;
    return 0;
}

int acl_dev_prerun(struct device* dev, struct subgraph* subgraph, void* options)
{
    subgraph->device_graph = new CLGraph;
    auto engine = (CLGraph*)subgraph->device_graph;

    struct acl_option* opt = (struct acl_option*)options;

    return engine->prerun(subgraph, opt);
}

int acl_dev_run(struct device* dev, struct subgraph* subgraph)
{
    auto engine = (CLGraph*)subgraph->device_graph;
    return engine->run(subgraph);
}

int acl_dev_postrun(struct device* dev, struct subgraph* subgraph)
{
    auto engine = (CLGraph*)subgraph->device_graph;
    engine->postrun(subgraph);
    delete engine;

    return 0;
}

int acl_dev_release(struct device* dev)
{
    (void)dev;
    return 0;
}
