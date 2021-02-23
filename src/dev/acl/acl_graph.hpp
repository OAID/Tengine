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

#ifndef __TRT_TRT_GRAPH_HPP__
#define __TRT_TRT_GRAPH_HPP__

extern "C"
{
#include "nn_device.h"
#include "tengine_ir.h"

int acl_dev_init(struct nn_device* dev);
int acl_dev_prerun(struct nn_device* dev, struct subgraph* subgraph, int num_thread, int cpu_affinity, int mode);
int acl_dev_run(struct nn_device* dev, struct subgraph* subgraph);
int acl_dev_postrun(struct nn_device* dev, struct subgraph* subgraph);
int acl_dev_release(struct nn_device* dev);

}

#endif
