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

#include <stdio.h>

#include "sys_port.h"
#include "tengine_c_api.h"
#include "tengine_ir.h"
#include "tengine_exec.h"

void init_exec_attr(struct exec_attr* attr, struct exec_context* context)
{
    attr->priv_context = 0;
    attr->exec_status = GRAPH_STAT_CREATED;
    attr->priority = 0;
    attr->policy = DEFAULT_POLICY;
    attr->fc_mt = 0;
    attr->pool_mt = 0;
    attr->exec_context = context;
}

void destroy_exec_attr(struct ir_graph* g, struct exec_attr* attr)
{
    sys_free(attr);
}

int release_dev_mem(struct nn_device* dev, struct dev_mem* dev_mem)
{
    // TODO:
    return -1;
}
