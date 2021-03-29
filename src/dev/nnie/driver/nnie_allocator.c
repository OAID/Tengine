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
 * Author: bsun@openailab.com
 */

#include "sys_port.h"
#include "vector.h"
#include "tengine_ir.h"
#include "tengine_exec.h"
#include "tengine_log.h"
#include "nnie_allocator.h"

static int nnie_allocate(struct dev_allocator* allocator, struct subgraph* subgraph)
{
    return 0;
}

static int nnie_describe(struct dev_allocator* allocator, struct vector* allowed_ops, struct vector* blocked_ops, struct vector* precision)
{
    return 0;
}


static struct dev_allocator nnie_allocator = {
    .name       = "nnie",
    .describe   = nnie_describe,
    .evaluation = NULL,
    .allocate   = nnie_allocate,
    .release    = NULL,
};

struct dev_allocator* get_nnie_allocator(void)
{
    return &nnie_allocator;
}
#ifndef STANDLONE_MODE
static void register_nnie_allocator(void)
#else
void register_nnie_allocator(void)
#endif
{
    TLOG_INFO("Tengine plugin allocator %s is registered.\n", nnie_allocator.name);
    init_allocator_registry(&nnie_allocator);
}

#ifndef STANDLONE_MODE
REGISTER_DEV_ALLOCATOR(register_nnie_allocator);
#endif
