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
 * Author: haitao@openailab.com
 * Revised: lswang@openailab.com
 */

#include "executer/executer.h"

#include "utility/sys_port.h"
#include "api/c_api.h"

#include <string.h>

void init_attribute(ir_attribute_t* attribute, ir_context_t* context)
{
    attribute->status = GRAPH_STAT_CREATED;
    attribute->priority = 0;
    attribute->policy = DEFAULT_POLICY;
    attribute->private_context = 0;
    attribute->context = context;
    attribute->device_privacy = NULL;
    attribute->scheduler_privacy = NULL;
}

void destroy_attribute(struct graph* graph, ir_attribute_t* attribute)
{
    if (NULL != attribute->device_privacy)
    {
        sys_free(attribute->device_privacy);
    }

    if (NULL != attribute->scheduler_privacy)
    {
        sys_free(attribute->scheduler_privacy);
    }

    sys_free(attribute);
}

int release_device_mem(struct device* dev, ir_memory_t* dev_mem)
{
    // TODO:
    return -1;
}

void init_ir_context(ir_context_t* context, const char* name)
{
    if (NULL != name)
    {
        size_t length = strlen(name);
        context->name = (char*)sys_malloc(length);
        memcpy(context->name, name, length);
    }
    else
    {
        context->name = NULL;
    }

    context->scheduler = NULL;
    context->device = NULL;
    context->default_options = NULL;
    context->device_options = NULL;
}
