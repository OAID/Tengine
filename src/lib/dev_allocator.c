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
 * Author: lswang@openailab.com
 */

#include <stdio.h>
#include <string.h>

#include "sys_port.h"
#include "vector.h"
#include "tengine_ir.h"
#include "tengine_exec.h"

#include "dev_allocator.h"

static int allocator_vector_created = 0;

static struct vector* allocator_vector;


int init_allocator_registry(struct dev_allocator* allocator)
{
    if (!allocator_vector_created)
    {
        allocator_vector = create_vector(sizeof(struct dev_allocator), NULL);
        allocator_vector_created = 1;
    }

    if (allocator_vector == NULL)
        return -1;

    push_vector_data(allocator_vector, allocator);

    return 0;
}

int release_allocator_registry()
{
    release_vector(allocator_vector);
    return 0;
}

struct dev_allocator* get_default_dev_allocator(void)
{
    for (int i = 0; i < get_vector_num(allocator_vector); i++)
    {
        struct dev_allocator* allocator = get_vector_data(allocator_vector, i);
        if (!strcmp("cpu_dev", allocator->name))
        {
            return allocator;
        }
    }

    return NULL;
}

struct dev_allocator* get_dev_allocator(const char* dev_name)
{
    for (int i = 0; i < get_vector_num(allocator_vector); i++)
    {
        struct dev_allocator* allocator = get_vector_data(allocator_vector, i);
        if (!strcmp(dev_name, allocator->name))
        {
            return allocator;
        }
    }

    return NULL;
}
