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
#include <stdlib.h>

#include "tengine_errno.h"
#include "vector.h"
#include "tengine_serializer.h"

static struct vector* serializer_list = NULL;

struct serializer* find_serializer(const char* name)
{
    char* real_name = strdup(name);

    char* p = strrchr(real_name, ':');

    if (p != NULL)
        *p = 0x0;

    int n = get_vector_num(serializer_list);

    for (int i = 0; i < n; i++)
    {
        struct serializer* s = *( struct serializer** )get_vector_data(serializer_list, i);

        if (!strcmp(s->get_name(s), real_name))
        {
            sys_free(real_name);
            return s;
        }
    }

    sys_free(real_name);

    return NULL;
}

int register_serializer(struct serializer* serializer)
{
    if (find_serializer(serializer->get_name(serializer)) != NULL)
    {
        set_tengine_errno(EEXIST);
        return -1;
    }

    if (serializer->init && serializer->init(serializer) < 0)
        return -1;

    push_vector_data(serializer_list, &serializer);

    return 0;
}

int unregister_serializer(struct serializer* serializer)
{
    if (find_serializer(serializer->get_name(serializer)) == NULL)
    {
        set_tengine_errno(ENOENT);
        return -1;
    }

    serializer->release(serializer);

    return remove_vector_data(serializer_list, &serializer);
}

int init_serializer_registry(void)
{
    serializer_list = create_vector(sizeof(struct serializer*), NULL);

    if (serializer_list == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    return 0;
}

int release_serializer_registry(void)
{
    int n = get_vector_num(serializer_list);

    for (int i = 0; i < n; i++)
    {
        struct serializer* s = *( struct serializer** )get_vector_data(serializer_list, i);

        if (s->release != NULL)
            s->release(s);
    }

    release_vector(serializer_list);

    return 0;
}
