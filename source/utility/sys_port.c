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
 */

#include "utility/sys_port.h"

#include <string.h>

#ifdef CONFIG_MEM_STAT

#include "mem_stat.h"

void* sys_malloc(size_t size)
{
    if (skip_stat())
        return malloc(size);
    else
        return stat_malloc(size);
}

void sys_free(void* ptr)
{
    if (skip_stat())
        return free(ptr);
    else
        return stat_free(ptr);
}

void* sys_realloc(void* ptr, size_t size)
{
    if (skip_stat())
        return realloc(ptr, size);
    else
        return stat_realloc(ptr, size);
}

#else

void* sys_malloc(size_t size)
{
    return malloc(size);
}

void sys_free(void* ptr)
{
    return free(ptr);
}

void* sys_realloc(void* ptr, size_t size)
{
    return realloc(ptr, size);
}

#endif

#ifdef CONFIG_ARCH_CORTEX_M

char* strdup(const char* src)
{
    if (src == NULL)
        return NULL;

    int n = strlen(src);

    char* new_str = (char*)sys_malloc(n + 1);

    if (new_str == NULL)
        return NULL;

    memcpy(new_str, src, n + 1);

    return new_str;
}

#endif
