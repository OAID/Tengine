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

#include "mem_stat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api/c_api.h"
#include "utility/sys_port.h"
#include "utility/vector.h"
#include "utility/log.h"

#ifdef CONFIG_MEM_STAT

extern void (*enable_mem_stat)(void);
extern void (*disable_mem_stat)(void);

struct mem_stat
{
    int alloc_count;
    int free_count;
    int realloc_count;
    int max_block_size;
    int min_block_size;
    int peak_mem_size;
    int cur_mem_size;
};

struct block_stat
{
    void* ptr;
    int size;
};

static int mem_stat_skipped = 1;
static struct mem_stat mem_stat;
static struct vector* block_list;

DECLARE_AUTO_INIT_FUNC(init_mem_stat);
DECLARE_AUTO_EXIT_FUNC(release_mem_stat);

static int find_block_list(void* ptr)
{
    int n = get_vector_num(block_list);
    int i;

    for (i = 0; i < n; i++)
    {
        struct block_stat* block_stat = ( struct block_stat* )get_vector_data(block_list, i);

        if (block_stat->ptr == ptr)
            break;
    }

    if (i == n)
        return -1;

    return i;
}

static void real_enable_mem_stat(void)
{
    set_skip_stat(0);
}

static void real_disable_mem_stat(void)
{
    set_skip_stat(1);
}

static void init_mem_stat(void)
{
    memset(&mem_stat, 0x0, sizeof(mem_stat));
    mem_stat.min_block_size = 1 << 20;

    block_list = create_vector(sizeof(struct block_stat), NULL);

    enable_mem_stat = real_enable_mem_stat;
    disable_mem_stat = real_disable_mem_stat;
}

void dump_mem_stat(void)
{
    TLOG_INFO("memory usage stats:\n");
    TLOG_INFO("\talloc_count: %d\n", mem_stat.alloc_count);
    TLOG_INFO("\tfree_count: %d\n", mem_stat.free_count);
    TLOG_INFO("\trealloc_count: %d\n", mem_stat.realloc_count);
    TLOG_INFO("\tmax_block_size: %d\n", mem_stat.max_block_size);
    TLOG_INFO("\tmin_block_size: %d\n", mem_stat.min_block_size);
    TLOG_INFO("\tpeak_mem_size: %d\n", mem_stat.peak_mem_size);
    TLOG_INFO("\tcur_mem_size: %d\n", mem_stat.cur_mem_size);
}

static void release_mem_stat(void)
{
    release_vector(block_list);
    dump_mem_stat();
}

void set_skip_stat(int skip)
{
    mem_stat_skipped = skip;
}

int skip_stat(void)
{
    return mem_stat_skipped;
}

void* stat_malloc(int size)
{
    void* ptr = malloc(size);

    if (ptr == NULL)
    {
        TLOG_ERR("cannot alloc size: %d\n", size);
        TLOG_ERR("cur mem size: %d peak mem size: %d\n", mem_stat.cur_mem_size, mem_stat.peak_mem_size);

        return NULL;
    }

    mem_stat.alloc_count++;
    mem_stat.cur_mem_size += size;

    if (mem_stat.cur_mem_size > mem_stat.peak_mem_size)
        mem_stat.peak_mem_size = mem_stat.cur_mem_size;

    if (size > mem_stat.max_block_size)
        mem_stat.max_block_size = size;
    if (size < mem_stat.min_block_size)
        mem_stat.min_block_size = size;

    mem_stat_skipped = 1;

    struct block_stat block_stat;

    block_stat.ptr = ptr;
    block_stat.size = size;

    push_vector_data(block_list, &block_stat);

    mem_stat_skipped = 0;

    return ptr;
}

void stat_free(void* ptr)
{
    int idx = find_block_list(ptr);

    if (idx < 0)
    {
        /* a memory not allocated by us ? */
        free(ptr);
        return;
    }

    struct block_stat* block_stat = ( struct block_stat* )get_vector_data(block_list, idx);

    mem_stat.free_count++;
    mem_stat.cur_mem_size -= block_stat->size;

    mem_stat_skipped = 1;

    remove_vector_via_index(block_list, idx);

    mem_stat_skipped = 0;

    free(ptr);
}

void* stat_realloc(void* ptr, size_t size)
{
    if (ptr == NULL)
        return stat_malloc(size);

    int idx = find_block_list(ptr);

    if (idx < 0)
        return realloc(ptr, size);

    void* new_ptr = realloc(ptr, size);

    struct block_stat* block_stat = ( struct block_stat* )get_vector_data(block_list, idx);

    if (new_ptr == NULL)
    {
        TLOG_ERR("cannot realloc size: %d --> %d\n", block_stat->size, size);
        TLOG_ERR("cur mem size: %d peak mem size: %d\n", mem_stat.cur_mem_size, mem_stat.peak_mem_size);
        return NULL;
    }

    if (size > mem_stat.max_block_size)
        mem_stat.max_block_size = size;

    if (size < mem_stat.min_block_size)
        mem_stat.min_block_size = size;

    mem_stat.realloc_count++;
    mem_stat.cur_mem_size += size - block_stat->size;

    if (mem_stat.cur_mem_size > mem_stat.peak_mem_size)
        mem_stat.peak_mem_size = mem_stat.cur_mem_size;

    block_stat->ptr = new_ptr;
    block_stat->size = size;

    return new_ptr;
}

#endif
