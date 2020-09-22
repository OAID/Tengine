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

#include <string.h>

#include "vector.h"

struct vector* create_vector(int elem_size, void (*free_data)(void*))
{
    struct vector* v = ( struct vector* )sys_malloc(sizeof(struct vector));

    if (v == NULL)
        return NULL;

    v->elem_num = 0;
    v->elem_size = elem_size;
    v->free_func = free_data;
    v->entry_size = elem_size + sizeof(struct vector_entry);
    v->entry_size = (v->entry_size + VECTOR_ALIGN_SIZE) & (~(VECTOR_ALIGN_SIZE - 1));

    v->ahead_num = 8;

    v->space_num = v->ahead_num;

    v->real_mem = sys_malloc(v->entry_size * v->space_num + VECTOR_ALIGN_SIZE);
    v->mem = ( void* )((( long )v->real_mem) & (~(VECTOR_ALIGN_SIZE - 1)));

    for (int i = 0; i < v->space_num; i++)
    {
        struct vector_entry* e = get_vector_entry(v, i);
        e->valid = 0;
    }

    return v;
}

int resize_vector(struct vector* v, int new_size)
{
    void* new_mem;

    /* need to free the reduced element */
    if (new_size <= v->elem_num)
    {
        for (int i = v->elem_num - 1; i > new_size - 1; i--)
            remove_vector_by_idx(v, i);

        return 0;
    }

    if (new_size <= v->space_num)
    {
        v->elem_num = new_size;
        return 0;
    }

    new_mem = sys_realloc(v->real_mem, new_size * v->entry_size + VECTOR_ALIGN_SIZE);

    if (new_mem == NULL)
        return -1;

    v->real_mem = new_mem;
    v->mem = ( void* )((( long )(v->real_mem)) & (~(VECTOR_ALIGN_SIZE - 1)));

    for (int i = v->space_num; i < new_size; i++)
    {
        struct vector_entry* e = get_vector_entry(v, i);
        e->valid = 0;
    }

    v->space_num = new_size;

    return 0;
}

void remove_vector_data_not_tail(struct vector* v, int idx)
{
    struct vector_entry* e = NULL;
    void* ptr;
    int cpy_number;

    free_vector_data_resource(v, idx);

    ptr = ( char* )v->mem + idx * v->entry_size;
    cpy_number = v->elem_num - 1 - idx;

    memmove(ptr, ( char* )ptr + v->entry_size, cpy_number * v->entry_size);

    v->elem_num--;

    // clear the valid flag
    e = get_vector_entry(v, v->elem_num);
    e->valid = 0;
}

int remove_vector_data(struct vector* v, void* data)
{
    int n = v->elem_num;
    int idx;

    for (idx = 0; idx < n; idx++)
    {
        void* content = get_vector_data(v, idx);

        if (memcmp(content, data, v->elem_size) == 0)
            break;
    }

    if (idx == n)
        return -1;

    remove_vector_by_idx(v, idx);

    return 0;
}

void release_vector(struct vector* v)
{
    for (int i = 0; i < v->elem_num; i++)
    {
        free_vector_data_resource(v, i);
    }

    sys_free(v->real_mem);
    sys_free(v);
}
