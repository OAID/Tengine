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

#include "utility/vector.h"

#include "defines.h"
#include "utility/sys_port.h"
#include "utility/math.h"

#include <string.h>


typedef struct vector_entry
{
    int valid;
    unsigned char data[];
} vector_entry_t;


static inline vector_entry_t* get_vector_entry(vector_t* v, int idx)
{
    return (vector_entry_t*)((char*)v->mem + v->entry_size * idx);
}


static inline void free_vector_data_resource(vector_t* v, int idx)
{
    vector_entry_t* e = get_vector_entry(v, idx);

    if(e->valid && v->free_func)
    {
        v->free_func(e->data);
    }

    e->valid = 0;
}


static inline void remove_vector_data_not_tail(vector_t* v, int idx)
{
    vector_entry_t* entry_ptr = NULL;
    void* start_data_ptr;
    int remaining_elements_count;

    free_vector_data_resource(v, idx);

    start_data_ptr = (char*)v->mem + idx * v->entry_size;
    remaining_elements_count = v->elem_num - 1 - idx;

    memmove(start_data_ptr, (char*)start_data_ptr + v->entry_size, (size_t)remaining_elements_count * v->entry_size);

    v->elem_num--;

    // clear the valid flag
    entry_ptr = get_vector_entry(v, v->elem_num);
    entry_ptr->valid = 0;
}


vector_t* create_vector(int elem_size, void (*free_data)(void*))
{
    vector_t* v = (vector_t*)sys_malloc(sizeof(vector_t));

    if (v == NULL)
    {
        return NULL;
    }

    v->elem_num = 0;
    v->elem_size = elem_size;
    v->free_func = free_data;
    v->entry_size = align(elem_size + (int)sizeof(vector_entry_t), TE_VECTOR_ALIGN_SIZE);

    v->ahead_num = 8;

    v->space_num = v->ahead_num;

    v->real_mem = sys_malloc(v->entry_size * v->space_num + TE_VECTOR_ALIGN_SIZE);
    v->mem = align_address(v->real_mem, TE_VECTOR_ALIGN_SIZE);

    for (int i = 0; i < v->space_num; i++)
    {
        vector_entry_t* e = get_vector_entry(v, i);
        e->valid = 0;
    }

    return v;
}


void release_vector(vector_t* v)
{
    for (int i = 0; i < v->elem_num; i++)
    {
        free_vector_data_resource(v, i);
    }

    sys_free(v->real_mem);
    sys_free(v);
}


int get_vector_num(vector_t* v)
{
    if (NULL != v)
    {
        return v->elem_num;
    }

    return 0;
}


int resize_vector(vector_t* v, int new_size)
{
    void* new_mem = NULL;

    /* need to free the reduced element */
    if (new_size < v->elem_num)
    {
        for (int i = v->elem_num - 1; i > new_size - 1; i--)
        {
            remove_vector_via_index(v, i);
        }

        return 0;
    }

    if (new_size <= v->space_num)
    {
        v->elem_num = new_size;
        return 0;
    }

    new_mem = sys_realloc(v->real_mem, new_size * v->entry_size + TE_VECTOR_ALIGN_SIZE);

    if (new_mem == NULL)
    {
        return -1;
    }

    v->real_mem = new_mem;
    v->mem = ( void* )(((size_t)(v->real_mem)) & (~(TE_VECTOR_ALIGN_SIZE - 1)));

    for (int i = v->space_num; i < new_size; i++)
    {
        vector_entry_t* e = get_vector_entry(v, i);
        e->valid = 0;
    }

    v->space_num = new_size;

    return 0;
}


int push_vector_data(vector_t* v, void* data)
{
    if(v->elem_num == v->space_num && resize_vector(v, v->elem_num + v->ahead_num) < 0)
    {
        return -1;
    }

    v->elem_num++;
    set_vector_data(v, v->elem_num - 1, data);

    return 0;
}


int set_vector_data(vector_t* v, int idx, void* data)
{
    vector_entry_t* e = NULL;

    if(idx >= v->elem_num)
        return -1;

    free_vector_data_resource(v, idx);

    e = get_vector_entry(v, idx);
    e->valid = 1;

    memcpy(e->data, data, v->elem_size);

    return 0;
}


void* get_vector_data(vector_t* v, int index)
{
    if(index >= v->elem_num)
    {
        return NULL;
    }

    vector_entry_t* e = get_vector_entry(v, index);

    return e->data;
}


int remove_vector_via_pointer(vector_t* v, void* data)
{
    const int count = v->elem_num;
    int index;

    for (index = 0; index < count; index++)
    {
        void* content = get_vector_data(v, index);

        if (memcmp(content, data, v->elem_size) == 0)
        {
            break;
        }
    }

    if (count == index)
    {
        return -1;
    }

    remove_vector_via_index(v, index);
    return 0;
}


void remove_vector_via_index(vector_t* v, int idx)
{
    // the last one
    if(idx == v->elem_num - 1)
    {
        free_vector_data_resource(v, idx);
        v->elem_num--;

        return;
    }

    remove_vector_data_not_tail(v, idx);
}
