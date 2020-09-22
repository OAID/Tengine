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

#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <string.h>

#include "sys_port.h"

#define ALIGNED_VECTOR

#define VECTOR_ALIGN_SIZE 8

struct vector_entry
{
    int valid;
    unsigned char data[0];
};

struct vector
{
    int elem_size;
    int elem_num;

    int entry_size;
    int space_num;
    int ahead_num;
    void * real_mem;
    void* mem;
    void (*free_func)(void*);
};

struct vector* create_vector(int elem_size, void (*free_data)(void*));
void release_vector(struct vector* v);

int resize_vector(struct vector* v, int new_size);
void remove_vector_data_not_tail(struct vector* v, int idx);

static inline struct vector_entry* get_vector_entry(struct vector* v, int idx)
{
    return ( struct vector_entry* )(( char* )v->mem + v->entry_size * idx);
}

static inline void free_vector_data_resource(struct vector* v, int idx)
{
    struct vector_entry* e = get_vector_entry(v, idx);

    if(e->valid && v->free_func)
        v->free_func(e->data);

    e->valid = 0;
}

static inline int set_vector_data(struct vector* v, int idx, void* data)
{
    struct vector_entry* e = NULL;

    if(idx >= v->elem_num)
        return -1;

    free_vector_data_resource(v, idx);

    e = get_vector_entry(v, idx);
    e->valid = 1;

    memcpy(e->data, data, v->elem_size);

    return 0;
}

static inline int push_vector_data(struct vector* v, void* data)
{
    if(v->elem_num == v->space_num && resize_vector(v, v->elem_num + v->ahead_num) < 0)

        return -1;

    v->elem_num++;
    set_vector_data(v, v->elem_num - 1, data);

    return 0;
}

static inline void* get_vector_data(struct vector* v, int idx)
{
    if(idx >= v->elem_num)
        return NULL;

    struct vector_entry* e = get_vector_entry(v, idx);

    return e->data;
}

static inline int get_vector_num(struct vector* v)
{
    return v->elem_num;
}

static inline void remove_vector_by_idx(struct vector* v, int idx)
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

int remove_vector_data(struct vector* v, void* data);

#endif
