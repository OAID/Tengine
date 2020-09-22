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

#ifndef __HASH_H__
#define __HASH_H__

#include "sys_port.h"

typedef unsigned int (*hash_key_t)(const void* key, int size);
typedef void* hash_entry_t;
typedef void (*free_data_t)(void* data);

struct hash
{
    int (*init)(struct hash* t, int bucket_size, hash_key_t h);
    void (*release)(struct hash* t);

    void (*config)(struct hash* t, int cpy_key, free_data_t func, int mt_safe, int max_num);

    /* if the same key exists, return -1 */
    int (*insert)(struct hash* t, const void* key, int key_size, void* data);

    /* if none entry with target key, return -1*/
    int (*delete)(struct hash* t, const void* key, int size);

    void* (*find)(struct hash* t, const void* key, int size);

    hash_entry_t (*find_entry)(struct hash* t, const void* key, int size);

    void* (*get_data)(hash_entry_t* e);
    void (*remove_entry)(struct hash* t, hash_entry_t* e);

    /* sequential interface */
    int (*get_entry_num)(struct hash* t);

    void (*reset_seq_access)(struct hash* t);
    hash_entry_t (*get_next_entry)(struct hash* t);
};

struct hash* create_hash(int bucket_size, hash_key_t hash_func, int cpy_key, free_data_t free_func, int mt_safe);

void destroy_hash(struct hash* h);

#endif
