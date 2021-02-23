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

#ifndef __HASH_IMPL__
#define __HASH_IMPL__

#include "hash.h"
#include "list.h"

#define HASH_STATS

struct hash_bucket
{
	int entry_count;
	lock_t lock;
	int idx;

	struct list head;

#ifdef HASH_STATS
	uint64_t ins_count;
	uint64_t del_count;
	uint64_t search_count;
	uint64_t hit_count;
#endif
};

struct hash_entry
{
    void* data;
    void* key;
    int key_size;
    struct list link;
    struct hash_bucket* bucket;
};

struct hash_impl
{
    struct hash hash_interface;

    /* data fields */
    int bucket_size;
    struct hash_bucket* bucket;

    hash_key_t hash_func;
    free_data_t free_func;

    int cpy_key;
    int mt_safe;

    int entry_num;
    int max_num;

    struct hash_entry* seq_ptr;
};

#endif
