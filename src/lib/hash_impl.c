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

#include "hash_impl.h"

static inline void init_bucket(struct hash_bucket* b, int idx)
{
    b->entry_count = 0;
    b->idx = idx;

    init_list(&b->head);
    init_lock(&b->lock);

#ifdef HASH_STATS
    b->del_count = 0;
    b->hit_count = 0;
    b->ins_count = 0;
    b->search_count = 0;
#endif
}

static void release_hash_entry(struct hash_impl* h, struct hash_entry* e)
{
    struct hash_bucket* b = e->bucket;

#ifdef HASH_STATS
    b->del_count++;
    b->entry_count--;
#endif

    h->entry_num--;

    remove_list(&e->link);

    if (h->cpy_key)
        sys_free(e->key);

    if (h->free_func)
        h->free_func(e->data);

    sys_free(e);
}

static void release_hash(struct hash* t)
{
    struct hash_impl* h = ( struct hash_impl* )t;

    for (int i = 0; i < h->bucket_size; i++)
    {
        struct hash_entry* pos;
        struct hash_entry* dummy;
        struct hash_bucket* b = &h->bucket[i];

        if (h->mt_safe)
            lock(&b->lock);
#ifdef _MSC_VER
		list_for_each_entry_safe(pos, struct hash_entry, dummy, &b->head, link)
#else
		list_for_each_entry_safe(pos, dummy, &b->head, link)
#endif
		{
			release_hash_entry(h, pos);
		}

        if (h->mt_safe)
            unlock(&b->lock);
    }

    sys_free(h->bucket);
    sys_free(h);
}

static void config_hash(struct hash* t, int cpy_key, free_data_t func, int mt_safe, int max_num)
{
    struct hash_impl* h = ( struct hash_impl* )t;

    if (cpy_key >= 0)
        h->cpy_key = cpy_key;

    if (func != NULL)
        h->free_func = func;

    if (mt_safe > 0)
        h->mt_safe = mt_safe;

    if (max_num > 0)
        h->max_num = max_num;
}

static inline struct hash_bucket* find_bucket(struct hash_impl* h, const void* key, int key_size)
{
    unsigned int hash_int = h->hash_func(key, key_size);
    int bucket_idx = hash_int % h->bucket_size;

    return &h->bucket[bucket_idx];
}

static inline int compare_hash_key(const void* key0, int key0_size, const void* key1, int key1_size)
{
    if (key0_size != key1_size)
        return -1;

    return memcmp(key0, key1, key0_size);
}
static struct hash_entry* find_entry(struct hash* t, const void* key, int key_size)
{
    struct hash_impl* h = ( struct hash_impl* )t;
    struct hash_bucket* b = find_bucket(h, key, key_size);
    struct hash_entry* e;
    int found = 0;

    if (h->mt_safe)
        lock(&b->lock);

#ifdef HASH_STATS
    b->search_count++;
#endif

#ifdef _MSC_VER
	list_entry_for_each(e, struct hash_entry, &b->head, link)
#else
	list_entry_for_each(e, &b->head, link)
#endif
    {
        if (compare_hash_key(e->key, e->key_size, key, key_size) == 0)
        {
            found = 1;
#ifdef HASH_STATS
            b->hit_count++;
#endif
            break;
        }
    }

    if (h->mt_safe)
        unlock(&b->lock);

    if (found)
        return e;
    else
        return NULL;
}

static hash_entry_t wrapper_find_entry(struct hash* t, const void* key, int key_size)
{
    return find_entry(t, key, key_size);
}

static int insert_hash(struct hash* t, const void* key, int key_size, void* data)
{
    struct hash_impl* h = ( struct hash_impl* )t;
    struct hash_bucket* b = find_bucket(h, key, key_size);
    struct hash_entry* e;

    if (h->mt_safe)
        lock(&b->lock);

#ifdef _MSC_VER
	list_entry_for_each(e, struct hash_entry, &b->head, link)
#else
	list_entry_for_each(e, &b->head, link)
#endif
    {
        if (compare_hash_key(e->key, e->key_size, key, key_size) == 0)
        {
            // already exists, report error
            if (h->mt_safe)
                unlock(&b->lock);

            return -1;
        }
    }

    e = ( struct hash_entry* )sys_malloc(sizeof(struct hash_entry));

    /* handle key */
    e->key_size = key_size;

    if (h->cpy_key)
    {
        e->key = sys_malloc(key_size);
        memcpy(e->key, key, key_size);
    }
    else
        e->key = ( void* )key;

    e->data = data;
    e->bucket = b;

    h->entry_num++;

#ifdef HASH_STATS
    b->ins_count++;
#endif

    append_list(&e->link, &b->head);

    if (h->mt_safe)
        unlock(&b->lock);

    return 0;
}

static int delete_hash(struct hash* t, const void* key, int key_size)
{
    struct hash_impl* h = ( struct hash_impl* )t;
    struct hash_bucket* b = find_bucket(h, key, key_size);
    struct hash_entry* e;

    int found = 0;

    if (h->mt_safe)
        lock(&b->lock);

#ifdef _MSC_VER
	list_entry_for_each(e, struct hash_entry, &b->head, link)
#else
	list_entry_for_each(e, &b->head, link)
#endif
    {
        if (compare_hash_key(e->key, e->key_size, key, key_size) == 0)
        {
            found = 1;
            break;
        }
    }

    if (!found)
    {
        if (h->mt_safe)
            unlock(&b->lock);
        return -1;
    }

    release_hash_entry(h, e);

    if (h->mt_safe)
        unlock(&b->lock);

    return 0;
}

static void* find_hash(struct hash* t, const void* key, int key_size)
{
    struct hash_entry* e = find_entry(t, key, key_size);

    if (e == NULL)
        return NULL;

    return e->data;
}

static void* get_data(hash_entry_t* e)
{
    struct hash_entry* entry = ( struct hash_entry* )e;

    return entry->data;
}

static void remove_entry(struct hash* t, hash_entry_t* e)
{
    struct hash_impl* h = ( struct hash_impl* )t;
    struct hash_entry* entry = ( struct hash_entry* )e;
    struct hash_bucket* b = entry->bucket;

    if (h->mt_safe)
        lock(&b->lock);

    release_hash_entry(h, entry);

    if (h->mt_safe)
        unlock(&b->lock);
}

static int get_entry_num(struct hash* t)
{
    struct hash_impl* h = ( struct hash_impl* )t;

    return h->entry_num;
}

static void reset_seq_access(struct hash* t)
{
    struct hash_impl* h = ( struct hash_impl* )t;

    h->seq_ptr = NULL;
}

static hash_entry_t get_next_entry(struct hash* t)
{
    struct hash_impl* h = ( struct hash_impl* )t;
    struct hash_entry* e = NULL;
    int start_bucket = 0;
    struct hash_bucket* b;

    if (h->seq_ptr)
    {
        b = h->seq_ptr->bucket;

        if (!list_entry_is_last(h->seq_ptr, &b->head, link))
        {
#ifdef _MSC_VER            
			h->seq_ptr = list_entry_next(h->seq_ptr, struct hash_entry, link);
#else
			h->seq_ptr = list_entry_next(h->seq_ptr, link);
#endif
            return h->seq_ptr;
        }

        // move to next bucket
        start_bucket = b->idx + 1;
    }
    else
        start_bucket = 0;

    for (int i = start_bucket; i < h->bucket_size; i++)
    {
        b = &h->bucket[i];

        if (!list_empty(&b->head))
        {
            e = list_entry_head(&b->head, struct hash_entry, link);
            h->seq_ptr = e;
            break;
        }
    }

    return e;
}

static int init_hash(struct hash* t, int bucket_size, hash_key_t func)
{
    struct hash_impl* h = ( struct hash_impl* )t;

    h->bucket_size = bucket_size;
    h->hash_func = func;
    h->free_func = NULL;
    h->mt_safe = 1;
    h->cpy_key = 1;
    h->entry_num = 0;
    h->max_num = -1;
    h->seq_ptr = NULL;

    h->bucket = sys_malloc(sizeof(struct hash_bucket) * bucket_size);

    for (int i = 0; i < bucket_size; i++)
        init_bucket(&h->bucket[i], i);

    return 0;
}

struct hash* create_hash_impl(void)
{
    struct hash* h = sys_malloc(sizeof(struct hash_impl));

    h->init = init_hash;
    h->release = release_hash;
    h->config = config_hash;
    h->insert = insert_hash;
    h->delete = delete_hash;
    h->find = find_hash;
    h->find_entry = wrapper_find_entry;
    h->get_data = get_data;
    h->remove_entry = remove_entry;
    h->get_entry_num = get_entry_num;
    h->reset_seq_access = reset_seq_access;
    h->get_next_entry = get_next_entry;

    return h;
}
