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

#include "map.h"
#include "hash.h"

struct map
{
    const char* name;
    struct hash* h;
};

static unsigned int map_hash(const void* key, int key_size)
{
    unsigned int base = 0xdeadbeaf;
    const unsigned char* d = key;

    for (int i = 0; i < key_size; i++)
    {
        base += (( int )d[i] << (i % 20)) + d[i];
    }

    return base;
}

struct map* create_map(const char* name, void (*free_data)(void*))
{
    struct map* m = ( struct map* )sys_malloc(sizeof(struct map));
    struct hash* h;

    if (m == NULL)
        return NULL;

    m->name = strdup(name);

    h = create_hash(1024, map_hash, 1, free_data, 1);

    m->h = h;

    return m;
}

void release_map(struct map* m)
{
    free(( void* )m->name);
    destroy_hash(m->h);
    sys_free(m);
}

int insert_map_data(struct map* m, const char* key, void* data)
{
    struct hash* h = m->h;
    int size = strlen(key);

    return h->insert(h, key, size, data);
}

void* get_map_data(struct map* m, const char* key)
{
    struct hash* h = m->h;
    int size = strlen(key);

    return h->find(h, key, size);
}

int remove_map_data(struct map* m, const char* key)
{
    struct hash* h = m->h;
    int size = strlen(key);

    return h->delete (h, key, size);
}

int replace_map_data(struct map* m, const char* key, void* data)
{
    remove_map_data(m, key);
    return insert_map_data(m, key, data);
}

int get_map_num(struct map* m)
{
    struct hash* h = m->h;

    return h->get_entry_num(h);
}
