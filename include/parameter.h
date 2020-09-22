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

#ifndef __PARAMETER_H__
#define __PARAMETER_H__

#include <string.h>

#include "sys_port.h"
#include "param_type.h"
#include "va_arg_util.h"

#define ADD_PARAM_ENTRY(map, idx, s, e)                   \
    {                                                     \
        struct s dummy;                                   \
        struct param_entry* p_entry = &map->entries[idx]; \
        int offset = ( long )&(( typeof(dummy)* )0)->e;   \
        int size = sizeof(dummy.e);                       \
        int type = GET_PARAM_ENTRY_TYPE(dummy.e);         \
        p_entry->type = type;                             \
        p_entry->name = #e;                               \
        p_entry->offset = offset;                         \
        p_entry->size = size;                             \
        idx++;                                            \
    }

#ifdef CONFIG_DISABLE_PARAM_ACCESS

#define GET_PARAM_PARSE_MAP(s) (( void* )0)

#define DEFINE_PARM_PARSE_ENTRY(s, ...)                                                                         \
    static int access_param_entry(void* param_mem, const char* entry_name, int entry_type, void* buf, int size, \
                                  int set)                                                                      \
    {                                                                                                           \
        return -1;                                                                                              \
    }

#else

#define GET_PARAM_PARSE_MAP(s) get_##s##_param_parse_map()

#define DEFINE_PARM_PARSE_ENTRY(s, ...)                                                                         \
    static inline struct param_entry_map* GET_PARAM_PARSE_MAP(s)                                                \
    {                                                                                                           \
        static int inited = 0;                                                                                  \
        static struct param_entry_map* map;                                                                     \
                                                                                                                \
        if(inited)                                                                                              \
            return map;                                                                                         \
                                                                                                                \
        int entry_number = COUNT_VA_ARG(__VA_ARGS__) + 1;                                                       \
        int idx = 0;                                                                                            \
                                                                                                                \
        map = ( struct param_entry_map* )sys_malloc(sizeof(struct param_entry_map) +                                \
                                                sizeof(struct param_entry) * entry_number);                     \
                                                                                                                \
        map->number = entry_number;                                                                             \
                                                                                                                \
        WALK_VA_ARG(ADD_PARAM_ENTRY, 3, map, idx, s, __VA_ARGS__);                                              \
        inited = 1;                                                                                             \
        return map;                                                                                             \
    }                                                                                                           \
                                                                                                                \
    static int access_param_entry(void* param_mem, const char* entry_name, int entry_type, void* buf, int size, \
                                  int set)                                                                      \
    {                                                                                                           \
        int ret;                                                                                                \
                                                                                                                \
        struct s* param = ( struct s* )(param_mem);                                                             \
                                                                                                                \
        if(set)                                                                                                 \
            ret = SET_PARAM_ENTRY(param, s, entry_name, entry_type, buf, size);                                 \
        else                                                                                                    \
            ret = GET_PARAM_ENTRY(param, s, entry_name, entry_type, buf, size);                                 \
                                                                                                                \
        return ret;                                                                                             \
    }

#endif

/* data structure definition */

struct param_entry
{
    const char* name;
    int type;
    int offset;
    int size;
};

struct param_entry_map
{
    int number;
    struct param_entry entries[0];
};

static inline int access_ds_entry(void* param_blob, struct param_entry_map* map, const char* name, int type, void* val,
                                  int val_size, int set)
{
    int entry_number = map->number;
    struct param_entry* p_entry = map->entries;

    for(int i = 0; i < entry_number; i++, p_entry++)
    {
        if(strcmp(p_entry->name, name))
            continue;

        if((type && p_entry->type && (type != p_entry->type)) || val_size != p_entry->size)
            return -1;

        void* mem_base = ( char* )param_blob + p_entry->offset;

        if(set)
            memcpy(mem_base, val, val_size);
        else
            memcpy(val, mem_base, val_size);

        return 0;
    }

    return -1;
}

#define GET_PARAM_ENTRY(param_blob, set, name, type, ptr, size) \
    access_ds_entry(param_blob, GET_PARAM_PARSE_MAP(set), name, type, ptr, size, 0)

#define SET_PARAM_ENTRY(param_blob, set, name, type, ptr, size) \
    access_ds_entry(param_blob, GET_PARAM_PARSE_MAP(set), name, type, ptr, size, 1)

#endif
