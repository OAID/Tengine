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

#include "module.h"
#include "vector.h"

struct module_init_func_entry
{
    const char* name;
    module_init_func_t func;
    void* arg;
    int critical;
};

struct module_exit_func_entry
{
    const char* name;
    module_exit_func_t func;
    void* arg;
};

static int init_vector_created = 0;
static int exit_vector_created = 0;

static struct vector* init_vector[MOD_MAX_LEVEL];
static struct vector* exit_vector[MOD_MAX_LEVEL];

/* this function may be called before main(), in constructor */
static int register_module_init(int level, const char* name, module_init_func_t func, void* arg, int crit)
{
    struct module_init_func_entry e;

    if (!init_vector_created)
    {
        for (int i = 0; i < MOD_MAX_LEVEL; i++)
        {
            init_vector[i] = create_vector(sizeof(struct module_init_func_entry), NULL);
        }

        init_vector_created = 1;
    }

    if (init_vector[level] == NULL)
        return -1;

    e.name = name;
    e.func = func;
    e.arg = arg;
    e.critical = crit;

    push_vector_data(init_vector[level], &e);

    return 0;
}

int register_norm_module_init(int level, const char* name, module_init_func_t func, void* arg)
{
    return register_module_init(level, name, func, arg, 0);
}

int register_crit_module_init(int level, const char* name, module_init_func_t func, void* arg)
{
    return register_module_init(level, name, func, arg, 1);
}

int register_module_exit(int level, const char* name, module_exit_func_t func, void* arg)
{
    struct module_exit_func_entry e;

    if (!exit_vector_created)
    {
        for (int i = 0; i < MOD_MAX_LEVEL; i++)
        {
            exit_vector[i] = create_vector(sizeof(struct module_exit_func_entry), NULL);
        }

        exit_vector_created = 1;
    }

    if (exit_vector[level] == NULL)
        return -1;

    e.name = name;
    e.func = func;
    e.arg = arg;

    push_vector_data(exit_vector[level], &e);

    return 0;
}

int exec_module_init(int stop_on_all_error)
{
    for (int i = 0; i < MOD_MAX_LEVEL; i++)
    {
        struct vector* v = init_vector[i];

        if (v == NULL)
            continue;

        int vector_num = get_vector_num(v);

        for (int j = 0; j < vector_num; j++)
        {
            struct module_init_func_entry* e;
            int ret;

            e = ( struct module_init_func_entry* )get_vector_data(v, j);
            ret = e->func(e->arg);

            // fprintf(stderr, "on executing %s() \n", e->name);

            if (ret < 0 && (stop_on_all_error || e->critical))
                return -1;
        }
    }

    // free the memory used
    for (int i = 0; i < MOD_MAX_LEVEL; i++)
    {
        struct vector* v = init_vector[i];
        if (v == NULL)
            continue;
        release_vector(v);
        init_vector[i] = NULL;
    }

    return 0;
}

int exec_module_exit(int stop_on_error)
{
    for (int i = MOD_MAX_LEVEL - 1; i >= 0; i--)
    {
        struct vector* v = exit_vector[i];

        if (v == NULL)
            continue;

        int vector_num = get_vector_num(v);

        for (int j = 0; j < vector_num; j++)
        {
            struct module_exit_func_entry* e;
            int ret;

            e = ( struct module_exit_func_entry* )get_vector_data(v, j);

            ret = e->func(e->arg);

            if (ret < 0 && !stop_on_error)
                return -1;
        }
    }

    for (int i = 0; i < MOD_MAX_LEVEL; i++)
    {
        struct vector* v = exit_vector[i];

        if (v == NULL)
            continue;

        release_vector(v);
    }

    return 0;
}
