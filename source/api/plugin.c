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

#include "api/c_api.h"

#include "utility/sys_port.h"
#include "utility/vector.h"
#include "utility/log.h"

#include <stdio.h>
#include <string.h>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef _MSC_VER
typedef int (*fun_ptr)(void);
typedef HINSTANCE so_handle_t;
#else
typedef void* so_handle_t;
#endif

struct plugin_header
{
    char* name;
    char* fname;
    so_handle_t handle;
};

static struct vector* plugin_list = NULL;

static int exec_so_func(so_handle_t handle, const char* func_name)
{
#ifdef _MSC_VER
    void* func = (fun_ptr)GetProcAddress(handle, func_name);
#else
    void* func = dlsym(handle, func_name);
#endif

    if (func == NULL)
    {
#ifdef _MSC_VER
        TLOG_ERR("find func: %s failed, error code %d\n", func_name, GetLastError());
#else
        TLOG_ERR("find func: %s failed, reason %s\n", func_name, dlerror());
#endif

        return -1;
    }

    int (*call_func)(void) = (int (*)(void))func;

    if (call_func() < 0)
    {
        TLOG_ERR("exec so func: %s failed\n", func_name);
        return -1;
    }
    TLOG_INFO("function:%s executed\n", func_name);

    return 0;
}

int load_tengine_plugin(const char* plugin_name, const char* file_name, const char* init_func_name)
{
    struct plugin_header header;

    /* TODO: MT safe */

    if (plugin_list == NULL)
    {
        plugin_list = create_vector(sizeof(struct plugin_header), NULL);

        if (plugin_list == NULL)
        {
            return -1;
        }
    }

    /* check if name duplicated */
    int list_num = get_vector_num(plugin_list);

    for (int i = 0; i < list_num; i++)
    {
        struct plugin_header* h = (struct plugin_header*)get_vector_data(plugin_list, i);

        if (!strcmp(h->name, plugin_name))
        {
            TLOG_ERR("duplicated plugin name: %s\n", plugin_name);
            return -1;
        }
    }

    /* load the so */
#ifdef _MSC_VER
    header.handle = LoadLibraryA(file_name);
#else
    header.handle = dlopen(file_name, RTLD_LAZY);
#endif

    if (header.handle == NULL)
    {
#ifdef _MSC_VER
        TLOG_ERR("load plugin failed: error code %d\n", GetLastError());
#else
        TLOG_ERR("load plugin failed: %s\n", dlerror());
#endif
        return -1;
    }

    /* execute the init function */
    if (init_func_name && exec_so_func(header.handle, init_func_name) < 0)
    {
#ifdef _MSC_VER
        FreeLibrary(header.handle);
#else
        dlclose(header.handle);
#endif

        return -1;
    }

    size_t plugin_name_length = strlen(plugin_name);
    size_t file_name_length = strlen(file_name);

    header.name = (char*)sys_malloc(plugin_name_length);
    header.fname = (char*)sys_malloc(file_name_length);

    memcpy(header.name, plugin_name, plugin_name_length);
    memcpy(header.fname, file_name, file_name_length);

    push_vector_data(plugin_list, &header);

    return 0;
}

int unload_tengine_plugin(const char* plugin_name, const char* rel_func_name)
{
    if (plugin_list == NULL)
        return -1;

    int list_num = get_vector_num(plugin_list);
    struct plugin_header* target = NULL;

    for (int i = 0; i < list_num; i++)
    {
        struct plugin_header* h = (struct plugin_header*)get_vector_data(plugin_list, i);

        if (!strcmp(h->name, plugin_name))
        {
            target = h;
            break;
        }
    }

    if (target == NULL)
    {
        return -1;
    }

    if (rel_func_name)
        exec_so_func(target->handle, rel_func_name);

#ifdef _MSC_VER
    FreeLibrary(target->handle);
#else
    dlclose(target->handle);
#endif

    remove_vector_via_pointer(plugin_list, target);

    if (get_vector_num(plugin_list) == 0)
    {
        release_vector(plugin_list);
    }

    return 0;
}

int get_tengine_plugin_number(void)
{
    int plugin_num = 0;

    if (plugin_list)
        plugin_num = get_vector_num(plugin_list);

    return plugin_num;
}

const char* get_tengine_plugin_name(int idx)
{
    int plugin_num = get_tengine_plugin_number();

    if (idx >= plugin_num)
        return NULL;

    struct plugin_header* h = (struct plugin_header*)get_vector_data(plugin_list, idx);

    return h->name;
}
