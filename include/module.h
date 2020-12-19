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
 * Author: lswang@openailab.com
 */

#ifndef __TENGINE_MODULE_H__
#define __TENGINE_MODULE_H__

#include <stdio.h>

#include "compiler.h"

#ifdef _MSC_VER
#define SECNAME ".CRT$XCG"
#pragma section(SECNAME, long, read)
typedef void(__cdecl* _PVFV)();
#endif

enum
{
    MOD_CORE_LEVEL = 0,
    MOD_DEVICE_LEVEL,
    MOD_OP_LEVEL,
    MOD_FUNC_LEVEL,
    MOD_PLUGIN_LEVEL,
    MOD_MAX_LEVEL
};

typedef int (*module_init_func_t)(void*);
typedef int (*module_exit_func_t)(void*);

int register_norm_module_init(int level, const char* name, module_init_func_t func, void* arg);
int register_crit_module_init(int level, const char* name, module_init_func_t func, void* arg);
int register_module_exit(int level, const char* name, module_exit_func_t func, void* arg);

int exec_module_init(int stop_on_all_error);
int exec_module_exit(int stop_on_all_error);

#ifdef _MSC_VER
#define REGISTER_MODULE_INIT_ARG(level, name, func, arg)   \
    static void dump_func(void)                            \
    {                                                      \
        register_norm_module_init(level, name, func, arg); \
    }                                                      \
    __declspec(allocate(SECNAME)) static _PVFV dump_ptr[] = {dump_func};
#else
#define REGISTER_MODULE_INIT_ARG(level, name, func, arg)   \
    DECLARE_AUTO_INIT_FUNC(UNIQ_DUMMY_NAME(mod_init));     \
    static void UNIQ_DUMMY_NAME(mod_init)(void)            \
    {                                                      \
        register_norm_module_init(level, name, func, arg); \
    }
#endif

#ifdef _MSC_VER
#define REGISTER_CRIT_MODULE_INIT_ARG(level, name, func, arg) \
    static void dump_func(void)                               \
    {                                                         \
        register_crit_module_init(level, name, func, arg);    \
    }                                                         \
    __declspec(allocate(SECNAME)) static _PVFV dump_ptr[] = {dump_func};
#else
#define REGISTER_CRIT_MODULE_INIT_ARG(level, name, func, arg) \
    DECLARE_AUTO_INIT_FUNC(UNIQ_DUMMY_NAME(mod_init));        \
    static void UNIQ_DUMMY_NAME(mod_init)(void)               \
    {                                                         \
        register_crit_module_init(level, name, func, arg);    \
    }
#endif

#define REGISTER_CRIT_MODULE_INIT(level, name, func) REGISTER_CRIT_MODULE_INIT_ARG(level, name, func, NULL)

#define REGISTER_MODULE_INIT(level, name, func) REGISTER_MODULE_INIT_ARG(level, name, func, NULL)

#define REGISTER_MODULE_EXIT_ARG(level, name, func, arg) \
    DECLARE_AUTO_INIT_FUNC(UNIQ_DUMMY_NAME(mod_exit));   \
    static void UNIQ_DUMMY_NAME(mod_exit)(void)          \
    {                                                    \
        register_module_exit(level, name, func, arg);    \
    }

#define REGISTER_CRIT_MODULE_EXIT_ARG(level, name, func, arg) \
    DECLARE_AUTO_INIT_FUNC(UNIQ_DUMMY_NAME(mod_exit));        \
    static void UNIQ_DUMMY_NAME(mod_exit)(void)               \
    {                                                         \
        register_module_exit(level, name, func, arg);         \
    }

#define REGISTER_CRIT_MODULE_EXIT(level, name, func) REGISTER_CRIT_MODULE_EXIT_ARG(level, name, func, NULL)

#define REGISTER_MODULE_EXIT(level, name, func) REGISTER_MODULE_EXIT_ARG(level, name, func, NULL)

#endif
