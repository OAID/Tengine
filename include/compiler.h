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

#ifndef __SYS_COMPILER_H__
#define __SYS_COMPILER_H__

#ifdef _MSC_VER
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __attribute((visibility("default")))
#endif

#ifndef offsetof
#define offsetof(TYPE, MEMBER) ((size_t) & (( TYPE* )0)->MEMBER)
#endif

#ifdef _MSC_VER
#define container_of(ptr, type, member) ( type* )(( char* )ptr - offsetof(type, member))
#else
#define container_of(ptr, type, member)                         \
    ({                                                          \
        const typeof((( type* )0)->member)* __mptr = (ptr);     \
        ( type* )(( char* )__mptr - offsetof(type, member));    \
    })
#endif

#define UNIQ_DUMMY_NAME_WITH_LINE0(a, b)                        \
    auto_dummy_##a##_##b

#define UNIQ_DUMMY_NAME_WITH_LINE(a, b)                         \
    UNIQ_DUMMY_NAME_WITH_LINE0(a, b)

#define UNIQ_DUMMY_NAME(name)                                   \
    UNIQ_DUMMY_NAME_WITH_LINE(name, __LINE__)

#ifdef __cplusplus
#define DECLARE_AUTO_INIT_FUNC(f)                               \
    static void f(void);                                        \
    struct f##_t_                                               \
    {                                                           \
        f##_t_(void)                                            \
        {                                                       \
            f();                                                \
        }                                                       \
    };                                                          \
    static f##_t_ f##_;                                         \
    static void f(void)
#elif defined(_MSC_VER)
#pragma section(".CRT$XCU", read)
#define DECLARE_AUTO_INIT_FUNC(func_name)                       \
    static void func_name(void);                                \
    __pragma(data_seg(".CRT$XIU"))                              \
    static void(*UNIQ_DUMMY_NAME(initptr))(void) = func_name;   \
    __pragma(data_seg())
#else
#define DECLARE_AUTO_INIT_FUNC(func_name)                       \
    static void func_name(void) __attribute__((constructor));   \
    static void func_name(void)
#endif

#ifdef _MSC_VER
#define DECLARE_AUTO_EXIT_FUNC(func_name)                       \
    static void func_name(void);                                \
    __pragma(data_seg(".CRT$XPU"))                              \
    static void(*UNIQ_DUMMY_NAME(exitptr))(void) = func_name;   \
    __pragma(data_seg())
#else
#define DECLARE_AUTO_EXIT_FUNC(func_name)                       \
    static void(func_name)(void) __attribute__((destructor))    \
    static void func_name(void)
#endif

#ifdef _MSC_VER
#define __attribute__(a)
#endif

#endif
