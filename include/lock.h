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

#ifndef __SYS_LOCK_H__
#define __SYS_LOCK_H__

#ifdef CONFIG_BAREMETAL_BUILD

typedef int lock_t;

static inline void init_lock(lock_t* lock)
{
    lock[0] = 0;
}

static inline void lock(lock_t* lock)
{
    lock[0] = 1;
}

static inline void unlock(lock_t* lock)
{
    lock[0] = 0;
}

#else    // CONFIG_BAREMETAL_BUILD
#ifdef _MSC_VER
#include <stdbool.h>
#include <windows.h>

typedef CRITICAL_SECTION lock_t;

static inline void init_lock(lock_t* mutex)
{
    if (mutex != NULL)
    {
        InitializeCriticalSection(mutex);
    }
}

static inline void lock(lock_t* mutex)
{
    if (mutex != NULL)
    {
        EnterCriticalSection(mutex);
    }
}

static inline void unlock(lock_t* mutex)
{
    if (mutex != NULL)
    {
        LeaveCriticalSection(mutex);
    }
}
#else    //_MSC_VER
#include <pthread.h>

typedef pthread_mutex_t lock_t;

static inline void init_lock(lock_t* lock)
{
    pthread_mutex_init(lock, NULL);
}

static inline void lock(lock_t* lock)
{
    pthread_mutex_lock(lock);
}

static inline void unlock(lock_t* lock)
{
    pthread_mutex_unlock(lock);
}
#endif    //_MSC_VER
#endif    // CONFIG_BAREMETAL_BUILD

#endif
