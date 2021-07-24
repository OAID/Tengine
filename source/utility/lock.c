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
 * Author: lswang@openailab.com
 */

#include "utility/lock.h"

#include "defines.h"
#include "utility/sys_port.h"

static inline void bare_metal_mutex_init(mutex_t* mutex)
{
    mutex->locker = sys_malloc(sizeof(mutex->locker));
    *((int*)(mutex->locker)) = 0;
}

static inline void bare_metal_mutex_lock(mutex_t* mutex)
{
    *((int*)(mutex->locker)) = 1;
}

static inline void bare_metal_mutex_unlock(mutex_t* mutex)
{
    *((int*)(mutex->locker)) = 0;
}

static inline void bare_metal_mutex_free(mutex_t* mutex)
{
    if (NULL != mutex->locker)
    {
        sys_free(mutex->locker);
    }

    mutex->locker = NULL;
}

// for WIN MSVC

#ifdef TENGINE_HAS_LIB_POSIX_THREAD
#include <pthread.h>

typedef pthread_mutex_t lock_t;

static inline void posix_thread_mutex_init(mutex_t* mutex)
{
    mutex->locker = sys_malloc(sizeof(lock_t));
    pthread_mutex_init((lock_t*)mutex->locker, NULL);
}

static inline void posix_thread_mutex_lock(mutex_t* mutex)
{
    pthread_mutex_lock((lock_t*)mutex->locker);
}

static inline void posix_thread_mutex_unlock(mutex_t* mutex)
{
    pthread_mutex_unlock((lock_t*)mutex->locker);
}

static inline void posix_thread_mutex_free(mutex_t* mutex)
{
    return bare_metal_mutex_free(mutex);
}

void init_mutex(mutex_t* mutex)
{
    mutex->init = posix_thread_mutex_init;
    mutex->lock = posix_thread_mutex_lock;
    mutex->unlock = posix_thread_mutex_unlock;
    mutex->free = posix_thread_mutex_free;

    return mutex->init(mutex);
}
#elif (defined _WIN32 && !(defined __MINGW32__))
#include <windows.h>

typedef CRITICAL_SECTION lock_t;

static inline void win_mutex_init(mutex_t* mutex)
{
    mutex->locker = sys_malloc(sizeof(lock_t));
    InitializeCriticalSection((lock_t*)mutex->locker);
}

static inline void win_mutex_lock(mutex_t* mutex)
{
    if (NULL != mutex->locker)
    {
        EnterCriticalSection((lock_t*)mutex->locker);
    }
}

static inline void win_mutex_unlock(mutex_t* mutex)
{
    if (NULL != mutex->locker)
    {
        LeaveCriticalSection((lock_t*)mutex->locker);
    }
}

static inline void win_mutex_free(mutex_t* mutex)
{
    return bare_metal_mutex_free(mutex);
}

void init_mutex(mutex_t* mutex)
{
    mutex->init = win_mutex_init;
    mutex->lock = win_mutex_lock;
    mutex->unlock = win_mutex_unlock;
    mutex->free = win_mutex_free;

    return mutex->init(mutex);
}
#else
void init_mutex(mutex_t* mutex)
{
    mutex->init = bare_metal_mutex_init;
    mutex->lock = bare_metal_mutex_lock;
    mutex->unlock = bare_metal_mutex_unlock;
    mutex->free = bare_metal_mutex_free;

    return mutex->init(mutex);
}
#endif // end TENGINE_HAS_LIB_POSIX_THREAD

void lock_mutex(mutex_t* mutex)
{
    return mutex->lock(mutex);
}

void unlock_mutex(mutex_t* mutex)
{
    return mutex->unlock(mutex);
}

void free_mutex(mutex_t* mutex)
{
    return mutex->free(mutex);
}
