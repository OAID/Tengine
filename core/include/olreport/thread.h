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
 * Copyright (c) 2019, Open AI Lab
 * Author: jjzeng@openailab.com
 */

#ifndef __THREAD_H__
#define __THREAD_H__

#include <stdio.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* (*thread_run_t )(void* context);
typedef unsigned char BOOL;


typedef struct
{
    pthread_t handle_;
    thread_run_t runner_;
}THREAD_CONTEXT;

void init_thread_context(THREAD_CONTEXT* ctx);
void free_thread_context(THREAD_CONTEXT* ctx);
void start_thread(void* ctx);
void stop_thread(void* ctx);
void join_thread(THREAD_CONTEXT* ctx);

typedef struct
{
    pthread_mutex_t mutex_;
}THREAD_MUTEX;

void init_thread_mutex(THREAD_MUTEX* mutex);
void free_thread_mutex(THREAD_MUTEX* mutex);
void lock(THREAD_MUTEX* mutex);
void unlock(THREAD_MUTEX* mutex);

#ifdef __cplusplus
}
#endif

#endif
