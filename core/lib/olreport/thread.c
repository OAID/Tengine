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

#include "thread.h"
#include <stdlib.h>

void init_thread_context(THREAD_CONTEXT* ctx)
{
    ctx->handle_ = 0;
    ctx->runner_ = 0;
}

void free_thread_context(THREAD_CONTEXT* ctx)
{
    ctx->handle_ = 0;
    ctx->runner_ = 0;
}

void start_thread(void* ctx)
{
    THREAD_CONTEXT* context = (THREAD_CONTEXT*)ctx;
    pthread_attr_t type;
    pthread_attr_init(&type);
    pthread_attr_setdetachstate(&type, PTHREAD_CREATE_JOINABLE);
    pthread_create(&(context->handle_ ),&type,context->runner_,ctx);
}

void stop_thread(void* ctx)
{
    THREAD_CONTEXT* context = (THREAD_CONTEXT*)ctx;
    if( context->handle_ == 0 )
    {
        return ;
    }

    pthread_detach( context->handle_ );
    //pthread_join(context->handle_,0);    
    context->handle_ = 0;
}

void join_thread(THREAD_CONTEXT* ctx)
{
    THREAD_CONTEXT* context = (THREAD_CONTEXT*)ctx;
    if( context->handle_ == 0 )
    {
        return ;
    }

    pthread_join(context->handle_,0); 
}

void init_thread_mutex(THREAD_MUTEX* mutex)
{
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE_NP);
	pthread_mutex_init(&(mutex->mutex_), &attr);
}

void free_thread_mutex(THREAD_MUTEX* mutex)
{
    pthread_mutex_destroy( &(mutex->mutex_) );
}

void lock(THREAD_MUTEX* mutex)
{
    pthread_mutex_lock( &(mutex->mutex_) );
}

void unlock(THREAD_MUTEX* mutex)
{
    pthread_mutex_unlock( &(mutex->mutex_) );
}
