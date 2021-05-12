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
 * Parts of the following code in this file refs to
 * https://github.com/Tencent/ncnn/tree/master/src/layer/vulkan/
 * Tencent is pleased to support the open source community by making ncnn
 * available.
 *
 * Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 */

/*
 * Copyright (c) 2020, Open AI Lab
 * Author: ddzhao@openailab.com
 */

#ifndef VULKAN_PLATFORM_HPP
#define VULKAN_PLATFORM_HPP

#include <pthread.h>

namespace TEngine {

class Mutex
{
public:
    Mutex() { pthread_mutex_init(&mutex, 0); }
    ~Mutex() { pthread_mutex_destroy(&mutex); }
    void lock() { pthread_mutex_lock(&mutex); }
    void unlock() { pthread_mutex_unlock(&mutex); }
private:
    friend class ConditionVariable;
    pthread_mutex_t mutex;
};

class MutexLockGuard
{
public:
    MutexLockGuard(Mutex& _mutex) : mutex(_mutex) { mutex.lock(); }
    ~MutexLockGuard() { mutex.unlock(); }
private:
    Mutex& mutex;
};

class ConditionVariable
{
public:
    ConditionVariable() { pthread_cond_init(&cond, 0); }
    ~ConditionVariable() { pthread_cond_destroy(&cond); }
    void wait(Mutex& mutex) { pthread_cond_wait(&cond, &mutex.mutex); }
    void broadcast() { pthread_cond_broadcast(&cond); }
    void signal() { pthread_cond_signal(&cond); }
private:
    pthread_cond_t cond;
};

class Thread
{
public:
    Thread(void* (*start)(void*), void* args = 0) { pthread_create(&t, 0, start, args); }
    ~Thread() {}
    void join() { pthread_join(t, 0); }
private:
    pthread_t t;
};

} // namespace TEngine

#endif