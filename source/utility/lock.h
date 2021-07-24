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

#pragma once

/*!
 * @struct abstract_mutex
 * @brief  Abstract mutex_t, platform independence
 */
typedef struct abstract_mutex
{
    void* locker;                                 //!< platform dependence mutex impl
    void (*init)(struct abstract_mutex* mutex);   //!< init this mutex
    void (*lock)(struct abstract_mutex* mutex);   //!< lock this mutex
    void (*unlock)(struct abstract_mutex* mutex); //!< unlock this mutex
    void (*free)(struct abstract_mutex* mutex);   //!< destroy this mutex
} mutex_t;

/*!
 * @brief Init a abstract mutex.
 *
 * @param [in]  mutex: the mutex_t.
 */
void init_mutex(mutex_t* mutex);

/*!
 * @brief Init a abstract mutex.
 *
 * @param [in]  mutex: the mutex_t.
 */
void lock_mutex(mutex_t* mutex);

/*!
 * @brief Init a abstract mutex.
 *
 * @param [in]  mutex: the mutex_t.
 */
void unlock_mutex(mutex_t* mutex);

/*!
 * @brief Init a abstract mutex.
 *
 * @param [in]  mutex: the mutex_t.
 */
void free_mutex(mutex_t* mutex);
