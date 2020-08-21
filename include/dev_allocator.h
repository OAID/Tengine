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

#ifndef __DEV_ALLOCATOR_H__
#define __DEV_ALLOCATOR_H__

struct ir_graph;

struct dev_allocator
{
    char* name;
    int (*allocate)(struct dev_allocator*, struct ir_graph*);
    int (*release)(struct dev_allocator*);
};

int init_allocator_registry(struct dev_allocator* allocator);

int release_allocator_registry();

struct dev_allocator* get_default_dev_allocator(void);

struct dev_allocator* get_dev_allocator(const char* dev_name);

#define REGISTER_DEV_ALLOCATOR(func_name) static void(func_name)(void) __attribute__((constructor))

#endif
