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
 * Author: haitao@openailab.com
 */

#ifndef __SYS_PORT_H__
#define __SYS_PORT_H__

#include <stdint.h>
#include <stdarg.h>
#include <stdlib.h>

#ifdef CONFIG_ARCH_CORTEX_M
char* strdup(const char*);
#else
#include <malloc.h>
#endif

#include "compiler.h"
#include "lock.h"

void* sys_malloc(size_t size);
void sys_free(void* ptr);
void* sys_realloc(void* ptr, size_t size);

#ifdef CONFIG_INTERN_ALLOCATOR

#define malloc buddy_malloc
#define free buddy_free
#define realloc buddy_realloc

void* buddy_malloc(size_t size);
void buddy_free(void* ptr);
void* buddy_realloc(void* ptr, size_t size);

/* insert mem block into buddy system,to be called by difference system*/
int insert_mem_block(void* ptr, size_t size);

void set_buddy_mem_status(int disabled);

#endif

#endif
