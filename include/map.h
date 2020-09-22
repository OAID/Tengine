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

#ifndef __MAP_H__
#define __MAP_H__

struct map;

struct map* create_map(const char* name, void (*free_data)(void*));

void release_map(struct map* m);

int insert_map_data(struct map* m, const char* key, void* data);

int replace_map_data(struct map* m, const char* key, void* data);

void* get_map_data(struct map* m, const char* key);

int remove_map_data(struct map* m, const char* key);

int get_map_num(struct map* m);

#endif
