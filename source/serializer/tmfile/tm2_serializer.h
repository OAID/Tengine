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
 * Author: haitao@openailab.com
 */

#pragma once

#include "tm2_format.h"

#define NULL_TM2_OP_LOADER ((tm2_op_loader_t)0x1)

struct node;
struct graph;

struct tm2_priv
{
    int fd; /* for file load */
    int mem_len;
    const char* base;             /* mem base for model */
    const TM2_Header* header;     /* file header */
    const TM2_Model* model;       /* model header */
    const TM2_Subgraph* subgraph; /* subgraph */
};

typedef int (*tm2_op_loader_t)(struct graph*, struct node*, const TM2_Node*, const TM2_Operator* tm_op);

typedef int (*tm2_map_t)(int);
