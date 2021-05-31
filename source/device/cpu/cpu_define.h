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

#include <stddef.h>

#define OPS_SCORE_STATIC            10000
#define OPS_SCORE_BEST              8000
#define OPS_SCORE_PREFER            6000
#define OPS_SCORE_CANDO             4000
#define OPS_SCORE_NOTSUP            2000

#define MEM_POOL_ALLOCATED          8
#define INPLACE_BLOCK_FLAG          0x40

#define CPU_DEVICE_NAME             "CPU"

#define TENGINE_DUMP_DIR            "TG_DEBUG_DUMP_DIR"
#define TENGINE_DUMP_LAYER          "TG_DEBUG_DATA"
#define TENGINE_PRINT_LAYER_COST    "TG_DEBUG_TIME"
#define TENGINE_FORCE_USE_REF_OP    "TG_DEBUG_REF"


typedef struct cpu_option
{
    char*   dev_name;
    int     num_thread;     //!< how many threads to run
    int     cluster;        //!< cpu cluster
    int     precision;      //!< precision of calculation
    size_t  affinity;       //!< affinity of cpu core, max 64 cpus
} cpu_opt_t;
