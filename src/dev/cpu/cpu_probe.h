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

#ifndef __CPU_PROBE_H__
#define __CPU_PROBE_H__

#include "cpu_model.h"

struct cpu_entry
{
    int cpu_id;
    int cluster_id;
};

struct cluster_entry
{
    int id;
    int leader_cpu;
    int cpu_model;
    int cpu_arch;
    int cpu_num;
    int l1_size;
    int l2_size;
    int max_freq;
};

struct probed_cpu_info
{
    int cpu_num;
    int cluster_num;

    struct cpu_entry* cpu_list;
    struct cluster_entry* cluster_list;
};

struct probed_cpu_info* get_probed_cpu_info(void);

#endif
