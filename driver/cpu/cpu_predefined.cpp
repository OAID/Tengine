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
 * Copyright (c) 2018, Open AI Lab
 * Author: haitao@openailab.com
 */
#include <string.h>

#include "cpu_device.h"

static struct cpu_cluster rk3399_cluster[2] = {{4, 1410, CPU_A53, ARCH_ARM_V8, 32 << 10, 512 << 10, {0, 1, 2, 3}},
                                               {2, 1800, CPU_A72, ARCH_ARM_V8, 32 << 10, 1024 << 10, {4, 5}}};

static struct cpu_info rk3399 = {"rk3399", "firefly", 2, 0, rk3399_cluster, -1, NULL};

static struct cpu_cluster a63_cluster = {4, 18000, CPU_A53, ARCH_ARM_V8, 32 << 10, 512 << 10, {0, 1, 2, 3}};

static struct cpu_info a63 = {"a63", "pad", 1, 0, &a63_cluster, -1, NULL};

static struct cpu_cluster rk3288_cluster = {4, 18000, CPU_A17, ARCH_ARM_V7, 32 << 10, 512 << 10, {0, 1, 2, 3}};

static struct cpu_info rk3288 = {"rk3288", "thinkerboard", 1, 0, &rk3288_cluster, -1, NULL};

static struct cpu_cluster r40_cluster = {4, 12000, CPU_A7, ARCH_ARM_V7, 32 << 10, 512 << 10, {0, 1, 2, 3}};

static struct cpu_info r40 = {"r40", "bananapi", 1, 0, &r40_cluster, -1, NULL};

static struct cpu_cluster hikey_cluster[2] = {{4, 15000, CPU_A53, ARCH_ARM_V8, 32 << 10, 512 << 10, {0, 1, 2, 3}},

                                              {4, 15000, CPU_A73, ARCH_ARM_V8, 32 << 10, 1024 << 10, {4, 5, 6, 7}}

};

static struct cpu_info hikey960 = {"kirin960", "hikey960", 2, 0, hikey_cluster, -1, NULL};

static struct cpu_cluster apq_cluster = {4, 20000, CPU_A72, ARCH_ARM_V8, 32 << 10, 1024 << 10, {0, 1, 2, 3}};

static struct cpu_info apq8096 = {"apq8096", "apq8096", 1, 0, &apq_cluster, -1, NULL};

static const struct cpu_info* g_prefined_table[] = {&rk3399, &a63, &rk3288, &r40, &hikey960, &apq8096};

const struct cpu_info* get_predefined_cpu(const char* cpu_name)
{
    int table_size = sizeof(g_prefined_table) / sizeof(struct cpu_info*);

    for(int i = 0; i < table_size; i++)
    {
        const struct cpu_info* p_info = g_prefined_table[i];
        if(!strcmp(p_info->cpu_name, cpu_name))
            return p_info;
    }

    return nullptr;
}
