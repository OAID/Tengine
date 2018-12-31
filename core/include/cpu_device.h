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
#ifndef __CPU_DEVICE_H__
#define __CPU_DEVICE_H__

#ifdef __cplusplus
extern "C" {
#endif

// cpu model list
#define CPU_GENERIC 0
#define CPU_A72 1
#define CPU_A53 2
#define CPU_A17 3
#define CPU_A7 4
#define CPU_A55 5
#define CPU_KRYO 6
#define CPU_A73 7

#define ARCH_GENERIC 0
#define ARCH_ARM_V8 1
#define ARCH_ARM_V7 2
#define ARCH_ARM_V8_2 3

#define MAX_CLUSTER_CPU_NUMBER 4

struct cpu_cluster
{
    int cpu_number;
    int max_freq;
    int cpu_model;
    int cpu_arch;
    int l1_size;
    int l2_size;
    int hw_cpu_id[MAX_CLUSTER_CPU_NUMBER];
};

struct cpu_info
{
    const char* cpu_name;
    const char* board_name;    // optional
    int cluster_number;
    int l3_size;
    struct cpu_cluster* cluster;
    int online_cpu_number;
    int* online_cpu_list;
};

const struct cpu_info* get_predefined_cpu(const char* cpu_name);

int create_cpu_device(const char* dev_name, const struct cpu_info* cpu_info);

const struct cpu_info* get_cpu_info(const char* dev_name);

void set_online_cpu(struct cpu_info* cpu_info, const int* cpu_list, int cpu_number);

#define set_working_cpu(cpu_list, cpu_number) set_online_cpu(NULL, cpu_list, cpu_number);

#ifdef __cplusplus
}
#endif

#endif
