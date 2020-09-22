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

#ifdef CONFIG_BAREMETAL_BUILD

#include "sys_port.h"
#include "module.h"
#include "tengine_log.h"
#include "cpu_probe.h"

static struct cpu_entry cpu0 = {.cpu_id = 0, .cluster_id = 0};

static struct cluster_entry cluster0 = {
    .id = 0,
    .leader_cpu = 0,
    .cpu_model = ARCH_GENERIC,
    .cpu_arch = ARCH_GENERIC,
    .cpu_num = 1,
    .l1_size = 1024,
    .l2_size = 16 * 1024,
    .max_freq = 200,
};

static struct probed_cpu_info probed_cpu_info = {
    .cpu_num = 1,
    .cluster_num = 1,
    .cpu_list = &cpu0,
    .cluster_list = &cluster0,
};

struct probed_cpu_info* get_probed_cpu_info(void)
{
    return &probed_cpu_info;
}

#else
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <assert.h>

#include "sys_port.h"
#include "module.h"
#include "tengine_log.h"
#include "cpu_probe.h"

static struct probed_cpu_info* probed_cpu_info = NULL;

struct probed_cpu_info* get_probed_cpu_info(void)
{
    return probed_cpu_info;
}

struct cpu_item
{
    int cpu_id;
    int max_freq;
    int cluster_leader;
    int cluster_id;
};

/*
   for the meaning files in /sys/device/system/cpu/cpu0/cpufreq
   please read documentation/cpu-freq/user-guide.txt
*/

int get_cpu_items(struct cpu_item** p_item)
{
    char cpu_path[128];
    char file_path[128];
    struct cpu_item* cpu_item = NULL;
    struct stat stat_buf;
    int i = 0;

    while (1)
    {
        FILE* fp;
        int ret;

        sprintf(cpu_path, "/sys/devices/system/cpu/cpu%d/cpufreq", i);

        if (stat(cpu_path, &stat_buf) < 0)
            break;

        cpu_item = ( struct cpu_item* )sys_realloc(cpu_item, sizeof(struct cpu_item) * (i + 1));
        cpu_item[i].cpu_id = i;

        ret = snprintf(file_path, 128, "%s/cpuinfo_max_freq", cpu_path);

        if (ret >= 128)
            file_path[127] = 0x0;

        fp = fopen(file_path, "rb");

        if (fp == NULL)
            break;

        if (fscanf(fp, "%d", &cpu_item[i].max_freq) < 0)
        {
            fclose(fp);
            break;
        }

        fclose(fp);

        ret = snprintf(file_path, 128, "%s/related_cpus", cpu_path);

        if (ret >= 128)
            file_path[127] = 0x0;

        fp = fopen(file_path, "rb");

        if (fp == NULL)
            break;

        if (fscanf(fp, "%d ", &cpu_item[i].cluster_leader) < 0)
        {
            fclose(fp);
            break;
        }

        fclose(fp);

        i++;
    }

    if (i == 0)
    {
        /*
         some weird thing happened! just fill a fake one
         TODO: add a log here
        */

        cpu_item = ( struct cpu_item* )sys_malloc(sizeof(struct cpu_item));

        cpu_item[0].cpu_id = 0;
        cpu_item[0].max_freq = 100;
        cpu_item[0].cluster_leader = 0;

        i++;
    }

    *p_item = cpu_item;

    return i;
}

#ifdef __ARM_ARCH

static char* get_target_line(FILE* fp, const char* target_prefix)
{
    static char line[256];

    while (fgets(line, 256, fp))
    {
        if (!memcmp(line, target_prefix, strlen(target_prefix)))
            return line;
    }

    return NULL;
}

static int get_cpu_model_arch(int id, struct cluster_entry* cluster)
{
    char cpu_fname[256];
    FILE* fp;
    char* line;
    int cur_id = 0;

    /* set the pre-set default info, in case of failure */

    cluster->l1_size = 32 << 10;
    cluster->l2_size = 512 << 10;

#if __ARM_ARCH >= 8
    cluster->cpu_arch = ARCH_ARM_V8;
    cluster->cpu_model = CPU_A53;
#else
    cluster->cpu_arch = ARCH_ARM_V7;
    cluster->cpu_model = CPU_A7;
#endif

    sprintf(cpu_fname, "/proc/cpuinfo");

    fp = fopen(cpu_fname, "r");

    if (!fp)
        return 0;

    while (get_target_line(fp, "processor"))
    {
        if (cur_id == id)
            break;

        cur_id++;
    }

    if (cur_id != id)
    {
        fclose(fp);
        return 0;
    }

    /*
        processor       : 4
        BogoMIPS        : 48.00
        Features        : fp asimd evtstrm aes pmull sha1 sha2 crc32
        CPU implementer : 0x41
        CPU architecture: 8
        CPU variant     : 0x0
        CPU part        : 0xd08
        CPU revision    : 2
    */

    /*
        line=get_target_line(fp,"CPU architecture");

        if(!line)
        {
            fclose(fp);
            return 0;
        }

        char * p=line;

        while(*p++!=':');

        //arch
        int cpu_arch=strtoul(p,NULL,10);

        if(cpu_arch==8)
            cluster->cpu_arch=ARCH_ARM_V8;
        else if(cpu_arch==7)
            cluster->cpu_arch=ARCH_ARM_V7;
    */
    line = get_target_line(fp, "CPU part");

    if (!line)
    {
        fclose(fp);
        return 0;
    }

    char* p = line;

    while (*p++ != ':')
        ;

    int cpu_part = strtoul(p, NULL, 16);

    if (cpu_part == 0xd08 && cluster->cpu_arch == ARCH_ARM_V8)
    {
        cluster->cpu_model = CPU_A72;

        cluster->l2_size = 1024 << 10;
    }
    if ((cpu_part == 0xc0d || cpu_part == 0xc0e) && cluster->cpu_arch == ARCH_ARM_V7)
        cluster->cpu_model = CPU_A17;

    fclose(fp);

    return 0;
}

#else

static int get_cpu_model_arch(int id, struct cluster_entry* cluster)
{
    cluster->cpu_model = CPU_GENERIC;
    cluster->cpu_arch = CPU_GENERIC;
    cluster->l1_size = 32 << 10;
    cluster->l2_size = 512 << 10;

    return 0;
}

#endif

static int probe_system_cpu(void* arg)
{
    struct cpu_item* cpu_item;
    int cpu_number;
    int cluster_number = 1;

    cpu_number = get_cpu_items(&cpu_item);

    assert(cpu_number >= 1);

    cpu_item[0].cluster_id = 0;

    /* assuming cluster cpus are continuous */
    for (int i = 1; i < cpu_number; i++)
    {
        if (cpu_item[i - 1].cluster_leader != cpu_item[i].cluster_leader)
        {
            cluster_number++;
        }

        cpu_item[i].cluster_id = cluster_number - 1;
    }

    /* allocate memory */
    probed_cpu_info = ( struct probed_cpu_info* )sys_malloc(sizeof(struct probed_cpu_info));

    if (probed_cpu_info == NULL)
        return -1;

    probed_cpu_info->cpu_list = ( struct cpu_entry* )sys_malloc(sizeof(struct cpu_entry) * cpu_number);

    if (probed_cpu_info->cpu_list == NULL)
        return -1;

    probed_cpu_info->cluster_list = ( struct cluster_entry* )sys_malloc(sizeof(struct cluster_entry) * cluster_number);

    if (probed_cpu_info->cluster_list == NULL)
        return -1;

    probed_cpu_info->cpu_num = cpu_number;
    probed_cpu_info->cluster_num = cluster_number;

    for (int i = 0; i < cluster_number; i++)
    {
        struct cluster_entry* cluster = &probed_cpu_info->cluster_list[i];
        cluster->id = 0;
        cluster->cpu_num = 0;
        cluster->cpu_model = -1;
        cluster->cpu_arch = -1;
        cluster->l1_size = -1;
        cluster->l2_size = -1;
        cluster->max_freq = -1;
        cluster->leader_cpu = -1;
    }

    for (int i = 0; i < cpu_number; i++)
    {
        struct cpu_entry* cpu_entry = probed_cpu_info->cpu_list + i;

        cpu_entry->cpu_id = cpu_item[i].cpu_id;
        cpu_entry->cluster_id = cpu_item[i].cluster_id;

        struct cluster_entry* cluster = probed_cpu_info->cluster_list + cpu_entry->cluster_id;

        cluster->cpu_num++;

        if (cluster->leader_cpu < 0)
        {
            cluster->leader_cpu = cpu_entry->cpu_id;
            cluster->max_freq = cpu_item[i].max_freq;
        }
    }

    sys_free(cpu_item);

    for (int i = 0; i < cluster_number; i++)
    {
        struct cluster_entry* cluster = probed_cpu_info->cluster_list + i;

        get_cpu_model_arch(cluster->leader_cpu, cluster);
    }

    return 0;
}

static int free_probed_cpu_info(void* arg)
{
    if (probed_cpu_info == NULL)
        return 0;

    sys_free(probed_cpu_info->cluster_list);
    sys_free(probed_cpu_info->cpu_list);
    sys_free(probed_cpu_info);

    return 0;
}

REGISTER_CRIT_MODULE_INIT(MOD_CORE_LEVEL, "probe_system_cpu", probe_system_cpu);
REGISTER_MODULE_EXIT(MOD_CORE_LEVEL, "free_probed_cpu_info", free_probed_cpu_info);

#endif
