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
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "cpu_device.h"

struct cpu_item
{
    int cpu_id;
    int max_freq;
    int cluster_leader;
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

    while(1)
    {
        FILE* fp;
        int ret;

        sprintf(cpu_path, "/sys/devices/system/cpu/cpu%d/cpufreq", i);

        if(stat(cpu_path, &stat_buf) < 0)
            break;

        cpu_item = ( struct cpu_item* )realloc(cpu_item, sizeof(struct cpu_item) * (i + 1));

        cpu_item[i].cpu_id = i;

        ret = snprintf(file_path, 128, "%s/cpuinfo_max_freq", cpu_path);

        if(ret >= 128)
            file_path[127] = 0x0;

        fp = fopen(file_path, "rb");

        if(fp == NULL)
            break;

        if(fscanf(fp, "%d", &cpu_item[i].max_freq) < 0)
        {
            fclose(fp);
            break;
        }

        fclose(fp);

        ret = snprintf(file_path, 128, "%s/related_cpus", cpu_path);

        if(ret >= 128)
            file_path[127] = 0x0;

        fp = fopen(file_path, "rb");

        if(fp == NULL)
            break;

        if(fscanf(fp, "%d ", &cpu_item[i].cluster_leader) < 0)
        {
            fclose(fp);
            break;
        }

        fclose(fp);

        i++;
    }

    if(i == 0)
    {
        FILE* fp = fopen("/proc/cpuinfo", "rb");
        if(fp != NULL)
        {
            char buf[1024];
            while(fgets(buf, 1024, fp))
            {
                if(memcmp(buf, "processor", 9) == 0)
                {
                    cpu_item = ( struct cpu_item* )realloc(cpu_item, sizeof(struct cpu_item) * (i + 1));

                    cpu_item[i].cpu_id = i;
                    cpu_item[i].max_freq = 100;
                    cpu_item[i].cluster_leader = i;
                    ++i;
                }
            }
            fclose(fp);
        }
    }

    if(i == 0)
    {
        /*
         some weird thing happened! just fill a fake one
         TODO: add a log here
        */
        cpu_item = ( struct cpu_item* )malloc(sizeof(struct cpu_item) * 4);

        cpu_item[0].cpu_id = 0;
        cpu_item[0].max_freq = 100;
        cpu_item[0].cluster_leader = 0;
        i++;
        cpu_item[1].cpu_id = 1;
        cpu_item[1].max_freq = 100;
        cpu_item[1].cluster_leader = 1;
        i++;
        cpu_item[2].cpu_id = 2;
        cpu_item[2].max_freq = 100;
        cpu_item[2].cluster_leader = 2;
        i++;
        cpu_item[3].cpu_id = 3;
        cpu_item[3].max_freq = 100;
        cpu_item[3].cluster_leader = 3;
        i++;
    }

    *p_item = cpu_item;

    return i;
}

#ifdef __ARM_ARCH

static char* get_target_line(FILE* fp, const char* target_prefix)
{
    static char line[256];

    while(fgets(line, 256, fp))
    {
        if(!memcmp(line, target_prefix, strlen(target_prefix)))
            return line;
    }

    return nullptr;
}

static int get_cpu_model_arch(int id, struct cpu_cluster* cluster)
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

    if(!fp)
        return 0;

    while(get_target_line(fp, "processor"))
    {
        if(cur_id == id)
            break;

        cur_id++;
    }

    if(cur_id != id)
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

    if(!line)
    {
        fclose(fp);
        return 0;
    }

    char* p = line;

    while(*p++ != ':')
        ;

    int cpu_part = strtoul(p, NULL, 16);

    if(cpu_part == 0xd08 && cluster->cpu_arch == ARCH_ARM_V8)
    {
        cluster->cpu_model = CPU_A72;

        cluster->l2_size = 1024 << 10;
    }
    if((cpu_part == 0xc0d || cpu_part == 0xc0e) && cluster->cpu_arch == ARCH_ARM_V7)
        cluster->cpu_model = CPU_A17;

    fclose(fp);

    return 0;
}

#else

static int get_cpu_model_arch(int id, struct cpu_cluster* cluster)
{
    cluster->cpu_model = CPU_GENERIC;
    cluster->cpu_arch = CPU_GENERIC;
    cluster->l1_size = 32 << 10;
    cluster->l2_size = 512 << 10;

    return 0;
}

#endif
struct cpu_info* probe_system_cpu(void)
{
    static struct cpu_info cpu_dev;

    struct cpu_item* cpu_item;
    int cpu_number;
    int cluster_number = 1;

    cpu_number = get_cpu_items(&cpu_item);

    /* assuming cluster cpus are continuous */
    for(int i = 1; i < cpu_number; i++)
    {
        if(cpu_item[i - 1].cluster_leader != cpu_item[i].cluster_leader)
            cluster_number++;
    }

    struct cpu_cluster* cpu_cluster = ( struct cpu_cluster* )malloc(sizeof(struct cpu_cluster) * cluster_number);

    memset(cpu_cluster->hw_cpu_id, -1, sizeof(int) * MAX_CLUSTER_CPU_NUMBER);

    /* setup cpu 0 */
    cpu_cluster[0].cpu_number = 1;
    cpu_cluster[0].max_freq = cpu_item[0].max_freq;
    cpu_cluster[0].hw_cpu_id[0] = cpu_item[0].cpu_id;

    int top_max_freq = 0;
    struct cpu_cluster* cluster = cpu_cluster;

    for(int i = 1; i < cpu_number; i++)
    {
        /* assuming cluster's cpu is continuous*/

        if(cpu_item[i - 1].cluster_leader != cpu_item[i].cluster_leader)
        {
            cluster++;
            memset(cluster->hw_cpu_id, -1, sizeof(int) * MAX_CLUSTER_CPU_NUMBER);
            cluster->cpu_number = 0;
            cluster->max_freq = cpu_item[i].max_freq;

            if(cluster->max_freq > top_max_freq)
                top_max_freq = cluster->max_freq;
        }

        cluster->hw_cpu_id[cluster->cpu_number] = cpu_item[i].cpu_id;
        cluster->cpu_number++;
    }

    free(cpu_item);

    for(int i = 0; i < cluster_number; i++)
    {
        struct cpu_cluster* cluster = cpu_cluster + i;

        get_cpu_model_arch(cluster->hw_cpu_id[0], cluster);
    }

    cpu_dev.cluster_number = cluster_number;
    cpu_dev.cluster = cpu_cluster;

    cpu_dev.online_cpu_list = ( int* )malloc(sizeof(int) * cpu_number);

    int online_cpu_number = 0;

    for(int i = 0; i < cpu_dev.cluster_number; i++)
    {
        struct cpu_cluster* cluster = cpu_cluster + i;

        for(int j = 0; j < cluster->cpu_number; j++)
        {
            if(cluster->max_freq >= top_max_freq)
            {
                cpu_dev.online_cpu_list[online_cpu_number++] = cluster->hw_cpu_id[j];
            }
        }
    }

#ifdef __ARM_ARCH
    cpu_dev.cpu_name = "arm.probed";
#else
    cpu_dev.cpu_name = "x86.probed";
#endif

    cpu_dev.board_name = "generic.probed";
    cpu_dev.online_cpu_number = online_cpu_number;

    return &cpu_dev;
}

void free_probe_cpu_info(struct cpu_info* cpu_dev)
{
    free(cpu_dev->online_cpu_list);
    free(cpu_dev->cluster);
}
