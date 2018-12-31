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
#include <malloc.h>
#include <string.h>

#include "cpu_device.h"

int get_cpu_number(void)
{
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    int num = 0;
    char buf[256];

    if(fp == NULL)
        return 1;

    while(fgets(buf, 256, fp))
    {
        if(memcmp(buf, "processor", 9) == 0)
            num++;
    }

    fclose(fp);

    if(num < 1)
        num = 1;

    return num;
}

#ifdef __ARM_ARCH

#ifdef __ANDROID__
int get_cpu_max_freq(int id)
{
    char fname[256];
    int max_freq = 100;
    FILE* fp = NULL;

    sprintf(fname, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", id);

    fp = fopen(fname, "rb");

    if(!fp)
    {
        sprintf(fname, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", id);
        fp = fopen(fname, "rb");
    }

    if(fp)
    {
        while(!feof(fp))
        {
            int freq;
            if(fscanf(fp, "%d %*d\n", &freq) != 1)
                break;

            if(freq > max_freq)
                max_freq = freq;
        }

        fclose(fp);

        return max_freq;
    }

    sprintf(fname, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", id);
    fp = fopen(fname, "rb");

    if(fp)
    {
        fscanf(fp, "%d", &max_freq);
        fclose(fp);
    }

    return max_freq;
}

#else
int get_cpu_max_freq(int id)
{
    char cpu_fname[256];
    FILE* fp;
    int max_freq;

    sprintf(cpu_fname, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", id);

    fp = fopen(cpu_fname, "r");

    if(!fp)
        return 0;

    if(fscanf(fp, "%d", &max_freq) < 0)
        return 0;

    fclose(fp);

    return max_freq;
}
#endif

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

int get_cpu_model_arch(int id, struct cpu_cluster* cluster)
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

struct cpu_info* probe_system_cpu(void)
{
    static struct cpu_info cpu_dev;

    int cluster_idx = -1;
    int last_max_freq = -1;
    int top_max_freq = -1;

    int cpu_number = get_cpu_number();
    struct cpu_cluster* cpu_cluster = ( struct cpu_cluster* )malloc(sizeof(struct cpu_cluster) * (cpu_number / 4 + 1));

    for(int i = 0; i < cpu_number; i++)
    {
        int max_freq = 0;
        struct cpu_cluster* cluster;

        max_freq = get_cpu_max_freq(i);

        if(max_freq != last_max_freq)
        {
            cluster_idx++;
            cluster = cpu_cluster + cluster_idx;
            cluster->cpu_number = 0;
        }
        else
            cluster = cpu_cluster + cluster_idx;

        cluster->max_freq = max_freq;
        cluster->cpu_number++;

        last_max_freq = max_freq;

        if(top_max_freq < max_freq)
            top_max_freq = max_freq;

        if(get_cpu_model_arch(i, cluster) < 0)
            return NULL;
    }

    int start_cpu = 0;

    cpu_dev.cluster_number = cluster_idx + 1;
    cpu_dev.cluster = cpu_cluster;

    // setup the online cpu according to top_max_freq
    cpu_dev.online_cpu_list = ( int* )malloc(sizeof(int) * cpu_number);
    int online_cpu_number = 0;

    for(int i = 0; i < cpu_dev.cluster_number; i++)
    {
        struct cpu_cluster* cluster = cpu_cluster + i;

        for(int j = 0; j < cluster->cpu_number; j++)
        {
            cluster->hw_cpu_id[j] = start_cpu + j;

            if(cluster->max_freq == top_max_freq)
            {
                cpu_dev.online_cpu_list[online_cpu_number++] = cluster->hw_cpu_id[j];
            }
        }

        start_cpu += cluster->cpu_number;
    }

    cpu_dev.cpu_name = "arm.probed";
    cpu_dev.board_name = "generic.probed";

    cpu_dev.online_cpu_number = online_cpu_number;

    return &cpu_dev;
}

#else

struct cpu_info* probe_system_cpu(void)
{
    /* create cpu_info */
    static struct cpu_info cpu_dev;

    int cpu_number = get_cpu_number();

    struct cpu_cluster* cpu_cluster = ( struct cpu_cluster* )malloc(sizeof(struct cpu_cluster) * (cpu_number / 4 + 1));

    int cluster_number = 0;

    for(int i = 0; i < cpu_number; i += 4)
    {
        struct cpu_cluster* cluster = cpu_cluster + cluster_number;
        int start_cpu_id = cluster_number * 4;

        cluster->cpu_number = start_cpu_id + 4 > cpu_number ? cpu_number - start_cpu_id : 4;
        cluster->max_freq = 2000;
        cluster->cpu_model = CPU_GENERIC;
        cluster->cpu_arch = CPU_GENERIC;
        cluster->l1_size = 32 << 10;
        cluster->l2_size = 512 << 10;

        for(int j = 0; j < cluster->cpu_number; j++)
            cluster->hw_cpu_id[j] = start_cpu_id + j;

        cluster_number++;
    }

    int online_cpu_number = cpu_number;

    cpu_dev.cpu_name = "geneirc chip";
    cpu_dev.board_name = "generic board";

    cpu_dev.cluster_number = 1;
    cpu_dev.l3_size = 512 << 10;

    cpu_dev.online_cpu_number = online_cpu_number;
    cpu_dev.online_cpu_list = ( int* )malloc(sizeof(int) * online_cpu_number);

    for(int i = 0; i < cpu_number; i++)
    {
        cpu_dev.online_cpu_list[i] = i;
    }

    cpu_dev.cluster_number = cluster_number;
    cpu_dev.cluster = cpu_cluster;

    return &cpu_dev;
}

#endif

void free_probe_cpu_info(struct cpu_info* cpu_dev)
{
    free(cpu_dev->online_cpu_list);
    free(cpu_dev->cluster);
}
