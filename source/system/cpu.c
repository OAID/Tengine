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
 * Parts of the following code in this file refs to
 * https://github.com/Tencent/ncnn/blob/master/src/cpu.cpp
 * Tencent is pleased to support the open source community by making ncnn
 * available.
 *
 * Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 */

/*
 * Copyright (c) 2021, OPEN AI LAB
 * Author: lswang@openailab.com
 */

#include "cpu.h"

#include "api/c_api.h"

#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>

#ifndef _MSC_VER
#include <pthread.h>
#include <sys/syscall.h>
#include <sched.h>
#include <unistd.h>
#endif

#if __APPLE__
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/machine.h>
#define __APPLE_IOS__ 1
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

static size_t core_count = 0;

static size_t affinity_mask_all_cluster = 0;
static size_t affinity_mask_big_cluster = 0;
static size_t affinity_mask_medium_cluster = 0;
static size_t affinity_mask_little_cluster = 0;

int init_cpu_count()
{
    if (0 < core_count)
        return core_count;

#ifdef __ANDROID__
    {
        FILE* cpu_info = fopen("/proc/cpuinfo", "rb");
        if (!cpu_info)
            return -1;

        char buffer[1024];
        while (!feof(cpu_info))
        {
            char* s = fgets(buffer, 1024, cpu_info);
            if (!s)
                break;

            if (memcmp(buffer, "processor", 9) == 0)
                core_count++;
        }

        fclose(cpu_info);
    };
#elif __APPLE_IOS__
    {
        size_t len = sizeof(core_count);
        sysctlbyname("hw.ncpu", &core_count, &len, NULL, 0);
    };
#else
    {
#ifdef _OPENMP
        core_count = omp_get_max_threads();
#else
        core_count = 1;
#endif
    }
#endif

    // check count range
    if (core_count < 1)
        core_count = 1;
    // TODO: deal with this conditions
    if (core_count > sizeof(size_t) * 8)
        core_count = sizeof(size_t) * 8;

    return core_count;
}

#ifndef _MSC_VER
static int get_max_freq_khz(int cpuid)
{
    // first try, for all possible cpu
    char path[256];
    sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuid);

    FILE* fp = fopen(path, "rb");

    if (!fp)
    {
        // second try, for online cpu
        sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", cpuid);
        fp = fopen(path, "rb");

        if (fp)
        {
            int max_freq_khz = 0;
            while (!feof(fp))
            {
                int freq_khz = 0;
                int nscan = fscanf(fp, "%d %*d", &freq_khz);
                if (nscan != 1)
                    break;

                if (freq_khz > max_freq_khz)
                    max_freq_khz = freq_khz;
            }

            fclose(fp);

            if (max_freq_khz != 0)
                return max_freq_khz;

            fp = NULL;
        }

        if (!fp)
        {
            // third try, for online cpu
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuid);
            fp = fopen(path, "rb");

            if (!fp)
                return -1;

            int max_freq_khz = -1;
            int ret = fscanf(fp, "%d", &max_freq_khz);

            fclose(fp);

            if (max_freq_khz <= 0 && EOF == ret)
                return -1;
            else
                return max_freq_khz;
        }
    }

    int max_freq_khz = 0;
    while (!feof(fp))
    {
        int freq_khz = 0;
        int nscan = fscanf(fp, "%d %*d", &freq_khz);
        if (nscan != 1)
            break;

        if (freq_khz > max_freq_khz)
            max_freq_khz = freq_khz;
    }

    fclose(fp);

    return max_freq_khz;
}

static int set_sched_affinity(size_t thread_affinity_mask)
{
    // cpu_set_t definition
    // ref http://stackoverflow.com/questions/16319725/android-set-thread-affinity
#ifndef CPU_SETSIZE
#define CPU_SETSIZE 1024
#endif
#ifndef __NCPUBITS
#define __NCPUBITS (8 * sizeof(unsigned long))
#endif

    typedef struct
    {
        unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
    } cpu_set_t;

#define CPU_SET(cpu, cpusetp) ((cpusetp)->__bits[(cpu) / __NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))

#define CPU_ZERO(cpusetp) memset((cpusetp), 0, sizeof(cpu_set_t))

    // set affinity for threads
#if (defined __GLIBC__) || (defined _OHOS_) || (defined V831)
    pid_t pid = syscall(SYS_gettid);
#else
#ifdef PI3
    pid_t pid = getpid();
#else

#ifdef __APPLE__
    uint64_t tid64;
    pthread_threadid_np(NULL, &tid64);
    pid_t pid = (pid_t)tid64;
#else
    pid_t pid = gettid();
#endif
#endif
#endif
    cpu_set_t mask;
    CPU_ZERO(&mask);
    //    for (int i = 0; i < ( int )sizeof(size_t) * 8; i++)
    for (int i = 0; i < core_count; i++)
    {
        if (thread_affinity_mask & (1 << i))
            CPU_SET(i, &mask);
    }
#if __APPLE__
    int syscallret = syscall(set_sched_affinity, pid, sizeof(mask), &mask);
#else
    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
#endif

    if (syscallret)
    {
        fprintf(stderr, "syscall error %d\n", syscallret);
        return -1;
    }

    return 0;
}
#endif

int init_cluster_mask()
{
    init_cpu_count();

    if (0 != affinity_mask_all_cluster)
        return 0;

    affinity_mask_all_cluster = ((size_t)(1) << core_count) - (size_t)(1);
    //affinity_mask_all_cluster = (size_t)(0) - (size_t)(1);

#ifndef _MSC_VER
    int max_freq_min_val = INT_MAX;
    int max_freq_max_val = 0;

    // TODO: deal with very large count of cores
    int max_freq_array[sizeof(size_t) * 8];
    for (int i = 0; i < core_count; i++)
    {
        int max_freq_khz = get_max_freq_khz(i);
        // fprintf(stderr, "cpu %d, max_freq_khz %d\n", i, max_freq_khz);
        max_freq_array[i] = max_freq_khz;

        if (max_freq_khz > max_freq_max_val)
            max_freq_max_val = max_freq_khz;
        if (max_freq_khz < max_freq_min_val)
            max_freq_min_val = max_freq_khz;
    }

    if (max_freq_max_val == max_freq_min_val)
    {
        affinity_mask_big_cluster = affinity_mask_all_cluster;
        affinity_mask_medium_cluster = 0;
        affinity_mask_little_cluster = 0;
    }
    else
    {
        for (int i = 0; i < core_count; i++)
        {
            if (max_freq_array[i] == max_freq_max_val)
                affinity_mask_big_cluster |= (1 << i);
            else if (max_freq_array[i] == max_freq_min_val)
                affinity_mask_little_cluster |= (1 << i);
            else
                affinity_mask_medium_cluster |= (1 << i);
        }
    }
#else
    // TODO implement me for other platforms
    affinity_mask_big_cluster = affinity_mask_all_cluster;
#endif

    return 0;
}

int check_cpu()
{
    init_cpu_count();
    init_cluster_mask();

    return 0;
}

int get_cpu_mask_count(size_t mask)
{
    int count = 0;

    for (int i = 0; i < core_count; i++)
        if (mask & (1 << i))
            count++;

    return count;
}

int set_cpu_affine(size_t mask)
{
#if defined __ANDROID__ || defined __linux__
    int count = get_cpu_mask_count(mask);

#ifdef _OPENMP
    // set affinity for each threads
    omp_set_num_threads(count);

    int status[sizeof(size_t) * 8] = {0};
#pragma omp parallel for num_threads(count)
    for (int i = 0; i < count; i++)
    {
        status[i] = set_sched_affinity(mask);
    }

    for (int i = 0; i < count; i++)
    {
        if (status[i] != 0)
            return -1;
    }
#else
    int status = set_sched_affinity(mask);
    if (0 != status)
        return -1;
#endif

#elif __APPLE_IOS__ || _MSC_VER
    // threads affinity not supported on ios
    (void)mask;
    return -1;
#else
    int status = set_sched_affinity(mask);
    if (0 != status) return -1;

    return 0;
#endif

    return 0;
}

size_t get_cpu_cluster_mask(int cluster)
{
    switch (cluster)
    {
    case TENGINE_CLUSTER_BIG:
        if (0 != affinity_mask_big_cluster)
            return affinity_mask_big_cluster;
        break;
    case TENGINE_CLUSTER_MEDIUM:
        if (0 != affinity_mask_medium_cluster)
            return affinity_mask_medium_cluster;
        break;
    case TENGINE_CLUSTER_LITTLE:
        if (0 != affinity_mask_little_cluster)
            return affinity_mask_little_cluster;
        break;
    default:
        break;
    }

    return affinity_mask_all_cluster;
}
