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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#ifndef __COMMON_UTIL_HPP__
#define __COMMON_UTIL_HPP__

#include <algorithm>
#include <vector>
#include <utility>
#include <stdio.h>
#include <string.h>

#include "cpu_device.h"

namespace TEngine {

static inline bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
{
    return lhs.first > rhs.first;
}

static inline std::vector<int> Argmax(const std::vector<float>& v, int N)
{
    std::vector<std::pair<float, int>> pairs;
    for(size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for(int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

static inline void DumpFloat(const char* fname, float* data, int number)
{
    FILE* fp = fopen(fname, "w");

    for(int i = 0; i < number; i++)
    {
        if(i % 16 == 0)
        {
            fprintf(fp, "\n%d:", i);
        }
        fprintf(fp, " %.5f", data[i]);
    }

    fprintf(fp, "\n");

    fclose(fp);
}

static inline unsigned long get_cur_time(void)
{
    struct timespec tm;

    clock_gettime(CLOCK_MONOTONIC, &tm);

    return (tm.tv_sec * 1000000 + tm.tv_nsec / 1000);
}

static inline std::vector<int> parse_cpu_list(char* cpu_list_str)
{
    std::vector<int> cpu_list;

    char* p = strtok(cpu_list_str, ",");
    while(p)
    {
        int cpu_id = strtoul(p, NULL, 10);
        cpu_list.push_back(cpu_id);
        p = strtok(NULL, ",");
    }

    return cpu_list;
}

static inline void set_cpu_list(char* cpu_list_str)
{
    std::vector<int> cpu_list = parse_cpu_list(cpu_list_str);

    int* int_buf = cpu_list.data();

    set_working_cpu(int_buf, cpu_list.size());
}

}    // namespace TEngine

#endif
