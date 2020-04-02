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
#include <iostream>
#include <algorithm>
#include <cstdio>

#include "prof_record.hpp"
#include "prof_utils.hpp"

namespace TEngine {

const ProfTime::TimeRecord* ProfTime::GetRecord(int idx) const
{
    return &record[idx];
}
int ProfTime::GetRecordNum(void) const
{
    return record.size();
}

bool ProfTime::Start(int idx, void* ident)
{
    TimeRecord& r = record[idx];

    r.start_time = get_cur_time();
    r.ident = ident;

    return true;
}

bool ProfTime::Stop(int idx)
{
    TimeRecord& r = record[idx];

    r.end_time = get_cur_time();

    uint64_t used_time = r.end_time - r.start_time;

    if(used_time > r.max_time)
        r.max_time = used_time;
    if(used_time < r.min_time)
        r.min_time = used_time;

    r.total_used_time += used_time;
    r.count++;

    return true;
}

void ProfTime::Reset(void)
{
    int num = record.size();

    for(int i = 0; i < num; i++)
    {
        TimeRecord& tr = record[i];
        tr.Reset();
    }
}

void ProfTime::Dump(int method)
{
    if(method == PROF_DUMP_DECREASE)
    {
        std::sort(record.begin(), record.end(), [](const TimeRecord& a, const TimeRecord& b) {
            if(a.total_used_time > b.total_used_time)
                return true;
            else
                return false;
        });
    }
    else if(method == PROF_DUMP_INCREASE)
    {
        std::sort(record.begin(), record.end(), [](const TimeRecord& a, const TimeRecord& b) {
            if(a.total_used_time > b.total_used_time)
                return false;
            else
                return true;
        });
    }

    uint64_t accum_time = 0;

    for(unsigned int i = 0; i < record.size(); i++)
        accum_time += record[i].total_used_time;

    int forward_count = 1;
    int idx = 0;
    for(unsigned int i = 0; i < record.size(); i++)
    {
        const TimeRecord& r = record[i];

        if(r.count == 0)
            continue;

        forward_count = r.count;

        std::printf("%3d [ %3.2f%% : %.3f ms ]", idx, 100.0 * r.total_used_time / accum_time,
                    ( float )(( unsigned long )r.total_used_time / 1000.f / r.count));

        parser(r.ident, r.count, r.total_used_time);
        idx += 1;
        std::cout << "\n";
    }

    if(accum_time > 0)
    {
        std::printf("\ntotal accumulated time: %lu us. roughly [%lu] us per run\n", ( unsigned long )accum_time,
                    ( unsigned long )(accum_time / forward_count));
    }
}

}    // namespace TEngine
