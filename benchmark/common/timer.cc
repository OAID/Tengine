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

#include "timer.hpp"

#ifdef _WINDOWS
#include <Windows.h>
#else // _WINDOWS
#include <sys/time.h>
#endif // _WINDOWS


double Timer::get_current_time() const
{
#ifdef _WINDOWS
    LARGE_INTEGER freq;
        LARGE_INTEGER pc;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&pc);

        return pc.QuadPart * 1000.0 / freq.QuadPart;
#else // _WINDOWS
    struct timeval tv;
    gettimeofday(&tv, nullptr);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
#endif
}


Timer::Timer()
{
    start_time_ = get_current_time();
    end_time_ = start_time_;
}


void Timer::Start()
{
    start_time_ = get_current_time();
    end_time_ = start_time_;
}


void Timer::Stop()
{
    end_time_ = get_current_time();
}


float Timer::TimeCost()
{
    if (end_time_ <= start_time_)
    {
        Stop();
    }

    return (float)(end_time_ - start_time_);
}
