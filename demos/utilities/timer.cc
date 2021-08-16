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


Timer::Timer()
{
    Start();
}


void Timer::Start()
{
    Stop();
    this->start_time = this->end_time;
}


void Timer::Stop()
{
#ifdef _MSC_VER
    this->end_time = std::chrono::system_clock::now();
#else
    this->end_time = std::chrono::high_resolution_clock::now();
#endif
}


float Timer::Cost()
{
    if (this->end_time <= this->start_time)
    {
        this->Stop();
    }

    return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(this->end_time - this->start_time).count()) / 1000.f;
}
