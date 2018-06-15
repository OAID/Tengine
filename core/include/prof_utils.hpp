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
#ifndef __PROF_UTILS_HPP__
#define __PROF_UTILS_HPP__

#include <time.h>

namespace TEngine {

#if 1
static inline unsigned long get_cur_time(void)
{
   struct timespec tm;

   clock_gettime(CLOCK_MONOTONIC, &tm);

   return (tm.tv_sec*1000000+tm.tv_nsec/1000);
}
#else

#include <sys/time.h>

static inline unsigned long get_cur_time(void)
{
   struct timeval tv;

   gettimeofday(&tv,NULL);

   return (tv.tv_sec*1000000+tv.tv_usec);
}

#endif


}


#endif
