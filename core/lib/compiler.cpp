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
#include <cstdio>
#include <cstdlib>
#include <string>

#include "compiler.hpp"

#ifdef ANDROID

namespace std {
template <> std::string to_string<int>(int n)
{
    char buf[128];
    snprintf(buf, 127, "%d", n);
    buf[127] = 0x0;

    return std::string(buf);
}
}    // namespace std

#endif

#ifdef STATIC_BUILD

extern "C" void __pthread_cond_broadcast(void);
extern "C" void __pthread_cond_destroy(void);
extern "C" void __pthread_cond_signal(void);
extern "C" void __pthread_cond_wait(void);

void static_compiling_workaround(void)
{
    __pthread_cond_broadcast();
    __pthread_cond_destroy();
    __pthread_cond_signal();
    __pthread_cond_wait();
}

#endif
