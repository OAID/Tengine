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
#ifndef __TENGINE_LOCK_HPP__
#define __TENGINE_LOCK_HPP__

#include <mutex>

#include "tengine_config.hpp"

namespace TEngine {

#ifdef TENGINE_MT_SUPPORT

static inline void TEngineLock(std::mutex& mutex)
{
   if(GetTEngineMTMode())
        mutex.lock();
}

static inline void TEngineUnlock(std::mutex& mutex)
{
   if(GetTEngineMTMode())
        mutex.unlock();
}

#else

static inline void TEngineLock(std::mutex& mutex) {};
static inline void TEngineUnlock(std::mutex& mutex){};

#endif


} //namespace TEngine


#endif
