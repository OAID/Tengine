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
 * Copyright (c) 2021, Open AI Lab
 * Author: hhchen@openailab.com
 */

#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>


#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#if (!defined(__ANDROID__) && defined(__aarch64__)) || defined(__QNX__)
#define ENABLE_DLA_API 1
#endif

#define CHECK(status)                                                       \
    do                                                                      \
    {                                                                       \
        auto ret = (status);                                                \
        if (ret != 0)                                                       \
        {                                                                   \
            Log(Loglevel, "TensorRT Engine",  "Cuda failure: %d", ret);     \
            abort();                                                        \
        }                                                                   \
    } while (0)


constexpr long double operator"" _GiB(long double val)
{
    return val * (1 << 30);
}
constexpr long double operator"" _MiB(long double val) { return val * (1 << 20); }
constexpr long double operator"" _KiB(long double val) { return val * (1 << 10); }

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(long long unsigned int val) { return val * (1 << 30); }
constexpr long long int operator"" _MiB(long long unsigned int val) { return val * (1 << 20); }
constexpr long long int operator"" _KiB(long long unsigned int val) { return val * (1 << 10); }
