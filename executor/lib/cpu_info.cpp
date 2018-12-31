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

#include "cpu_device.h"
#include "cpu_info.hpp"

namespace TEngine {

/* MUST be the same order in cpu_device.h */

static const char* cpu_arch_table[] = {
    "generic",
    "arm64",
    "arm32",
    "armv8.2",
};

static const char* cpu_model_table[] = {
    "generic", "A72", "A53", "A17", "A7", "A55", "Kyro",
};

const char* CPUInfo::GetCPUArchString(int cpu_id) const
{
    int cpu_arch = GetCPUArch(cpu_id);

    if(cpu_arch < 0)
        return nullptr;

    return cpu_arch_table[cpu_arch];
}

const char* CPUInfo::GetCPUModelString(int cpu_id) const
{
    int cpu_model = GetCPUModel(cpu_id);

    if(cpu_model < 0)
        return nullptr;

    return cpu_model_table[cpu_model];
}
}    // namespace TEngine
