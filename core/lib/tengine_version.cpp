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
#include <string.h>
#include <string>
#include <iostream>

#include "tengine_errno.hpp"
#include "tengine_version.hpp"
#include "logger.hpp"

namespace TEngine {

#define TENGINE_VERSION "1.12.0"

#ifdef CONFIG_VERSION_POSTFIX
const std::string tengine_version(TENGINE_VERSION "-" CONFIG_VERSION_POSTFIX);
#else
const std::string tengine_version(TENGINE_VERSION);
#endif

static inline int get_version_major(const char* ver)
{
    return strtoul(ver, NULL, 10);
}

static inline int get_version_minor(const char* ver)
{
    const char* p = strchr(ver, '.');

    if(!p)
        return 0;

    p++;

    return strtoul(p, NULL, 10);
}

static inline int get_version_fix(const char* ver)
{
    const char* p = strchr(ver, '.');

    if(!p)
        return 0;
    p++;

    p = strchr(p, '.');

    if(!p)
        return 0;

    p++;

    return strtoul(p, NULL, 10);
}

static char* remove_version_postfix(const char* ver)
{
    char* tmp_str = strdup(ver);

    return strtok(tmp_str, "-");
}

static unsigned int get_version_int(const char* ver)
{
    char* clear_ver = remove_version_postfix(ver);

    int major = get_version_major(clear_ver) & 0xff;
    int minor = get_version_minor(clear_ver) & 0xff;
    int fix = get_version_fix(clear_ver) & 0xff;

    free(clear_ver);

    return (major << 16) | (minor << 8) | fix;
}

static bool version_compatible(int cur_ver, int req_ver)
{
    return true;
}

static bool check_version(const char* req_version)
{
    int cur_ver = get_version_int(tengine_version.c_str());
    int req_ver = get_version_int(req_version);

    if(!version_compatible(cur_ver, req_ver))
    {
        LOG_ERROR() << "requested version: " << req_version << " is not supported by"
                    << " this library with version: " << tengine_version << "\n";
        set_tengine_errno(EPROTONOSUPPORT);
        return false;
    }

    return true;
}

}    // namespace TEngine

using namespace TEngine;

const char* get_tengine_version(void)
{
    return tengine_version.c_str();
}

int request_tengine_version(const char* version)
{
    if(check_version(version))
        return 1;

    return 0;
}
