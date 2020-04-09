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
#include <iostream>

#include "tengine_c_api.h"

#include "data_type.hpp"
#include "logger.hpp"

namespace TEngine {

namespace DataType {

struct TypeInfo
{
    int type;
    int size;
    const char* name;
};

static TypeInfo type_info[] = {{TENGINE_DT_FP32, 4, "float32"}, {TENGINE_DT_FP16, 2, "float16"},
                               {TENGINE_DT_INT8, 1, "int8"},    {TENGINE_DT_UINT8, 1, "uint8"},
                               {TENGINE_DT_INT32, 4, "int"},    {TENGINE_DT_INT16, 4, "int16"}};

static TypeInfo* FindType(int data_type)
{
    for(unsigned int i = 0; i < sizeof(type_info) / sizeof(TypeInfo); i++)
    {
        TypeInfo* p_info = &type_info[i];

        if(p_info->type == data_type)
            return p_info;
    }

    return nullptr;
}

int GetTypeSize(int data_type)
{
    TypeInfo* p_info = FindType(data_type);

    if(p_info)
        return p_info->size;

    XLOG_ERROR() << "Unknow data type: " << data_type << "\n";

    return 0;
}

const char* GetTypeName(int data_type)
{
    TypeInfo* p_info = FindType(data_type);

    if(p_info)
        return p_info->name;

    XLOG_ERROR() << "Unknow data type: " << data_type << "\n";

    return nullptr;
}

int GetTypeID(const char* name)
{
    for(unsigned int i = 0; i < sizeof(type_info) / sizeof(TypeInfo); i++)
    {
        TypeInfo* p_info = &type_info[i];

        if(!strcmp(p_info->name, name))
            return i;
    }

    return -1;
}

}    // namespace DataType

}    // namespace TEngine
