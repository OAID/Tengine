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
#include <string>

#include "operator.hpp"
#include "attribute.hpp"
#include "serializer.hpp"

namespace TEngine {

template class SpecificFactory<Serializer>;
// template SpecificFactory<Serializer> SpecificFactory<Serializer>::instance;
template SerializerManager SimpleObjectManagerWithLock<SerializerManager, SerializerPtr>::instance;

static Attribute op_method_load_map;
static Attribute op_method_save_map;

void RegisterOpLoadMethod(const std::string& op_name, const std::string& method_name, any func)
{
    std::string key = op_name + method_name;
    op_method_load_map[key] = func;
}

void RegisterOpSaveMethod(const std::string& op_name, const std::string& method_name, any func)
{
    std::string key = op_name + method_name;
    op_method_save_map[key] = func;
}

bool FindOpLoadMethod(const std::string& op_name, const std::string& method_name)
{
    std::string key = op_name + method_name;

    if(op_method_load_map.ExistAttr(key))
        return true;

    return false;
}

any& GetOpLoadMethod(const std::string& op_name, const std::string& method_name)
{
    std::string key = op_name + method_name;

    return op_method_load_map[key];
}

bool FindOpSaveMethod(const std::string& op_name, const std::string& method_name)
{
    std::string key = op_name + method_name;

    if(op_method_save_map.ExistAttr(key))
        return true;

    return false;
}

any& GetOpSaveMethod(const std::string& op_name, const std::string& method_name)
{
    std::string key = op_name + method_name;

    return op_method_save_map[key];
}

}    // namespace TEngine
