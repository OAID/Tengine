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
#ifndef __PARAMETER_HPP__
#define __PARAMETER_HPP__

#include <functional>
#include <unordered_map>

#include "any.hpp"

namespace TEngine {

struct NamedParam
{
    using item_cpy_t = void (*)(void*, const void*);
    using item_set_any = void (*)(void*, const any&);

    struct ItemInfo
    {
        item_cpy_t cpy_func;
        item_set_any cpy_any;
        const char* type_name;
        int data;
    };

    ItemInfo* FindItem(const std::string& name, const char* type_name)
    {
        if(item_map_.count(name) == 0)
            return nullptr;

        ItemInfo& entry = item_map_.at(name);

        // skip type checking if type_name is nullptr
        if(type_name && entry.type_name && strcmp(type_name, entry.type_name))
        {
            // printf("requested: %s recorded:%s\n",item_type->name(),entry.type_info->name());
            return nullptr;
        }

        return &entry;
    }

    bool GetItemVal(const std::string& name, const char* type_name, void* val)
    {
        ItemInfo* entry = FindItem(name, type_name);

        if(entry == nullptr)
            return false;

        entry->cpy_func(val, ( char* )this + entry->data);

        return true;
    }

    bool SetItemVal(const std::string& name, const char* type_name, const void* val)
    {
        ItemInfo* entry = FindItem(name, type_name);

        if(entry == nullptr)
            return false;

        entry->cpy_func(( char* )this + entry->data, val);

        return true;
    }

    bool SetItemCompatibleAny(const std::string& name, const any& n)
    {
        if(item_map_.count(name) == 0)
            return false;

        ItemInfo& entry = item_map_.at(name);
        const char* item_type = entry.type_name;
        const char* any_type = n.type().name();

        /* several special cases */
        if(!strcmp(item_type, typeid(const char*).name()) && !strcmp(any_type, typeid(std::string).name()))
        {
            const char** ptr = ( const char** )(( char* )this + entry.data);
            const std::string& str = any_cast<std::string>(n);

            ptr[0] = str.c_str();    // unsafe, since any may be destroyed soon

            return true;
        }

        if(!strcmp(item_type, typeid(std::string).name()) && !strcmp(any_type, typeid(const char*).name()))
        {
            std::string* p_str = ( std::string* )(( char* )this + entry.data);
            const char* ptr = any_cast<const char*>(n);

            *p_str = ptr;

            return true;
        }

        return false;
    }

    bool SetItemFromAny(const std::string& name, const any& n)
    {
        ItemInfo* entry = FindItem(name, n.type().name());

        if(entry == nullptr)
            return SetItemCompatibleAny(name, n);

        entry->cpy_any(( char* )this + entry->data, n);

        return true;
    }

    const std::unordered_map<std::string, ItemInfo>& GetItemMap(void)
    {
        return item_map_;
    }

protected:
    std::unordered_map<std::string, ItemInfo> item_map_;
};

#define DECLARE_PARSER_STRUCTURE(s) s(void)

#define DECLARE_PARSER_ENTRY(e)                                                          \
    {                                                                                    \
        typedef decltype(e) T;                                                           \
        ItemInfo info;                                                                   \
        info.type_name = typeid(T).name();                                               \
        info.data = ( char* )&e - ( char* )this;                                         \
        info.cpy_func = [](void* data, const void* v) { *( T* )data = *( const T* )v; }; \
        info.cpy_any = [](void* data, const any& n) { *( T* )data = any_cast<T>(n); };   \
        item_map_[#e] = info;                                                            \
    }

}    // namespace TEngine

#endif
