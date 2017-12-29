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
 * Author: honggui@openailab.com
 */
#pragma once
#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <cstdlib>
//For Debug
//获取类型易读的名称
namespace TEngine {
    template <class T>
    static std::string type_name()
    {
            typedef typename std::remove_reference<T>::type TR;
            std::unique_ptr<char, void(*)(void*)> own
                    (
    #ifndef __GNUC__
    nullptr,
    #else
    abi::__cxa_demangle(typeid(TR).name(), nullptr,
                    nullptr, nullptr),
    #endif
                    std::free
                    );
            std::string r = own != nullptr ? own.get() : typeid(TR).name();
            if (std::is_const<TR>::value)
                    r += " const";
            if (std::is_volatile<TR>::value)
                    r += " volatile";
            if (std::is_lvalue_reference<T>::value)
                    r += "&";
            else if (std::is_rvalue_reference<T>::value)
                    r += "&&";
            return r;
    }

    template <typename T>
    static void OutputTypeName(T&& t)
    {
            std::cout << type_name<T>() << std::endl;
    }

    template <typename T>
    static void OutputTypeNameToCerr(T&& t)
    {
            std::cerr << type_name<T>() << std::endl;
    }

    template <typename T>
    static std::string GetNameForType(T&& t)
    {
            return "org:" + type_name<T>() + "  -- decay:" + type_name<typename std::decay<T>::type>();
    }

    static std::string GetTypeName(const char* name)
    {
        #ifndef __GNUC__
            return name;
        #else
            std::unique_ptr<char, void(*)(void*)> own
            (
                abi::__cxa_demangle(name, nullptr,
                        nullptr, nullptr),
                std::free
            );
            std::string r = own.get();        
            return r;
        #endif
    }

    static inline void OutputTypeName(const char* name)
    {
        std::cout << "=== " << GetTypeName(name) << std::endl;
    }
    
}
