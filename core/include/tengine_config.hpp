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
 * Author: jingyou@openailab.com
 */
#ifndef __TENGINE_CONFIG_HPP__
#define __TENGINE_CONFIG_HPP__

#include <string>
#include <fstream>
#include <vector>

#include "attribute.hpp"
#include "parameter.hpp"
#include "logger.hpp"

#define TENGINE_MT_SUPPORT

namespace TEngine {

template <typename T>
bool ConvertSpecialAny(T& entry, const std::type_info & info, any& data);


struct TEngineConfig
{
    static bool tengine_mt_mode;  // multithread mode
    static char delim_ch;  // separator between key and value, default is '='
    static char commt_ch;  // character followed by comments, default is '#'

    static const std::string version;  // TEngine version 

    using ConfManager = Attribute;
    static ConfManager *GetConfManager(void);

    // Load a config file
    static bool Load(const std::string filename, const char delim_ch = '=', 
                                                 const char commt_ch = '#');

    // Get the value corresponding to the key
    template<typename T>static bool Get(const std::string& key, T& value, bool show_warning=false);

    // Set the key and the value
    template<typename T>static bool Set(const std::string& key, const T& value, 
                                        const bool create=true);

    // Remove the item corresponding to the key
    static void Remove(const std::string& key);

    // Dump the contents of the config file
    static void DumpConfig(void);
  
    // Remove the leading and trailing whitespaces of a string
    static void Trim(std::string& s);

    // Parse the string of a key (seperator is '.')
    // eg.   Input  string : "xyz.123.abc"
    //       Output vector : ("xyz", "123", "abc")
    static std::vector<std::string> ParseKey(const std::string& key);

private:
    TEngineConfig()=default;
    TEngineConfig(const TEngineConfig&)=delete;
    TEngineConfig(TEngineConfig&&)=delete;
}; // end of struct TEngineConfig

template <typename T> 
bool TEngineConfig::Get(const std::string& key, T& value, bool show_warning)
{
    ConfManager *manager = GetConfManager();
    if(!manager->ExistAttr(key))
    {
        if(show_warning)
            LOG_ERROR()<<"The key is not existed in the config file!\n";

        return false;
    }

    any& data = manager->GetAttr(key);
    if(typeid(T)==data.type())
    {
        value=any_cast<T>(data);
        return true;
    }
    if(!ConvertSpecialAny(value, data.type(), data))
    {
        LOG_ERROR()<<"Type mismatch on config item: "<<key<<"\n";
        return false;
    }
    return true;
}

template <typename T>
bool TEngineConfig::Set(const std::string& key, const T& value, const bool create)
{
    ConfManager *manager = GetConfManager();

    if(!create  && !manager->ExistAttr(key))
    {
        return false;
    }

    manager->SetAttr(key, value);

    return true;
}

#ifdef TENGINE_MT_SUPPORT
static inline bool GetTEngineMTMode(void)
{
    return TEngineConfig::tengine_mt_mode;
}
#else
static inline bool GetTEngineMTMode(void)
{
    return false;
}
#endif

/* if the graph should be run in sync mode ? */
bool GetSyncRunMode(void);

} //end of namespace TEngine

#endif
