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
#ifndef __TENGINE_PLUGIN_HPP__
#define __TENGINE_PLUGIN_HPP__

#include <string>
#include <vector>
#include <unordered_map>
#include <map>

#include "attribute.hpp"
#include "logger.hpp"
#include "share_lib_parser.hpp"

namespace TEngine {

using module_init_func_t=int(*)(void);
using module_release_func_t=int(*)(void);

/*
 In a config file, we use the following format to set the fullname and the
 init program name of a plugin:  (XXX is the short name of the plugin)
 plugin.XXX.so = fullname of the plugin
 plugin.XXX.init = init program name of the plugin
 eg.
     plugin.operator.so = ./build/operator/liboperator.so
     plugin.operator.init = tengine_plugin_init
*/
struct TEnginePlugin
{
    using NameManager = Attribute;
    static NameManager *GetNameManager(void);

    using InitManager = Attribute;
    static InitManager *GetInitManager(void);

    using PrioManager=std::map<int,std::string>;
    static PrioManager * GetPrioManager(void);

    using HandlerManager = Attribute;
    static HandlerManager *GetHandlerManager(void);
    
    // Set the PluginManager corresponding to the content of config file
    static void SetPluginManager(void);

    // Store a config key and a config value into the PluginManager
    static void SetPlugin(const std::string& key, const std::string& value);

    // Load and initiate all the plugins
    static bool LoadAll(void);

    // Load a single plugin corresponding to the short name
    static bool LoadPlugin(const std::string& name);

    // Init a single plugin corresponding to the short name
    static bool InitPlugin(const std::string& name);

    // Get the fullname of the plugin corresponding to the short name
    static bool GetFullName(const std::string& name, std::string& fullname);

    // Get the init program name of the plugin corresponding to the short name
    static bool GetInitName(const std::string& name, std::string& initname);

    // Get the handler of the plugin corresponding to the short name
    static bool GetHandler(const std::string& name, ShareLibParser& handler);

    // Dump all the plugin information as the following format:
    // short name : full name : init program name
    static void DumpPlugin(void);

    static std::unordered_map<int,module_init_func_t> * GetInitTable(void);
    static std::unordered_map<int,module_release_func_t> * GetReleaseTable(void);

    //Register Initialization functions that should be executed after all plugins are loaded
    //Priority: the lower one is higher priority
    static void RegisterModuleInit(int priority, module_init_func_t init_func);

    //Note: Release is executed on reverse priority
    static void RegisterModuleRelease(int priority, module_release_func_t rel_func);
    static void InitModule(void);
    static void ReleaseModule(void);

private:
    TEnginePlugin()=default;
    TEnginePlugin(const TEnginePlugin&)=delete;
    TEnginePlugin(TEnginePlugin&&)=delete;
}; // end of struct TEnginePlugin

} //end of namespace TEngine

#endif
