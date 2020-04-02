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

#include <memory>

#include "tengine_plugin.hpp"
#include "tengine_config.hpp"
#include "tengine_c_api.h"
#include "tengine_errno.hpp"

namespace TEngine {

using NameManager = Attribute;
using InitManager = Attribute;
using HandlerManager = Attribute;
using ConfManager = Attribute;
using ShareLibParserPtr = std::shared_ptr<ShareLibParser>;

NameManager* TEnginePlugin::GetNameManager(void)
{
    static NameManager instance;

    return &instance;
}

TEnginePlugin::PrioManager* TEnginePlugin::GetPrioManager(void)
{
    static PrioManager instance;
    return &instance;
}

InitManager* TEnginePlugin::GetInitManager(void)
{
    static InitManager instance;

    return &instance;
}

HandlerManager* TEnginePlugin::GetHandlerManager(void)
{
    static HandlerManager instance;

    return &instance;
}

void TEnginePlugin::SetPluginManager(void)
{
    ConfManager* cm = TEngineConfig::GetConfManager();
    std::vector<std::string> keys;
    std::string value;

    // Get the list of keys
    keys = cm->ListAttr();
    for(unsigned int i = 0; i < keys.size(); i++)
    {
        value = "";
        if(!TEngineConfig::Get<std::string>(keys.at(i), value))
            continue;
        // Set the plugin manager
        SetPlugin(keys.at(i), value);
    }
}

void TEnginePlugin::SetPlugin(const std::string& key, const std::string& value)
{
    NameManager* nm = GetNameManager();
    InitManager* im = GetInitManager();
    PrioManager* pm = GetPrioManager();

    std::vector<std::string> subkey;

    subkey = TEngineConfig::ParseKey(key);
    if(subkey.size() > 2 && subkey.at(0) == "plugin")
    {
        if(subkey.at(2) == "so")
        {
            nm->SetAttr(subkey.at(1), value);
        }
        else if(subkey.at(2) == "init")
        {
            im->SetAttr(subkey.at(1), value);
        }
        else if(subkey.at(2) == "prio")
        {
            int prio = strtoul(value.c_str(), NULL, 10);
            (*pm)[prio] = subkey.at(1);
        }
    }
}

bool TEnginePlugin::LoadAll(void)
{
    NameManager* nm = GetNameManager();
    std::vector<std::string> names;
    names = nm->ListAttr();

    PrioManager* pm = GetPrioManager();

    auto ir = pm->begin();

    while(ir != pm->end())
    {
        const std::string& key = ir->second;

        if(!LoadPlugin(key))
            return false;

        if(!InitPlugin(key))
            return false;

        ir++;
    }

    return true;
}

bool TEnginePlugin::LoadPlugin(const std::string& name)
{
    HandlerManager* hm = GetHandlerManager();

    std::string fullname;
    ShareLibParser* p = new ShareLibParser();

    if(!GetFullName(name, fullname))
        return false;

    if(!fullname.size() || p->Load(fullname) < 0)
    {
        LOG_ERROR() << "Failed in loading the plugin!\n";
        return false;
    }

    hm->SetAttr(name, ShareLibParserPtr(p));
    LOG_INFO() << "Successfully load plugin : " << fullname << "\n";
    return true;
}

bool TEnginePlugin::InitPlugin(const std::string& name)
{
    std::string initname;
    ShareLibParser p;

    if(!GetInitName(name, initname))
        return false;

    if(!initname.size() || !GetHandler(name, p))
        return false;

    p.ExecuteFunc<int()>(initname);
    return true;
}

bool TEnginePlugin::GetFullName(const std::string& name, std::string& fullname)
{
    NameManager* nm = GetNameManager();
    if(!nm->ExistAttr(name))
    {
        LOG_ERROR() << "The plugin is not set in the config file!\n";
        return false;
    }

    nm->GetAttr<std::string>(name, &fullname);

    if(!fullname.size())
    {
        LOG_ERROR() << "The fullname of the plugin is not set in the config file!\n";
        return false;
    }
    return true;
}

bool TEnginePlugin::GetInitName(const std::string& name, std::string& initname)
{
    InitManager* im = GetInitManager();
    if(!im->ExistAttr(name))
    {
        LOG_ERROR() << "The plugin is not set in the config file!\n";
        return false;
    }

    im->GetAttr<std::string>(name, &initname);

    if(!initname.size())
    {
        LOG_ERROR() << "The initname of the plugin is not set in the config file!\n";
        return false;
    }
    return true;
}

bool TEnginePlugin::GetHandler(const std::string& name, ShareLibParser& handler)
{
    HandlerManager* hm = GetHandlerManager();
    if(!hm->ExistAttr(name))
    {
        LOG_ERROR() << "The handler is not got yet!\n";
        return false;
    }

    handler = *any_cast<ShareLibParserPtr>(hm->GetAttr(name)).get();
    return true;
}

void TEnginePlugin::DumpPlugin(void)
{
    NameManager* nm = GetNameManager();
    InitManager* im = GetInitManager();
    std::vector<std::string> names;

    names = nm->ListAttr();
    for(unsigned int i = 0; i < names.size(); i++)
    {
        std::string fullname, initname;
        nm->GetAttr(names.at(i), &fullname);
        im->GetAttr(names.at(i), &initname);
        std::cout << names.at(i) << " : " << fullname << " : " << initname << std::endl;
    }
}

std::unordered_map<int, module_init_func_t>* TEnginePlugin::GetInitTable(void)
{
    static std::unordered_map<int, module_init_func_t> instance;

    return &instance;
}

void TEnginePlugin::RegisterModuleInit(int priority, module_init_func_t init_func)
{
    std::unordered_map<int, module_init_func_t>* p_table = GetInitTable();

    if(p_table->count(priority))
    {
        LOG_WARN() << "init priority " << priority << " has been registered already\n";
        return;
    }

    (*p_table)[priority] = init_func;
}

std::unordered_map<int, module_release_func_t>* TEnginePlugin::GetReleaseTable(void)
{
    static std::unordered_map<int, module_release_func_t> instance;

    return &instance;
}

void TEnginePlugin::RegisterModuleRelease(int priority, module_release_func_t rel_func)
{
    std::unordered_map<int, module_release_func_t>* p_table = GetReleaseTable();

    (*p_table)[priority] = rel_func;
}

int TEnginePlugin::InitModule(void)
{
    std::unordered_map<int, module_init_func_t>* p_table = GetInitTable();
    auto ir = p_table->begin();
    auto end = p_table->end();

    while(ir != end)
    {
        module_init_func_t func = ir->second;

        if(func() < 0)
            return -1;

        ir++;
    }

    return 0;
}

void TEnginePlugin::ReleaseModule(void)
{
    std::unordered_map<int, module_release_func_t>* p_table = GetReleaseTable();
    auto ir = p_table->begin();
    auto end = p_table->end();

    std::vector<module_release_func_t> func_list;

    while(ir != end)
    {
        func_list.push_back(ir->second);
        ir++;
    }

    auto r_start = func_list.rbegin();
    auto r_end = func_list.rend();

    while(r_start != r_end)
    {
        (*r_start)();
        r_start++;
    }
}

}    // end of namespace TEngine

namespace TEngine {

struct PluginInfo
{
    std::string plugin_name;
    std::string fname;
    ShareLibParserPtr handle;
};
}    // namespace TEngine

using namespace TEngine;

using PluginInfoPtr = std::shared_ptr<PluginInfo>;

using plugin_table_t = std::vector<PluginInfoPtr>;

static plugin_table_t g_plugin_table;

int load_tengine_plugin(const char* plugin_name, const char* fname, const char* init_func_name)
{
    if(plugin_name == nullptr || fname == nullptr)
        return -1;

    ShareLibParserPtr handle(new ShareLibParser());

    try
    {
        if(handle->Load(fname) < 0)
            return -1;

        if(init_func_name)
        {
            if(handle->ExecuteFunc<int()>(init_func_name) < 0)
                return -1;
        }
    }

    catch(const std::exception& e)
    {
        LOG_ERROR() << e.what() << "\n";
        set_tengine_errno(EFAULT);
        return -1;
    }

    PluginInfo* p_info = new PluginInfo();

    p_info->plugin_name = plugin_name;
    p_info->fname = fname;
    p_info->handle = handle;

    g_plugin_table.emplace_back(p_info);

    return 0;
}

int unload_tengine_plugin(const char* plugin_name, const char* rel_func_name)
{
    if(plugin_name == nullptr)
        return -1;

    auto ir = g_plugin_table.begin();
    auto ir_end = g_plugin_table.end();

    while(ir != ir_end)
    {
        auto& ptr = *ir;

        if(ptr->plugin_name == std::string(plugin_name))
            break;
    }

    if(ir == ir_end)
        return -1;

    int ret = 0;

    if(rel_func_name)
        ret = (*ir)->handle->ExecuteFunc<int()>(rel_func_name);

    g_plugin_table.erase(ir);

    return ret;
}

int get_tengine_plugin_number(void)
{
    return g_plugin_table.size();
}

const char* get_tengine_plugin_name(int idx)
{
    int plugin_number = g_plugin_table.size();

    if(idx >= plugin_number || idx < 0)
        return nullptr;

    auto& ptr = g_plugin_table[idx];

    return ptr->plugin_name.c_str();
}
