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
#include "tengine_config.hpp"

namespace TEngine {

using ConfManager = Attribute;

const std::string TEngineConfig::version("0.1.2");

bool TEngineConfig::tengine_mt_mode = true;
char TEngineConfig::delim_ch = '=';
char TEngineConfig::commt_ch = '#';

ConfManager * TEngineConfig::GetConfManager(void)
{
    static ConfManager instance;

    return &instance;
}

void TEngineConfig::Trim(std::string& s)
{
    static const char whitespaces[] = " \t\f\v\n\r";

    // Erase the leading whitespaces
    s.erase(0, s.find_first_not_of(whitespaces));
    // Erase the trailing whitespaces
    s.erase(s.find_last_not_of(whitespaces)+1);
}

bool TEngineConfig::Load(const std::string filename, const char delimiter, 
                                                     const char comment)
{
    std::fstream cfgfile(filename.c_str());
    if(!cfgfile)
    {
        LOG_ERROR()<<"Can not open the config file!\n";
        LOG_ERROR()<<"Please cp ./etc/config.example ./etc/config\n";
        return false;
    }

    // Set the seperators
    delim_ch = delimiter;
    commt_ch = comment;

    // Read the keys and values from the config file
    typedef std::string::size_type pos;
    ConfManager *manager=GetConfManager();
    while(!cfgfile.eof())
    {
        std::string line;
        getline(cfgfile, line);

        // Ignore comments
        line = line.substr(0, line.find(commt_ch));
        if(!line.length())
            continue;

        // Parse the line if it contains a delimiter
        pos delim_pos = line.find(delim_ch);
        if(delim_pos != std::string::npos)
        {
            // Get the key and the value
            std::string key = line.substr(0, delim_pos);
            line.replace(0, delim_pos+1, "");
            // Remove the leading and trailing whitespaces
            Trim(key);
            Trim(line);
            // Store the key and the value, overwrites if the key is repeated
            if(key == "mt.mode" && line == "0")
            {
                tengine_mt_mode = false;
            }
            // Set the config manager
            manager->SetAttr(key, line);
        }
    }
    return true;
}

void TEngineConfig::Remove(const std::string& key)
{
    ConfManager *manager = GetConfManager();

    manager->RemoveAttr(key);
}

void TEngineConfig::DumpConfig(void)
{
    ConfManager *manager = GetConfManager();
   	std::vector<std::string> keys;

    keys = manager->ListAttr();
    std::string value;
    for(unsigned int i=0; i < keys.size(); i++)
    {
        Get<std::string>(keys.at(i), value);
        std::cout<<keys.at(i)<<" = "<<value<<std::endl;
    }
}

std::vector<std::string> TEngineConfig::ParseKey(const std::string& key)
{
   	std::vector<std::string> result;
    std::string subkey = key;

    typedef std::string::size_type pos;
    pos dot_pos = subkey.find('.');
    while(dot_pos != std::string::npos)
    {
        result.push_back(subkey.substr(0, dot_pos));
        subkey.replace(0, dot_pos+1, "");
        dot_pos = subkey.find('.');
    }
   	if(subkey.size() > 0)
        result.push_back(subkey);

   	return result;
}

} //end of namespace TEngine
