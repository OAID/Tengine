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
 * Author: cmeng@openailab.com
 */
#include <iostream>
#include <map>
#include <vector>
#include <fstream>
#include <string>
#include <regex>
#include "tengine_log.h"
#include "nnie_model_cfg.hpp"

namespace Tengine {

NnieCpuNodes::NnieCpuNodes(std::string filepath)
{
    nodeStart = false;
    std::string line, key, value;
    std::ifstream infile;
    infile.open(filepath.c_str());
    if(!infile)
    {
        XLOG(LOG_ERR,"error:enable open file:%s\n", filepath.c_str());
        return;
    }
    else
    {
        TLOG_INFO( "open file:%s\n success\n" , filepath.c_str());
    }
    while(infile >> line)
    {
        if(line.find("cpu_node") != std::string::npos)
        {
            nodeStart = true;
            trans_map = new NnieCpuNode();
        }
        else if(line.find("}") != std::string::npos && nodeStart == true)
        {
            /* code */
            nodeStart = false;
            nodes.push_back(*trans_map);
        }
        else if(nodeStart == true)
        {
            std::vector<std::string> parts = s_split(line, ":");
            std::string key = parts[0];
            std::string value = parts[1];
            trans_map->insert(key, value);
        }
    }
    infile.close();
}

std::vector<std::string> NnieCpuNodes::s_split(const std::string& in, const std::string& delim)
{
    std::regex re{delim};
    return std::vector<std::string>{std::sregex_token_iterator(in.begin(), in.end(), re, -1),
                                    std::sregex_token_iterator()};
}

}    // namespace Tengine
