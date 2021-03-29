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

#ifndef __NNIE_MODEL_HPP__
#define __NNIE_MODEL_HPP__

#include <iostream>
#include <map>
#include <vector>
#include <fstream>
#include <string>
#include <regex>

#include "tengine_c_api.h"

namespace Tengine {
class NnieCpuNode
{
public:
    NnieCpuNode(){};
    void insert(std::string key, std::string value)
    {
        trans_map.insert(make_pair(key, value));
        return;
    };
    std::string getName(){
        return trans_map["name"];
    };
    int getInputNum()
    {
        return atoi(trans_map["inputnum"].c_str());
    };
    std::string getInputTensorName(int index)
    {
        return trans_map["input_" + std::to_string(index)];
    };

    int getOutputNum()
    {
        return atoi(trans_map["outputnum"].c_str());
    };

    std::string getOutputTensorName(int index)
    {
       return trans_map["output_" + std::to_string(index)];
    };


private:
    std::map<std::string, std::string> trans_map;
};

class NnieCpuNodes
{
public:
    NnieCpuNodes(std::string filepath);
    ~NnieCpuNodes(){};
    int getnode_size(){
        return nodes.size();
    };
    NnieCpuNode getnode(int index)
    {
        return nodes[index];
    }

private : 
    bool nodeStart;
    std::vector<std::string> s_split(const std::string& in, const std::string& delim);
    NnieCpuNode* trans_map;
    std::vector<NnieCpuNode> nodes;
};

}// namespace Tengine

#endif    //__NNIE_MODEL_HPP__
