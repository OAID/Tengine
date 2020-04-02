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
#ifndef __MXNET_SERIALIZER_HPP__
#define __MXNET_SERIALIZER_HPP__

#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <map>
#include <vector>

#include "serializer.hpp"
#include "static_graph_interface.hpp"
#include "logger.hpp"

namespace TEngine {

struct MxnetNode
{
    std::string op;
    std::string name;
    std::map<std::string, std::string> attrs;
    std::vector<int> inputs;
};

struct MxnetParam
{
    int dim_size;
    int data_len;
    std::string name;
    std::vector<int> dims;
    uint8_t* raw_data;
};

class MxnetSerializer : public Serializer
{
public:
    MxnetSerializer()
    {
        name_ = "mxnet_loader";
        version_ = "0.1";
        format_name_ = "mxnet";
    }
    virtual ~MxnetSerializer() {}

    unsigned int GetFileNum(void) override
    {
        return 2;
    }

    bool LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph) override;

    bool LoadConstTensor(const std::string& fname, StaticTensor* const_tensor) override
    {
        return false;
    }
    bool LoadConstTensor(int fd, StaticTensor* const_tensor) override
    {
        return false;
    }

protected:
    bool LoadTextFile(const char* fname, std::vector<MxnetNode>& nodelist);
    bool LoadBinaryFile(const char* fname, std::vector<MxnetParam>& paramlist);

    bool LoadGraph(StaticGraph* graph, const std::vector<MxnetNode>& nodelist,
                   const std::vector<MxnetParam>& paramlist);

    void LoadNode(StaticGraph* graph, StaticNode* node, const MxnetNode& mxnet_node,
                  const std::vector<MxnetNode>& nodelist);

    bool LoadConstTensor(StaticGraph* graph, const std::vector<MxnetNode>& nodelist,
                         const std::vector<MxnetParam>& paramlist);
    void CreateInputNode(StaticGraph* graph, const std::vector<MxnetNode>& nodelist,
                         const std::vector<MxnetParam>& paramlist);
};

}    // namespace TEngine

#endif
