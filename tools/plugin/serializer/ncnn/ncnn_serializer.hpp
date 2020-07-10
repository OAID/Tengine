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
 * Copyright (c) 2019, Open AI Lab
 * Author: bzhang@openailab.com
 */
#ifndef __NCNN_SERIALIZER_HPP__
#define __NCNN_SERIALIZER_HPP__

#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <map>
#include <vector>

#include "serializer.hpp"
#include "static_graph_interface.hpp"
#include "logger.hpp"

#define NCNN_MAX_PARAM_COUNT 32
namespace TEngine {

struct NcnnNode
{
    std::string op;
    std::string name;
    int optimized;
    std::map<int, std::string> attrs;
    std::map<int, std::vector<std::string>> opt_attrs;
    //std::vector<int> inputs;
    std::vector<std::string> inputs_name;
    std::vector<std::string> output_name;
};

struct NcnnParam
{
    int dim_size;
    int data_len;
    std::string name;
    std::vector<int> dims;
    void* data;
};



class NcnnSerializer : public Serializer
{
public:
    NcnnSerializer()
    {
        name_ = "ncnn_loader";
        version_ = "0.1";
        format_name_ = "ncnn";
    }
    virtual ~NcnnSerializer() {}

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
    bool LoadTextFile(const char* fname, std::vector<NcnnNode>& nodelist);
    bool LoadBinaryFile(const char* fname, std::vector<NcnnParam>& paramlist, std::vector<NcnnNode>& nodelist);

    bool LoadGraph(StaticGraph* graph, const std::vector<NcnnNode>& nodelist,
                   const std::vector<NcnnParam>& paramlist);

    void LoadNode(StaticGraph* graph, StaticNode* node,const NcnnNode& ncnn_node,
                  const std::vector<NcnnNode>& nodelist,const std::vector<NcnnParam>& paramlist);
    //bool LoadConstNode(StaticGraph* graph, const std::vector<NcnnNode>& nodelist,
    //                    const std::vector<NcnnParam>& paramlist);
    bool LoadConstTensor(StaticGraph* graph, const std::vector<NcnnNode>& nodelist,
                         const std::vector<NcnnParam>& paramlist);
    void CreateInputNode(StaticGraph* graph, const std::vector<NcnnNode>& nodelist,
                         const std::vector<NcnnParam>& paramlist);
    bool vstr_is_float(const char vstr[16]);

    int read(void* buf, int size);

    struct
    {
        int loaded;
        union { int i; float f; };
        float* f_data;
        int* i_data;
        float* f_data_array;
        int* i_data_array;
    } params[NCNN_MAX_PARAM_COUNT];
    
    FILE* fp;
};


}    // namespace TEngine

#endif
