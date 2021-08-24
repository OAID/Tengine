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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: bzhang@openailab.com
 */

#ifndef __NCNN2TENGINE_HPP__
#define __NCNN2TENGINE_HPP__

#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <map>
#include <vector>
#include <unordered_map>
#include <stdio.h>
#include <set>
#include <algorithm>

extern "C" {
#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "graph/node.h"
#include "graph/tensor.h"
#include "executer/executer.h"
#include "module/module.h"
#include "utility/log.h"
#include "utility/sys_port.h"
#include "utility/vector.h"
#include "save_graph/op_include.h"
}
#define NCNN_MAX_PARAM_COUNT 32

struct NcnnNode
{
    std::string op;
    std::string name;
    int optimized;
    std::map<int, std::string> attrs;
    std::map<int, std::vector<std::string> > opt_attrs;
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

class ncnn_serializer
{
public:
    graph_t ncnn2tengine(std::string params_file, std::string bin_file);
    typedef int (*op_load_t)(ir_graph_t* graph, ir_node_t* node, const NcnnNode& ncnn_node);
    typedef std::map<int, std::string>::const_iterator const_iterator;

private:
    std::unordered_map<std::string, std::pair<int, op_load_t> > op_load_map;
    int load_model(ir_graph_t* graph, std::string params_file, std::string bin_file);
    int set_graph_input(ir_graph_t* graph, const std::vector<NcnnNode>& nodelist, const std::vector<NcnnParam>& paramlist);
    int load_constant_tensor(ir_graph_t* graph, const std::vector<NcnnNode>& nodelist, const std::vector<NcnnParam>& paramlist);
    int load_binary_file(const char* fname, std::vector<NcnnParam>& paramlist, std::vector<NcnnNode>& nodelist);
    int load_model_file(const char* fname, std::vector<NcnnNode>& nodelist);
    int load_graph_node(ir_graph_t* graph, const std::vector<NcnnNode>& nodelist, const std::vector<NcnnParam>& paramlist);
    bool find_op_load_method(const std::string& op_name);
    int read(void* buf, int size);
    ir_tensor_t* find_tensor(ir_graph_t* graph, const std::string& tensor_name);
    void register_op_load();
    int set_graph_output(ir_graph_t* graph, const std::vector<NcnnNode>& nodelist, const std::vector<NcnnParam>& paramlist);

    FILE* fp;
    struct
    {
        int loaded;
        union
        {
            int i;
            float f;
        };
        float* f_data;
        int* i_data;
        float* f_data_array;
        int* i_data_array;
    } params[NCNN_MAX_PARAM_COUNT];
};

#endif