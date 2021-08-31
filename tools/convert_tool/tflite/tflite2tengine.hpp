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
 * Author: qwang02@openailab.com
 */

#ifndef __TFLITE2TENGINE_HPP__
#define __TFLITE2TENGINE_HPP__

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include "flatbuffers/flexbuffers.h"
#include "schema_generated.h"

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

using TFLiteTensor = ::tflite::Tensor;
using TFLiteOperator = ::tflite::Operator;
using LiteModel = ::tflite::Model;

struct LiteTensor;

typedef struct LiteNode
{
    int idx;
    std::string op;
    std::string name;
    std::vector<struct LiteTensor*> inputs;
    std::vector<struct LiteTensor*> outputs;

    const TFLiteOperator* lite_op;

    ir_node_t* ir_node;
} LiteNode_t;

typedef struct LiteTensor
{
    int idx;
    std::string name;
    std::string type;
    std::vector<int> shape;

    ir_tensor_t* ir_tensor;
    struct LiteNode* producer;
    const TFLiteTensor* tf_tensor;
    bool graph_input;
    bool graph_output;

    LiteTensor(void)
    {
        tf_tensor = nullptr;
        ir_tensor = nullptr;
        producer = nullptr;
        graph_input = false;
        graph_output = false;
    }
} LiteTensor_t;

typedef struct LiteGraph
{
    std::vector<struct LiteNode*> seq_nodes;
    std::vector<struct LiteTensor*> input_tensors;
    std::vector<struct LiteTensor*> output_tensors;
    std::vector<struct LiteTensor*> tensor_list;

    const LiteModel* lite_model;

    ~LiteGraph(void)
    {
        for (auto node : seq_nodes)
            delete node;
    }
} LiteGraph_t;

class tflite_serializer
{
public:
    graph_t tflite2tengine(std::string model_file);
    typedef int (*op_load_t)(ir_graph_t* graph, ir_node_t* ir_node, LiteNode_t* lite_node);

private:
    std::unordered_map<std::string, std::pair<int, op_load_t> > op_load_map;
    int load_model(ir_graph_t* graph, std::string model_file);
    bool load_model_from_mem(char* mem_addr, int mem_size, ir_graph_t* graph);
    bool construct_graph(const LiteModel* lite_model, LiteGraph_t* lite_graph);
    bool optimize_graph(LiteGraph_t* lite_graph);
    bool find_op_load_method(const std::string& op_name);
    int get_lite_tensor_data_type(const std::string type);
    int load_tensor_scale_and_zero(ir_tensor_t* ir_tensor, LiteTensor_t* lite_tensor);

    int load_lite_tensor(ir_graph_t* graph, LiteGraph_t* lite_graph);
    int set_graph_input(ir_graph_t* graph, LiteGraph_t* lite_graph);
    int load_graph_node(ir_graph_t* graph, LiteGraph_t* lite_graph);
    int set_graph_output(ir_graph_t* graph);
    void register_op_load();

    std::vector<std::string> unsupport_op;
    std::vector<std::string> support_op;
};

#endif