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
 * Author: xlchen@openailab.com
 */

#ifndef __DARKNET2TENGINE_HPP__
#define __DARKNET2TENGINE_HPP__

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <set>
#include <algorithm>

#include "te_darknet.hpp"
#include "../utils/graph_optimizer/graph_opt.hpp"

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

class darknet_serializer
{
public:
    graph_t darknet2tengine(std::string model_file, std::string proto_file);
    typedef int (*op_load_t)(ir_graph_t* graph, ir_node_t* node, std::vector<std::string>& tensor_name_map,
                             list* options, int index, FILE* fp);

private:
    std::unordered_map<std::string, std::pair<int, op_load_t> > op_load_map;
    int load_model(ir_graph_t* graph, std::string model_file, std::string proto_file);
    int load_weight_file(ir_graph_t* graph, const char* weight_file, list* sections);
    int set_graph_output(ir_graph_t* graph);
    void register_op_load();
};

#endif