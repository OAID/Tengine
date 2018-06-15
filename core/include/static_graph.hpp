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
 * Author: haitao@openailab.com
 */
#ifndef __STATIC_GRAPH_HPP__
#define __STATIC_GRAPH_HPP__

#include <string>
#include <vector>
#include <memory>

#include "attribute.hpp"
#include "safe_object_manager.hpp"
#include "tensor_shape.hpp"

namespace TEngine {


struct StaticGraph;
struct StaticNode;
struct StaticTensor;
struct StaticOp;

using StaticGraphPtr=std::shared_ptr<StaticGraph>;
using StaticNodePtr=std::shared_ptr<StaticNode>;
using StaticTensorPtr=std::shared_ptr<StaticTensor>;
using StaticOpPtr=std::shared_ptr<StaticOp>;


struct StaticGraph {
    std::string  model_name; //name assigned when load the model
    std::string  domain;    
    std::string  name;  //
    std::string  version;
    std::string  source;   //From where to load the model?
    std::string  source_format; 
    std::string  const_tensor_file; 
    Attribute    attrs;
    std::vector<int> input_node_list;
    std::vector<int> output_node_list;
    std::vector<StaticNodePtr> node_list;
    std::vector<StaticTensorPtr> tensor_list;
    std::unordered_map<std::string,StaticTensorPtr> const_tensor_map;
};


struct StaticNode {
    std::string name;
    int         index;
    StaticOpPtr op;

    std::vector<int> input_tensor_list;
    std::vector<int> output_tensor_list;

};

struct NodeSynapse {
   int   node_index;
   int   entry_index;
};

struct StaticTensor {
    std::string name;
    int         index;
    int         mem_size;
    std::vector<int> dims;
    std::string   data_type;
    std::string   data_layout;
    int           type;
    NodeSynapse   producer;
    std::vector<NodeSynapse> consumer; 
    virtual ~StaticTensor(){}
};

struct StaticConstTensor: public StaticTensor {
    void *  mem_addr;
    int     file_offset;
    int     file_size;

    StaticConstTensor() { mem_addr=nullptr; }
    
    virtual ~StaticConstTensor() { if(mem_addr) std::free(mem_addr);}
};


struct StaticOp {
    std::string name;
    bool        dynamic_shape;
    any         param;
    Attribute   attrs;
    StaticOp() { dynamic_shape=false;}
};


class StaticGraphManager : public SimpleObjectManagerWithLock<StaticGraphManager,StaticGraphPtr>{};


} //namespace TEngine



#endif
