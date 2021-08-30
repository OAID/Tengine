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
 * Copyright (c) 2021, Institute of Computing Technology
 * Author: wanglei21c@mails.ucas.ac.cn
 */

#include "odla_executor.hpp"

extern "C"
{
#include "operator/op.h"
#include "concat_param.h"
}


nvdla::priv::canonical_ast::Node * ODLAEngine::AddConcatNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct concat_param* param = (struct concat_param*)ir_node->op.param_mem;

    nvdla::priv::canonical_ast::Node * Node ;
    auto * concatenationNode = new nvdla::priv::canonical_ast::ConcatenationNode();
    concatenationNode->params().setNumInputs(ir_node->input_num);

    Node = concatenationNode;
    nvdla::priv::canonical_ast::NodeFactory::s_concat_priv.insert(
        std::pair<nvdla::priv::canonical_ast::Node*, nvdla::priv::canonical_ast::ConcatenationNode*>(Node, concatenationNode)
    );

    return Node;
}
