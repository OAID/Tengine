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
#include "scale_param.h"
}


nvdla::priv::canonical_ast::Node * ODLAEngine::AddScaleNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct scale_param* param = (struct scale_param*)ir_node->op.param_mem;
    nvdla::priv::canonical_ast::Node * Node ;
    auto * scaleNode = new nvdla::priv::canonical_ast::ScaleNode();

    nvdla::Dims4 scaleDims, shiftDims, powerDims;
    if(param->num_axes == 0){
        scaleNode->params().setMode(nvdla::ScaleMode::sUNIFORM);
        scaleDims.c = 1;
        scaleDims.h = 1;
        scaleDims.w = 1;
    } else if(param->num_axes == 1){
        scaleNode->params().setMode(nvdla::ScaleMode::sCHANNEL);
        scaleDims.c = input_tensor->dims[1];
        scaleDims.h = 1;
        scaleDims.w = 1;
    } else if (param->num_axes == 3){
        scaleNode->params().setMode(nvdla::ScaleMode::sm_ELEMENTWISE);
        scaleDims.c = input_tensor->dims[1];
        scaleDims.h = input_tensor->dims[2];
        scaleDims.w = input_tensor->dims[3];
    }

    scaleNode->params().setHasBiasTerm(param->bias_term != 0);
    if (scaleNode->params().hasBiasTerm()) {
        shiftDims = scaleDims;
        scaleNode->params().setShiftDims(shiftDims);
    }

    Node = scaleNode;
    nvdla::priv::canonical_ast::NodeFactory::s_scale_priv.insert(
        std::pair<nvdla::priv::canonical_ast::Node*, nvdla::priv::canonical_ast::ScaleNode*>(Node, scaleNode)
    );
    return Node;
}
