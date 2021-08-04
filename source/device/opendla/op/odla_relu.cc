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
 * Copyright (c) 2021, Open AI Lab
 * Author: lswang@openailab.com
 */

#include "odla_executor.hpp"

extern "C"
{
#include "operator/op.h"
#include "relu_param.h"
}


bool ODLAEngine::AddReluNode(struct node* ir_node)
{
    int op_type = OP_RELU;

    struct graph* ir_graph = ir_node->graph;
    struct subgraph* subgraph = get_ir_graph_subgraph(ir_graph, ir_node->subgraph_idx);
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    nvdla::priv::canonical_ast::Edge * inputEdge = new nvdla::priv::canonical_ast::Edge();
    nvdla::priv::canonical_ast::Edge * outputEdge = new nvdla::priv::canonical_ast::Edge();
    nvdla::ActivationType activationType;
    nvdla::priv::canonical_ast::Node * Node ;
    nvdla::priv::canonical_ast::ActivationNode * activationNode = new nvdla::priv::canonical_ast::ActivationNode();

    if (OP_CONV == ir_node->op.type)
    {

        if (nullptr == input_tensor)
        {
            fprintf(stderr, "Tengine: Get input for ReLU(id: %d, name: %s) layer failed.\n", ir_node->index, ir_node->name);
            return false;
        }

        auto param = (struct conv_param*)ir_node->op.param_mem;

        switch (param->activation)
        {
            case 0:
                op_type = OP_RELU;
                break;
            default:
                fprintf(stderr, "Tengine: Unsupported RelU type(%d).\n", param->activation);
                return false;
        }

    }
    else
    {
        switch (ir_node->op.type)
        {
            case OP_RELU:
                op_type = OP_RELU;
                break;
            default:
                fprintf(stderr, "Tengine: Unsupported RelU type(%d).\n", ir_node->op.type);
                return false;
        }
    }

    activationNode->params().setActivationType(nvdla::ActivationType::kRELU);
    activationNode->setGraph(this->graph);
    this->graph->insertNode(activationNode);
    activationNode->setId(this->graph->nextNodeId());
    activationNode->setName(ir_node->name);

    // Init Edge
    inputEdge->setGraph(this->graph);
    inputEdge->setId(graph->nextEdgeId());
    inputEdge->setOriginalTensor(odla_tensor_map[input_tensor->index]->clone());
    activationNode->markInputEdge(inputEdge);
    this->graph->insertEdge(inputEdge);


    outputEdge->setGraph(this->graph);
    outputEdge->setId(graph->nextEdgeId());
    outputEdge->setOriginalTensor(odla_tensor_map[output_tensor->index]->clone());
    activationNode->markOutputEdge(outputEdge);
    this->graph->insertEdge(outputEdge);

    // Second represents Input and First is Output
    this->graph->appendNodeToEdge(inputEdge, nvdla::priv::ast::EdgeSideEnum::SECOND, activationNode);
    this->graph->appendNodeToEdge(outputEdge, nvdla::priv::ast::EdgeSideEnum::FIRST, activationNode);

    // if the tensor is Graph Input or Output
    std::vector<nvdla::priv::canonical_ast::Edge *> inputEdges;
    std::vector<nvdla::priv::canonical_ast::Edge *> outputEdges;
    inputEdges.reserve(subgraph->input_num);
    outputEdges.reserve(subgraph->output_num);
    // Insert priv pair
    Node = activationNode;
    nvdla::priv::canonical_ast::NodeFactory::s_act_priv.insert(
            std::pair<nvdla::priv::canonical_ast::Node*, nvdla::priv::canonical_ast::ActivationNode*>(Node, activationNode)
    );
    if(subgraph->input_tensor_list[0] == ir_node->input_tensors[0]){

        inputEdges.push_back(inputEdge);
        this->graph->setInputEdges(inputEdges);
    }
    if(subgraph->output_tensor_list[0] == ir_node->output_tensors[0]){

        outputEdges.push_back(outputEdge);
        this->graph->setOutputEdges(outputEdges);
    }

    return true;
}
