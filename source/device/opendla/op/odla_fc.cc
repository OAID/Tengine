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
 * Author: hhchen@openailab.com
 */

#include "odla_executor.hpp"

extern "C"
{
#include "operator/op.h"
#include "convolution_param.h"
}


bool ODLAEngine::AddFullyConnectionNode(struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    struct subgraph* subgraph = get_ir_graph_subgraph(ir_graph, ir_node->subgraph_idx);

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    nvdla::priv::canonical_ast::Edge* inputEdge = new nvdla::priv::canonical_ast::Edge();
    nvdla::priv::canonical_ast::Edge* outputEdge = new nvdla::priv::canonical_ast::Edge();
    nvdla::priv::canonical_ast::Node* Node ;
    nvdla::priv::canonical_ast::FullyConnectedNode * fullyConnectedNode = new nvdla::priv::canonical_ast::FullyConnectedNode();


    fullyConnectedNode->setGraph(this->graph);
    this->graph->insertNode(fullyConnectedNode);
    fullyConnectedNode->setId(this->graph->nextNodeId());
    fullyConnectedNode->setName(ir_node->name);


    // Init Edge
    inputEdge->setGraph(this->graph);
    inputEdge->setId(graph->nextEdgeId());
    inputEdge->setOriginalTensor(odla_tensor_map[input_tensor->index]->clone());
    fullyConnectedNode->markInputEdge(inputEdge);
    this->graph->insertEdge(inputEdge);


    outputEdge->setGraph(this->graph);
    outputEdge->setId(graph->nextEdgeId());
    outputEdge->setOriginalTensor(odla_tensor_map[output_tensor->index]->clone());
    fullyConnectedNode->markOutputEdge(outputEdge);
    this->graph->insertEdge(outputEdge);

    // Second represents Input and First is Output
    this->graph->appendNodeToEdge(inputEdge, nvdla::priv::ast::EdgeSideEnum::SECOND, fullyConnectedNode);
    this->graph->appendNodeToEdge(outputEdge, nvdla::priv::ast::EdgeSideEnum::FIRST, fullyConnectedNode);

    // if the tensor is Graph Input or Output
    std::vector<nvdla::priv::canonical_ast::Edge *> inputEdges;
    std::vector<nvdla::priv::canonical_ast::Edge *> outputEdges;
    inputEdges.reserve(subgraph->input_num);
    outputEdges.reserve(subgraph->output_num);


    fullyConnectedNode->params().setBiasMode(nvdla::bNONE);
    fullyConnectedNode->params().setHasBiasTerm(false);


    nvdla::Weights kernelWeights;
    nvdla::Weights biasWeights;

    nvdla::Dims4 weightDims(weight_tensor->dims[0], weight_tensor->dims[1], weight_tensor->dims[2], weight_tensor->dims[3]);
    fullyConnectedNode->params().setWeightDims(weightDims);

    kernelWeights.count = weight_tensor->elem_num;
    kernelWeights.values = weight_tensor->data;
    kernelWeights.type = odla_tensor_map[weight_tensor->index]->getDataType();
    if (ir_node->input_num > 2)
    {
        struct tensor* bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        nvdla::Dims4 biasDims(odla_tensor_map[bias_tensor->index]->getDimensions());

        biasWeights.count = bias_tensor->elem_num;
        biasWeights.values = bias_tensor->data;
        biasWeights.type = odla_tensor_map[bias_tensor->index]->getDataType();

        fullyConnectedNode->params().setHasBiasTerm(true);
        if(bias_tensor->dim_num == 1) fullyConnectedNode->params().setBiasMode(nvdla::bCHANNEL);
        fullyConnectedNode->params().setBiasDims(biasDims);
    }

    fullyConnectedNode->params().setWeights(kernelWeights);
    fullyConnectedNode->params().setBiasData(biasWeights);

    // Insert priv pair
    Node = fullyConnectedNode;
    nvdla::priv::canonical_ast::NodeFactory::s_fc_priv.insert(
        std::pair<nvdla::priv::canonical_ast::Node*, nvdla::priv::canonical_ast::FullyConnectedNode*>(Node, fullyConnectedNode)
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
