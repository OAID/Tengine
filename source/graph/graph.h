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
 * Author: haitao@openailab.com
 * Revised: lswang@openailab.com
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct context;
struct node;
struct tensor;
struct device;
struct attribute;

/*!
 * @struct ir_graph_t
 * @brief  Abstract graph intermediate representation
 */
typedef struct graph
{
    struct tensor** tensor_list; //!< the tensor list of a graph
    struct node** node_list;     //!< the node list of a graph
    int16_t* input_nodes;        //!< input nodes index array of a graph
    int16_t* output_nodes;       //!< output nodes index array of a graph

    uint16_t tensor_num; //!< the count of all graph tensor
    uint16_t node_num;   //!< the count of all graph node
    uint16_t input_num;  //!< input nodes index count of a graph
    uint16_t output_num; //!< input nodes index count of a graph

    int8_t graph_layout; //!< the data layout of a graph
    int8_t model_layout; //!< model layout of graph source model
    int8_t model_format; //!< model format of graph source model

    uint8_t status; //!< the status of graph

    struct serializer* serializer; //!< serializer of graph
    void* serializer_privacy;      //!< privacy data of serializer

    struct device* device; //!< assigned nn_device for this graph
    void* device_privacy;  //!< privacy data of device

    struct attribute* attribute; //<! attribute of graph

    struct vector* subgraph_list; //!< subgraph list of this graph
} ir_graph_t;

/*!
 * @brief Create a graph.
 *
 * @param [in]  context: specific context for this graph.
 *
 * @return  The pointer of the graph.
 */
struct graph* create_ir_graph(struct context* context);

/*!
 * @brief Init a graph.
 *
 * @param [in]  graph: specific graph.
 * @param [in]  context: specific context for this graph.
 */
void init_ir_graph(ir_graph_t* graph, struct context* context);

/*!
 * @brief Destroy a graph.
 *
 * User should deal with other destroy works, such as ir_tensor and ir_node.
 *
 * @param [in]  graph: specific graph.
 */
void destroy_ir_graph(ir_graph_t* graph);

/*!
 * @brief Set input nodes for specific graph.
 *
 * Notice: This function will release older input nodes directly.
 *
 * @param [in]  graph: specific graph.
 * @param [in]  input_nodes: specific graph.
 * @param [in]  input_number: specific graph.
 *
 * @return statue value, 0 success, other value failure.
 */
int set_ir_graph_input_node(ir_graph_t* graph, int16_t input_nodes[], int input_number);

/*!
 * @brief Set output nodes for specific graph.
 *
 * Notice: This function will release older output nodes directly.
 *
 * @param [in]  graph: specific graph.
 * @param [in]  output_nodes: specific graph.
 * @param [in]  output_number: specific graph.
 *
 * @return statue value, 0 success, other value failure.
 */
int set_ir_graph_output_node(ir_graph_t* graph, int16_t output_nodes[], int output_number);

/*!
 * @brief Get specific tensor for a graph.
 *
 * @param [in]  graph: specific graph.
 * @param [in]  index: index of specific tensor.
 *
 * @return  The pointer of the tensor.
 */
struct tensor* get_ir_graph_tensor(ir_graph_t* graph, int index);

/*!
 * @brief Get specific node for a graph.
 *
 * @param [in]  graph: specific graph.
 * @param [in]  index: index of specific node.
 *
 * @return  The pointer of the node.
 */
struct node* get_ir_graph_node(ir_graph_t* graph, int index);

/*!
 * @brief Get output subgraph for a graph.
 *
 * @param [in]  graph: specific graph.
 * @param [in]  index: index of specific subgraph.
 *
 * @return  The pointer of the subgraph.
 */
struct subgraph* get_ir_graph_subgraph(ir_graph_t* graph, int index);

/*!
 * @brief Infer each node shape for a graph.
 *
 * @param [in]  graph: specific graph.
 *
 * @return statue value, 0 success, other value failure.
 */
int infer_ir_graph_shape(ir_graph_t* graph);

/*!
 * @brief  Dump the graph.
 *
 * @param [in]  ir_graph: specific graph.
 */
void dump_ir_graph(ir_graph_t* graph);

#ifdef __cplusplus
}
#endif
