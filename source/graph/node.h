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

#include "defines.h"

#include "operator/op.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct node;
struct tensor;
struct graph;

/*!
 * @struct ir_node_t
 * @brief  Abstract node intermediate representation
 */
typedef struct node
{
    uint16_t index;        //!< the index of a node
    uint8_t dynamic_shape; //!< flag of dynamic shape
    uint8_t input_num;     //!< count of input tensor
    uint8_t output_num;    //!< count of output tensor
    uint8_t node_type;     //!< type of node: { input, output, intermediate }
    int8_t subgraph_idx;   //!< id of the owner subgraph

    uint16_t* input_tensors;  //!< id array of input tensor
    uint16_t* output_tensors; //!< id array of output tensor

    char* name; //!< name of a node

    struct op op;        //!< operator of a node
    struct graph* graph; //!< pointer of the owner graph
} ir_node_t;

/*!
 * @brief Create a node for a graph.
 *
 * @param [in]  ir_graph: specific graph.
 * @param [in]  node_name: node name.
 * @param [in]  op_type: operator type, see "op.h".
 * @param [in]  op_version: operator defined version.
 *
 * @return  The pointer of the node.
 */
ir_node_t* create_ir_node(struct graph* ir_graph, const char* node_name, int op_type, int op_version);

/*!
 * @brief Destroy a node.
 *
 * User should deal with other destroy works, such as ir_node and ir_tensor.
 *
 * @param [in]  ir_graph: specific graph.
 * @param [in]  ir_node: the tensor pointer.
 */
void destroy_ir_node(struct graph* ir_graph, ir_node_t* ir_node);

/*!
 * @brief  Set node name from id, for anonymity ones.
 *
 * @param [in]  index: reference id.
 *
 * @return char array of the name.
 */
char* create_ir_node_name_from_index(int index);

/*!
 * @brief  Get node id from name, for anonymity ones.
 *
 *   It is possible to match the wrong node if the suffix is a digital
 * while the corresponding node has no name string.
 * But we leave this to the graph creator to avoid such case.
 *
 * @param [in]  ir_graph: specific graph.
 * @param [in]  node_name: reference name.
 *
 * @return node id.
 */
int get_ir_node_index_from_name(struct graph* ir_graph, const char* node_name);

/*!
 * @brief  Mark a tensor as node a specific input tensor.
 *
 * @param [in]  ir_node: specific node.
 * @param [in]  input_idx: input rank in all node input tensors.
 * @param [in]  tensor: pointer of the tensor.
 *
 * @return statue value, 0 success, other value failure.
 */
int set_ir_node_input_tensor(ir_node_t* ir_node, int input_idx, struct tensor* tensor);

/*!
 * @brief  Mark a tensor as node a specific output tensor.
 *
 * @param [in]  node: specific node.
 * @param [in]  output_idx: output rank in all node output tensors.
 * @param [in]  tensor: pointer of the tensor.
 *
 * @return statue value, 0 success, other value failure.
 */
int set_ir_node_output_tensor(ir_node_t* ir_node, int output_idx, struct tensor* tensor);

/*!
 * @brief  Dump the node.
 *
 * @param [in]  graph: specific graph.
 * @param [in]  node: specific node.
 */
void dump_ir_node(struct graph* ir_graph, ir_node_t* ir_node);

#ifdef __cplusplus
}
#endif /* __cplusplus */
