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

struct graph;
struct device;

/*!
 * @struct ir_subgraph_t
 * @brief  Abstract subgraph intermediate representation
 */
typedef struct subgraph
{
    uint8_t index;             //!< the index of a subgraph
    uint8_t input_ready_count; //!< the count of all ready input tensors
    uint8_t input_wait_count;  //!< the count of all input tensors that are not ready
    uint8_t input_num;         //!< the count of input tensors
    uint8_t output_num;        //!< the count of output tensors
    uint8_t status;            //!< the execution status of subgraph

    uint16_t node_num;   //!< the count of nodes in subgraph
    uint16_t* node_list; //!< all nodes index list of subgraph

    uint16_t* input_tensor_list;  //!< input tensors index list of subgraph
    uint16_t* output_tensor_list; //!< output tensors index list of subgraph

    struct graph* graph; //!< the pointer of the related graph

    struct device* device; //!< the device which will the subgraph running on
    void* device_graph;    //!< the related device graph
} ir_subgraph_t;

/*!
 * @brief Init a subgraph.
 *
 * @param [in]  graph: specific graph.
 * @param [in]  subgraph: specific subgraph.
 * @param [in]  index: index of specific subgraph.
 */
void init_ir_subgraph(struct graph* graph, ir_subgraph_t* subgraph, int index);

/*!
 * @brief Release a subgraph.
 *
 * @param [in]  graph: specific graph.
 * @param [in]  subgraph: specific subgraph.
 */
void release_ir_subgraph(struct graph* ir_graph, ir_subgraph_t* ir_subgraph);
