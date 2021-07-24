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

#include "optimizer/helper.h"

#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "operator/op.h"


int is_index_in_array(const uint16_t* array, const uint16_t array_size, const uint16_t index)
{
    for (uint16_t i = 0; i < array_size; i++)
    {
        const uint16_t selected_index = array[i];

        if (selected_index == index)
        {
            return 1;
        }
    }

    return 0;
}


int is_subgraph_input_tensor(const struct subgraph* subgraph, const uint16_t tensor_index)
{
    return is_index_in_array(subgraph->input_tensor_list, (uint16_t)subgraph->input_num, tensor_index);
}


int is_subgraph_output_tensor(const struct subgraph* subgraph, const uint16_t tensor_index)
{
    return is_index_in_array(subgraph->output_tensor_list, (uint16_t)subgraph->input_num, tensor_index);
}


int is_variable_tensor_in_subgraph(const ir_subgraph_t* subgraph, const uint16_t tensor_index)
{
    // only each node outputs need to be checked next
    for (uint16_t i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_index = subgraph->node_list[i];
        ir_node_t* node = get_ir_graph_node(subgraph->graph, node_index);

        if (OP_CONST != node->op.type && is_index_in_array(node->output_tensors, (uint16_t)node->output_num, tensor_index))
        {
            return 1;
        }
    }

    return 0;
}
